import logging
from types import MethodType
import torch as th
from torch import einsum
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np
import copy
import pdb
from torch.nn import Identity

from qdiff.quant_layer import UniformAffineQuantizer, StraightThrough, QuantModule_one
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, ResBlock, TimestepBlock, checkpoint
from ldm.modules.diffusionmodules.openaimodel import QKMatMul, SMVMatMul
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.attention import exists, default

from ddim.models.diffusion import ResnetBlock, AttnBlock, nonlinearity
from guided_diffusion.unet import ResBlock as AutoResBlock
from guided_diffusion.unet import AttentionBlock as AutoAttentionBlock
# from guided_diffusion.unet import QKVAttention as AutoQKVAttention

from guided_diffusion.unet import QKVAttention as guidedQKVAttention
from guided_diffusion.unet import TimestepBlock as guidedTimestepBlock
from guided_diffusion.unet import QKVAttentionLegacy as guidedQKVAttentionLegacy
from guided_diffusion.dynamic_unet import QKVAttention as DynamicQKVAttention
from guided_diffusion.dynamic_unet import ResBlock as DynamicResBlock
from guided_diffusion.dynamic_unet import AttentionBlock as DynamicAttentionBlock
from ldm.modules.diffusionmodules.openaimodel import QKVAttentionLegacy

logger = logging.getLogger(__name__)

        
def count_flops_attn(model, _x, y):#==hook function
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])

class BaseQuantBlock_one(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params_list):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.use_quantizer_idx=0
        self.act_quantizer_list=nn.ModuleList()
        for i in range(len(act_quant_params_list)):
            self.act_quantizer_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False
        self.split_quantization = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                m.set_quant_state(weight_quant, act_quant)
    
    def set_quant_idx(self,idx):
        self.use_quantizer_idx=idx


class QuantResBlock_one(BaseQuantBlock_one, TimestepBlock):##in lsun_church
    def __init__(
        self, res: ResBlock, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self, x, emb=None, split=0):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and not isinstance(self.skip_connection, Identity) and self.skip_connection.split == 0:
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )  

    def _forward(self, x, emb, split=0):
        # print(f"x shape {x.shape} emb shape {emb.shape}")
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


class QuantQKMatMul_one(BaseQuantBlock_one):#QKMatmul, in churches
    def __init__(
        self, act_quant_params_list):
        super().__init__(act_quant_params_list)
        self.scale = None
        self.use_act_quant = False
        self.act_quantizer_q_list = nn.ModuleList()
        self.act_quantizer_k_list = nn.ModuleList()
        for i in range(len(act_quant_params_list)):
            self.act_quantizer_q_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.act_quantizer_k_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
    
    def forward(self, q, k):
        if self.split_quantization == False:
            return self.forward_not_split(q, k)
        else:
            return self.forward_split(q, k)
    def forward_not_split(self, q, k):
        idx=self.use_quantizer_idx
        if self.use_act_quant:
            quant_q = self.act_quantizer_q_list[idx](q * self.scale)
            quant_k = self.act_quantizer_k_list[idx](k * self.scale)
            weight = th.einsum(
                    "bct,bcs->bts", quant_q, quant_k
                )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight
    def forward_split(self, q, k):
        q_idx=self.q_quantizer_idx
        k_idx =self.k_quantizer_idx
        if self.use_act_quant:
            quant_q = self.act_quantizer_q_list[q_idx](q * self.scale)
            quant_k = self.act_quantizer_k_list[k_idx](k * self.scale)
            weight = th.einsum(
                    "bct,bcs->bts", quant_q, quant_k
                )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant
    
    def set_quant_idx(self, idx, idx2 = None):
        if self.split_quantization == False:
            assert(idx2 == None)
            self.use_quantizer_idx = idx
        else:
            assert(idx2 != None)
            self.q_quantizer_idx = idx
            self.k_quantizer_idx = idx2
    ##def set_quant_whether_6(self, whether_6):
    #    self.whether_6=whether_6

class QuantSMVMatMul_one(BaseQuantBlock_one):#SMVMatMul, in churches
    def __init__(
        self, act_quant_params_list,sm_abit=8):
        super().__init__(act_quant_params_list)
        self.use_act_quant = False
        self.act_quantizer_v_list=nn.ModuleList()
        self.act_quantizer_w_list=nn.ModuleList()
        for i in range(len(act_quant_params_list)):
            self.act_quantizer_v_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            # act_quant_params_w = act_quant_params_list[i].copy()
            act_quant_params_w = copy.deepcopy(act_quant_params_list[i])
            act_quant_params_w['n_bits'] = sm_abit
            act_quant_params_w['symmetric'] = False
            act_quant_params_w['always_zero'] = True
            self.act_quantizer_w_list.append(UniformAffineQuantizer(**act_quant_params_w))
    
    def forward(self, weight, v):
        if self.split_quantization == False:
            return self.forward_not_split(weight, v)
        else:
            return self.forward_split(weight, v)
    def forward_not_split(self, weight, v):
        idx=self.use_quantizer_idx
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w_list[idx](weight), self.act_quantizer_v_list[idx](v))
            
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a
    
    def forward_split(self, weight, v):
        w_idx=self.w_quantizer_idx
        v_idx = self.v_quantizer_idx
        if self.use_act_quant:
            a = th.einsum("bts,bcs->bct", self.act_quantizer_w_list[w_idx](weight), self.act_quantizer_v_list[v_idx](v))
            
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_act_quant = act_quant
    def set_quant_idx(self, idx, idx2 = None):
        if self.split_quantization == False:
            assert(idx2 == None)
            self.use_quantizer_idx = idx
        else:
            assert(idx2 != None)
            self.v_quantizer_idx = idx
            self.w_quantizer_idx = idx2
    #def set_quant_state(self, whether_6):
    #    self.whether_6 = whether_6

class QuantAttentionBlock_one(BaseQuantBlock_one): #in churches
    def __init__(
        self, attn: AttentionBlock, act_quant_params_list):
        super().__init__(act_quant_params_list)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv
        
        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self.attention.qkv_matmul.set_quant_state(weight_quant, act_quant)
        self.attention.smv_matmul.set_quant_state(weight_quant, act_quant)
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                m.set_quant_state(weight_quant, act_quant)

    def set_quant_idx(self, idx):
        self.use_quantizer_idx = idx
        self.attention.qkv_matmul.set_quant_idx(idx)
        self.attention.smv_matmul.set_quant_idx(idx)
        for m in self.modules():
            if isinstance(m,QuantModule_one):
                m.set_quant_idx(idx)
    
    def set_quant_idx_cand(self, idx,idx2,idx3,idx4):
        self.attention.qkv_matmul.set_quant_idx(idx,idx2)
        self.attention.smv_matmul.set_quant_idx(idx3,idx4)
    
    def check_quant_state(self, weight_quant, act_quant):
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                if m.use_weight_quant != weight_quant or m.use_act_quant != act_quant:
                    print('error')
        if self.attention.qkv_matmul.use_act_quant != act_quant or self.attention.smv_matmul.use_act_quant != act_quant:
            print('error')

    def check_quant_idx(self, idx):
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                if m.use_quantizer_idx != idx:
                    print('error')
        if self.attention.qkv_matmul.use_quantizer_idx != idx or self.attention.smv_matmul.use_quantizer_idx != idx:
            print('error')
    

    
def cross_attn_forward_one(self, x, context=None, mask=None):
    h = self.heads
    if self.split_quantization == True:
        q_idx=self.q_quantizer_idx
        k_idx=self.k_quantizer_idx
        v_idx=self.v_quantizer_idx
        w_idx=self.w_quantizer_idx
    else:
        q_idx=k_idx=v_idx=w_idx=self.use_quantizer_idx
    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_act_quant:
        quant_q = self.act_quantizer_q_list[q_idx](q)
        quant_k = self.act_quantizer_k_list[k_idx](k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    if self.use_act_quant:
        out = einsum('b i j, b j d -> b i d', self.act_quantizer_w_list[w_idx](attn), self.act_quantizer_v_list[v_idx](v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

class QuantBasicTransformerBlock_one(BaseQuantBlock_one):#not in churches, in txt2img,img
    def __init__(
        self, tran: BasicTransformerBlock, act_quant_params_list,
        sm_abit: int = 8):
        super().__init__(act_quant_params_list)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2
        
        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint = tran.checkpoint
        self.whether_6=False

        # logger.info(f"quant attn matmul")
        self.attn1.act_quantizer_q_list=nn.ModuleList()
        self.attn1.act_quantizer_k_list=nn.ModuleList()
        self.attn1.act_quantizer_v_list=nn.ModuleList()
        self.attn1.act_quantizer_w_list=nn.ModuleList()
        self.attn2.act_quantizer_q_list=nn.ModuleList()
        self.attn2.act_quantizer_k_list=nn.ModuleList()
        self.attn2.act_quantizer_v_list=nn.ModuleList()
        self.attn2.act_quantizer_w_list=nn.ModuleList()
        for i in range(len(act_quant_params_list)):
            self.attn1.act_quantizer_q_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.attn1.act_quantizer_k_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.attn1.act_quantizer_v_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))

            self.attn2.act_quantizer_q_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.attn2.act_quantizer_k_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.attn2.act_quantizer_v_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
        
            act_quant_params_w = act_quant_params_list[i].copy()
            act_quant_params_w['n_bits'] = sm_abit
            act_quant_params_w['always_zero'] = True
            self.attn1.act_quantizer_w_list.append(UniformAffineQuantizer(**act_quant_params_w))
            self.attn2.act_quantizer_w_list.append(UniformAffineQuantizer(**act_quant_params_w))
        self.attn1.split_quantization = False
        self.attn2.split_quantization = False
        self.attn1.forward = MethodType(cross_attn_forward_one, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward_one, self.attn2)
        self.attn1.use_act_quant = False
        self.attn2.use_act_quant = False
        self.split_quantization=False
    
    def forward(self, x, context=None):
        # print(f"x shape {x.shape} context shape {context.shape}")
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        if context is None:
            assert(len(x) == 2)
            x, context = x
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
    
    def set_quant_idx(self, idx, idx2=None, idx3=None, idx4=None, idx5=None, idx6=None, idx7=None, idx8=None):
        if self.split_quantization == False:
            assert(idx2 == None)
            assert(idx3 == None)
            assert(idx4 == None)
            assert(idx5 == None)
            assert(idx6 == None)
            assert(idx7 == None)
            assert(idx8 == None)
            self.attn1.use_quantizer_idx=idx
            self.attn1.q_quantizer_idx=idx
            self.attn1.k_quantizer_idx=idx
            self.attn1.v_quantizer_idx=idx
            self.attn1.w_quantizer_idx=idx

            self.attn2.use_quantizer_idx=idx
            self.attn2.q_quantizer_idx=idx
            self.attn2.k_quantizer_idx=idx
            self.attn2.v_quantizer_idx=idx
            self.attn2.w_quantizer_idx=idx
            for m in self.modules():
                if isinstance(m, QuantModule_one):
                    m.set_quant_idx(idx)
        else:
            assert(idx2 != None)
            assert(idx3 != None)
            assert(idx4 != None)
            assert(idx5 != None)
            assert(idx6 != None)
            assert(idx7 != None)
            assert(idx8 != None)
            self.attn1.q_quantizer_idx=idx
            self.attn1.k_quantizer_idx=idx2
            self.attn1.v_quantizer_idx=idx3
            self.attn1.w_quantizer_idx=idx4

            self.attn2.q_quantizer_idx=idx5
            self.attn2.k_quantizer_idx=idx6
            self.attn2.v_quantizer_idx=idx7
            self.attn2.w_quantizer_idx=idx8
            

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.attn1.use_act_quant = act_quant
        self.attn2.use_act_quant = act_quant

        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                m.set_quant_state(weight_quant, act_quant)
    
    def check_quant_state(self, weight_quant, act_quant):
        for m in self.modules():
            if isinstance(m,QuantModule_one):
                if m.use_weight_quant!=weight_quant or m.use_act_quant!=act_quant:
                    print('error')
        if self.attn1.use_act_quant != act_quant or self.attn2.use_act_quant != act_quant:
            print('error')
    
    def check_quant_idx(self, idx):
        for m in self.modules():
            if isinstance(m, QuantModule_one):
                if m.use_quantizer_idx != idx:
                    print('error')
        if self.attn1.use_quantizer_idx != idx or self.attn2.use_quantizer_idx != idx:
            print('error')

class QuantResnetBlock_one(BaseQuantBlock_one):#in first_stage model, but not in unet, and different from that in first_stage(see parameters)
    def __init__(
        self, res: ResnetBlock, act_quant_params_list):
        super().__init__(act_quant_params_list)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut

    def forward(self, x, temb=None, split=0):
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x, split=split)
        out = x + h
        return out

class QuantAttnBlock_one(BaseQuantBlock_one):#一样，只是在first_stage里
    def __init__(
        self, attn: AttnBlock, act_quant_params_list, not_attn, sm_abit=8):
        super().__init__(act_quant_params_list)
        self.in_channels = attn.in_channels
        self.not_attn = not_attn
        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out
        self.act_quantizer_q_list=nn.ModuleList()
        self.act_quantizer_k_list=nn.ModuleList()
        self.act_quantizer_v_list=nn.ModuleList()
        self.act_quantizer_w_list=nn.ModuleList()
        for i in range(len(act_quant_params_list)):
            self.act_quantizer_q_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.act_quantizer_k_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
            self.act_quantizer_v_list.append(UniformAffineQuantizer(**act_quant_params_list[i]))
        
            # act_quant_params_w = act_quant_params_list[i].copy()
            act_quant_params_w = copy.deepcopy(act_quant_params_list[i])
            act_quant_params_w['n_bits'] = sm_abit

            self.act_quantizer_w_list.append(UniformAffineQuantizer(**act_quant_params_w))

    def set_quant_idx(self,idx, idx2 = None, idx3 = None, idx4 = None):
        if self.split_quantization == False:
            assert(idx2 == None)
            assert(idx3 == None)
            assert(idx3 == None)
            self.use_quantizer_idx=idx
        else:
            assert(idx2 != None)
            assert(idx3 != None)
            assert(idx4 != None)
            self.q_quantizer_idx=idx
            self.k_quantizer_idx=idx2
            self.v_quantizer_idx=idx3
            self.w_quantizer_idx=idx4

    def forward(self, x):
        if self.split_quantization == False:
            return self.forward_not_split(x)
        else:
            return self.forward_split(x)
    def forward_not_split(self, x):
        idx=self.use_quantizer_idx
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.use_act_quant and self.not_attn == False:
            q = self.act_quantizer_q_list[idx](q)
            k = self.act_quantizer_k_list[idx](k)
            
        w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5)) #bmm(q,k): q:(b,n,m),k:(b,m,l)->ans(b,n,l) flops: b*(n*m*l)
        w_ = nn.functional.softmax(w_, dim=2)
        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.use_act_quant and self.not_attn == False:
            v = self.act_quantizer_v_list[idx](v)
            w_ = self.act_quantizer_w_list[idx](w_)

        h_ = th.bmm(v, w_) #
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        
        out = x + h_
        '''
        print('q.shape:{}'.format(q.shape)) # 1,256,256
        print('k.shape:{}'.format(k.shape)) # 1,256,256
        print('w.shape:{}'.format(w_.shape)) # 1,256,256
        print('v.shape:{}'.format(v.shape)) # 1,256,256
        print('h.shape:{}'.format(h_.shape)) # 1,256,16,16
        print('out.shape:{}'.format(out.shape)) # 1,256,16,16
        exit()
        '''
        return out

    def forward_split(self, x):
        q_idx=self.q_quantizer_idx
        k_idx=self.k_quantizer_idx
        v_idx=self.v_quantizer_idx
        w_idx=self.w_quantizer_idx
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        if self.use_act_quant:
            q = self.act_quantizer_q_list[q_idx](q)
            k = self.act_quantizer_k_list[k_idx](k)
            
        w_ = th.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5)) #bmm(q,k): q:(b,n,m),k:(b,m,l)->ans(b,n,l) flops: b*(n*m*l)
        w_ = nn.functional.softmax(w_, dim=2)
        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        if self.use_act_quant:
            v = self.act_quantizer_v_list[v_idx](v)
            w_ = self.act_quantizer_w_list[w_idx](w_)

        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        
        out = x + h_
        '''
        print('q.shape:{}'.format(q.shape)) # 1,256,256
        print('k.shape:{}'.format(k.shape)) # 1,256,256
        print('w.shape:{}'.format(w_.shape)) # 1,256,256
        print('v.shape:{}'.format(v.shape)) # 1,256,256
        print('h.shape:{}'.format(h_.shape)) # 1,256,16,16
        print('out.shape:{}'.format(out.shape)) # 1,256,16,16
        exit()
        '''
        return out


def get_specials_one(quant_act=False):
    specials = {
        ResBlock: QuantResBlock_one,
        BasicTransformerBlock: QuantBasicTransformerBlock_one,
        ResnetBlock: QuantResnetBlock_one,
        AttnBlock: QuantAttnBlock_one,
        AttentionBlock:QuantAttentionBlock_one,
    }
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul_one
        specials[SMVMatMul] = QuantSMVMatMul_one
        pass
    else:
        specials[AttentionBlock] = QuantAttentionBlock_one
    return specials


    specials = {
        ResBlock: QuantResBlock_darts,
        BasicTransformerBlock: QuantBasicTransformerBlock_darts,
        ResnetBlock: QuantResnetBlock_darts,
        AttnBlock: QuantAttnBlock_darts,
        AttentionBlock:QuantAttentionBlock_darts,
        AutoResBlock: QuantAutoResBlock_darts,##
        AutoAttentionBlock: QuantAutoAttentionBlock_darts,
        # AutoQKVAttention: QuantAutoQKVAttention_darts,

        QKVAttentionLegacy:QuantQKVAttentionLegacy_darts,
        DynamicQKVAttention: QuantDynamicQKVAttention_darts,
        ##DynamicResBlock:QuantDynamicResBlock_one,
        ##DynamicAttentionBlock:QuantDynamicAttentionBlock_one,
        ##below is for ImageNet
        
    }
    if quant_act:
        specials[QKMatMul] = QuantQKMatMul_darts
        specials[SMVMatMul] = QuantSMVMatMul_darts
        pass
    else:
        specials[AttentionBlock] = QuantAttentionBlock_darts
    return specials