import logging
import torch.nn as nn
from qdiff.quant_block import get_specials_one, BaseQuantBlock_one, QuantAttentionBlock_one
from qdiff.quant_block import QuantSMVMatMul_one, QuantBasicTransformerBlock_one, QuantAttnBlock_one, QuantResnetBlock_one
from qdiff.quant_layer import StraightThrough, QuantModule_one
from ldm.modules.attention import BasicTransformerBlock
from guided_diffusion.unet import QKVAttention as AutoQKVAttention

import torch.nn.functional as F
import numpy as np
import pdb
import torch
logger = logging.getLogger(__name__)
def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res

class QuantModel_one(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params_list, act_quant_params_list, args,config,not_attn = False,**kwargs):#
        super().__init__()
        self.args=args
        self.model = model
        self.config=config
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        # print(config.data.image_size)   32
        # print('--')
        # print(config.data.channels)    3
        self.not_attn = not_attn
        self.specials = get_specials_one(act_quant_params_list[0]['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params_list, act_quant_params_list)
        self.quant_block_refactor(self.model, weight_quant_params_list, act_quant_params_list)
        self.layer_num=self.get_num()
        self.split_quantization = False

    def quant_module_refactor(self, module: nn.Module, weight_quant_params_list, act_quant_params_list):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule_one(
                    child_module, weight_quant_params_list, act_quant_params_list))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params_list, act_quant_params_list)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params_list, act_quant_params_list):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock_one, QuantAttnBlock_one]:
                    if self.specials[type(child_module)] in [QuantAttnBlock_one]:
                        setattr(module, name, self.specials[type(child_module)](child_module,
                            act_quant_params_list, sm_abit=self.sm_abit, not_attn = self.not_attn))
                    else:
                        setattr(module, name, self.specials[type(child_module)](child_module,
                            act_quant_params_list, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul_one:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params_list, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul_one:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params_list))
                elif self.specials[type(child_module)] == QuantAttentionBlock_one:
                    child_module.attention.qkv_matmul=self.specials[type(child_module.attention.qkv_matmul)](act_quant_params_list)
                    child_module.attention.smv_matmul=self.specials[type(child_module.attention.smv_matmul)](act_quant_params_list)
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params_list))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params_list))
            else:
                self.quant_block_refactor(child_module, weight_quant_params_list, act_quant_params_list)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                m.set_quant_state(weight_quant, act_quant)

    def cal_bitops(self,use_bitwidth):#two dims array
        ####To Do
        list_conv2d=[]
        list_conv1d=[]
        list_linear=[]
        list_QKV=[]
        def quantmodule_hook(self,input,output):# conv family
            ins = input[0].size()
            outs = output.size()
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitw=self.weight_quantizer_list[idx].n_bits
                bita=self.act_quantizer_list[idx].n_bits
            else:
                bitw = self.weight_quantizer_list[self.weight_quantizer_idx].n_bits
                bita = self.act_quantizer_list[self.act_quantizer_idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops = n_macs * bitw * bita * 1e-9
                list_conv2d.append(n_bitops)
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops=n_macs*bitw*bita*1e-9
                list_conv1d.append(n_bitops)
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
            
        def quantattnblock_hook(self,input,output): #attn block
            # b,c, *spatial = output[0].shape 
            c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c 
            matmul_macs = (num_spatial ** 2) *c 
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitq=self.act_quantizer_q_list[idx].n_bits
                bitk=self.act_quantizer_k_list[idx].n_bits
                bitv=self.act_quantizer_v_list[idx].n_bits
                bitw=self.act_quantizer_w_list[idx].n_bits
            else:
                bitq = self.act_quantizer_q_list[self.q_quantizer_idx].n_bits
                bitk = self.act_quantizer_k_list[self.k_quantizer_idx].n_bits
                bitv = self.act_quantizer_v_list[self.v_quantizer_idx].n_bits
                bitw = self.act_quantizer_w_list[self.w_quantizer_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)

        def quantAttentionBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.attention.qkv_matmul.use_quantizer_idx
                bitq=self.attention.qkv_matmul.act_quantizer_q_list[idx].n_bits
                bitk=self.attention.qkv_matmul.act_quantizer_k_list[idx].n_bits
                bitv=self.attention.smv_matmul.act_quantizer_v_list[idx].n_bits
                bitw=self.attention.smv_matmul.act_quantizer_w_list[idx].n_bits
            else:
                q_idx = self.attention.qkv_matmul.q_quantizer_idx
                k_idx = self.attention.qkv_matmul.k_quantizer_idx
                v_idx = self.attention.smv_matmul.v_quantizer_idx
                w_idx = self.attention.smv_matmul.w_quantizer_idx
                bitq = self.attention.qkv_matmul.act_quantizer_q_list[q_idx].n_bits
                bitk = self.attention.qkv_matmul.act_quantizer_k_list[k_idx].n_bits
                bitv = self.attention.smv_matmul.act_quantizer_v_list[v_idx].n_bits
                bitw = self.attention.smv_matmul.act_quantizer_w_list[w_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)
        
        def quantQuantBasicTransformerBlock_hook(self, input, output):
            #check一下是不是对的捏
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx = self.attn1.use_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[idx].n_bits
            else:
                q1_idx = self.attn1.q_quantizer_idx
                k1_idx = self.attn1.k_quantizer_idx
                v1_idx = self.attn1.v_quantizer_idx
                w1_idx = self.attn1.w_quantizer_idx

                q2_idx = self.attn2.q_quantizer_idx
                k2_idx = self.attn2.k_quantizer_idx
                v2_idx = self.attn2.v_quantizer_idx
                w2_idx = self.attn2.w_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[q1_idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[k1_idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[v1_idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[w1_idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[q2_idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[k2_idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[v2_idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[w2_idx].n_bits
            matmul_ops=matmul_macs*((bitq1*bitk1+bitw1*bitv1)+(bitq2*bitk2+bitw2*bitv2))*1e-9
            list_QKV.append(matmul_ops)

        handler_list=[]
        for i in range(1):#for i in range(len(use_bitwidth)):#为了效率，目前就用一步的设置了，多的不设置了
            #use_bitwidth[i]
            cnt=0
            for m in self.model.modules():
                if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                    if isinstance(m, QuantModule_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                        cnt += 2
                    elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3],use_bitwidth[i][cnt + 4],use_bitwidth[i][cnt + 5],use_bitwidth[i][cnt + 6],use_bitwidth[i][cnt + 7])
                        cnt += 8
                    elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                        m.set_quant_idx_cand(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1],use_bitwidth[i][cnt + 2],use_bitwidth[i][cnt + 3])
                        cnt += 4
                    else:
                        logger.info('problem')
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttentionBlock_one):
                        handler=m.register_forward_hook(quantAttentionBlock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantBasicTransformerBlock_one):
                        handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                        handler_list.append(handler)
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                # if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                #     if isinstance(m, QuantModule_one) and m.split_quantization == True:
                #         m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                #         cnt += 2
                #     elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                #         m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3])
                #         cnt += 4
                #     elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                #         m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3],use_bitwidth[i][cnt + 4],use_bitwidth[i][cnt + 5],use_bitwidth[i][cnt + 6],use_bitwidth[i][cnt + 7])
                #         cnt += 8
                #     elif isinstance(m, (QuantQKMatMul_one, QuantSMVMatMul_one)) and m.split_quantization == True:
                #         m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                #         cnt += 2
                #     elif isinstance(m, QuantModule_one) or isinstance(m, QuantAttnBlock_one) or isinstance(m, (QuantQKMatMul_one, QuantSMVMatMul_one)):
                #         m.set_quant_idx(use_bitwidth[i][cnt])
                #         cnt+=1
                #     if isinstance(m,QuantModule_one):
                #         handler=m.register_forward_hook(quantmodule_hook)
                #         handler_list.append(handler)
                #     elif isinstance(m,QuantAttnBlock_one):
                #         handler=m.register_forward_hook(quantattnblock_hook)
                #         handler_list.append(handler)
                #     elif isinstance(m,QuantAttentionBlock_one):
                #         handler=m.register_forward_hook(quantAttentionBlock_hook)
                #         handler_list.append(handler)
                #     elif isinstance(m,QuantBasicTransformerBlock_one):
                #         handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                #         handler_list.append(handler)
            ##for LDM
            
            if hasattr(self, 'image_size') and hasattr(self, 'image_size'):
                logger.info('LDM image size')
                image_size = self.image_size
                channels = self.in_channels
            else:#for cifar
                logger.info('cifar image size')
                image_size = self.config.data.image_size
                channels = self.config.data.channels
            if self.args.task =='cifar' or self.args.task=='lsun':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
                cali_xs,cali_ts=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda())
            elif self.args.task=='image':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,1,512)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            elif self.args.task=='txt':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,77,768)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            else:
                raise ValueError('unrecognized, please adjust task')
            for handler in handler_list:
                handler.remove()
        ans=sum(list_conv2d)+sum(list_conv1d)+sum(list_linear)+sum(list_QKV)
        
        logger.info('conv2d: {}, conv1d: {}, linear: {}, QKV: {}, all: {}'.format(sum(list_conv2d),sum(list_conv1d),sum(list_linear),sum(list_QKV),ans))
        return ans

    def set_split_quantization(self, whether):#True == split quantization
        self.split_quantization = whether
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                m.split_quantization = whether
            if isinstance(m, QuantBasicTransformerBlock_one):
                m.attn1.split_quantization=whether
                m.attn2.split_quantization=whether

    def set_quant_idx(self, i):
        for m in self.model.modules():
            if isinstance(m, QuantModule_one):
                m.set_quant_idx(i)
            elif isinstance(m, BaseQuantBlock_one):
                m.set_quant_idx(i)
            if isinstance(m, QuantBasicTransformerBlock_one):
                m.attn1.use_quantizer_idx = i
                m.attn2.use_quantizer_idx = i
            
    def get_allocation(self, l):#not adjusted
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantQKMatMul_one, QuantSMVMatMul_one,QuantBasicTransformerBlock_one)):
                if isinstance(m,QuantModule_one) and m.split_quantization == True:
                    l.append(m.weight_quantizer_idx)
                    l.append(m.act_quantizer_idx)
                    l.append(-10)
                elif isinstance(m,QuantQKMatMul_one) and m.split_quantization == True:
                    l.append(m.q_quantizer_idx)
                    l.append(m.k_quantizer_idx)
                    l.append(-10)
                elif isinstance(m,QuantSMVMatMul_one) and m.split_quantization == True:
                    l.append(m.v_quantizer_idx)
                    l.append(m.w_quantizer_idx)
                    l.append(-10)
                elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                    l.append(m.q_quantizer_idx)
                    l.append(m.k_quantizer_idx)
                    l.append(m.v_quantizer_idx)
                    l.append(m.w_quantizer_idx)
                    l.append(-10)
                elif isinstance(m,QuantBasicTransformerBlock_one) and m.split_quantization==True:
                    l.append(m.attn1.q_quantizer_idx)
                    l.append(m.attn1.k_quantizer_idx)
                    l.append(m.attn1.v_quantizer_idx)
                    l.append(m.attn1.w_quantizer_idx)

                    l.append(m.attn2.q_quantizer_idx)
                    l.append(m.attn2.k_quantizer_idx)
                    l.append(m.attn2.v_quantizer_idx)
                    l.append(m.attn2.w_quantizer_idx)
                    l.append(-10)
                else:
                    l.append(m.use_quantizer_idx)
            # elif not isinstance(m, QuantResnetBlock_one) and isinstance(m, BaseQuantBlock_one):
            #     logger.info(str(m))

    def set_quant_idx_cand(self, cand):
        #扫描modules，给符合条件的modules进行quant_idx的set
        #modified to same
        cnt=0
        for m in self.model.modules():
            # if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantSMVMatMul_one, QuantQKMatMul_one,QuantBasicTransformerBlock_one)):
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantAttentionBlock_one,QuantBasicTransformerBlock_one)):
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1])
                    cnt += 2
                elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1], cand[cnt + 2], cand[cnt + 3])
                    cnt += 4
                # elif isinstance(m, (QuantSMVMatMul_one, QuantQKMatMul_one)) and m.split_quantization == True:
                #     m.set_quant_idx(cand[cnt], cand[cnt + 1])
                #     cnt += 2
                elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                    m.set_quant_idx_cand(cand[cnt], cand[cnt + 1],cand[cnt+2],cand[cnt+3])
                    cnt += 4
                elif isinstance(m,QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1], cand[cnt + 2], cand[cnt + 3],cand[cnt + 4],cand[cnt + 5],cand[cnt + 6],cand[cnt + 7])
                    cnt += 8
                else:
                    m.set_quant_idx(cand[cnt])
                    cnt+=1
    def get_num(self):
        #modified to same
        cnt=0
        for m in self.model.modules():
            # if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantSMVMatMul_one, QuantQKMatMul_one,QuantBasicTransformerBlock_one)): # BaseQuantBlock_one
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantAttentionBlock_one, QuantBasicTransformerBlock_one)): # BaseQuantBlock_one
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    cnt = cnt + 2
                elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                    cnt = cnt + 4
                # elif isinstance(m, (QuantSMVMatMul_one, QuantQKMatMul_one)) and m.split_quantization == True:
                #     cnt = cnt + 2
                elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization==True:
                    cnt = cnt+4
                elif isinstance(m,QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    cnt = cnt + 8
                elif isinstance(m, QuantAttnBlock_one) or isinstance(m,QuantModule_one) or isinstance(m,(QuantSMVMatMul_one, QuantQKMatMul_one,QuantBasicTransformerBlock_one)): # modified: BaseQuantBlock not count
                    cnt = cnt + 1
        return cnt
    def get_num_offload(self):
        cnt=0
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one, QuantAttentionBlock_one,QuantBasicTransformerBlock_one)): # BaseQuantBlock_one
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    m.weight_idx_all=cnt
                    m.act_idx_all=cnt+1
                    cnt = cnt + 2
                elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                    m.q_idx_all=cnt
                    m.k_idx_all=cnt+1
                    m.v_idx_all=cnt+2
                    m.w_idx_all=cnt+3
                    cnt = cnt + 4
                elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                    m.q_idx_all=cnt
                    m.k_idx_all=cnt+1
                    m.v_idx_all=cnt+2
                    m.w_idx_all=cnt+3
                    cnt = cnt + 4
                elif isinstance(m,QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    m.attn1_q_idx_all = cnt
                    m.attn1_k_idx_all = cnt+1
                    m.attn1_v_idx_all = cnt+2
                    m.attn1_w_idx_all = cnt+3

                    m.attn2_q_idx_all = cnt+4
                    m.attn2_k_idx_all = cnt+5
                    m.attn2_v_idx_all = cnt+6
                    m.attn2_w_idx_all = cnt+7
                    cnt = cnt + 8
                elif isinstance(m, QuantAttnBlock_one) or isinstance(m,QuantModule_one) or isinstance(m,(QuantSMVMatMul_one, QuantQKMatMul_one,QuantBasicTransformerBlock_one)): # modified: BaseQuantBlock not count
                    raise ValueError('not split quantization not permitted')
                    cnt = cnt + 1
        return cnt
    def set_quant_idx_offload(self, cand):
        #扫描modules，给符合条件的modules进行quant_idx的set
        cnt=0
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1])
                    cnt += 2
                elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1], cand[cnt + 2], cand[cnt + 3])
                    cnt += 4
                elif isinstance(m,QuantAttentionBlock_one) and m.split_quantization == True:
                    m.set_quant_idx_cand(cand[cnt], cand[cnt + 1], cand[cnt +2 ],cand[cnt+3])
                    cnt += 4
                elif isinstance(m,QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    m.set_quant_idx(cand[cnt], cand[cnt + 1], cand[cnt + 2], cand[cnt + 3],cand[cnt + 4],cand[cnt + 5],cand[cnt + 6],cand[cnt + 7])
                    cnt += 8
                else:
                    raise ValueError('only split quantization need it')
                    m.set_quant_idx(cand[cnt])
                    cnt+=1
        logger.info('cnt:{}'.format(cnt))
    def cal_bitops_offload(self,use_bitwidth):#two dims array
        ####To Do
        list_conv=[]
        list_QKV=[]
        list_trans=[]
        list_conv_idx=[]
        list_QKV_idx=[]
        list_trans_idx=[]
        def quantmodule_hook(self,input,output):# conv family
            ins = input[0].size()
            outs = output.size()
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitw=self.weight_quantizer_list[idx].n_bits
                bita=self.act_quantizer_list[idx].n_bits
            else:
                bitw = self.weight_quantizer_list[self.weight_quantizer_idx].n_bits
                bita = self.act_quantizer_list[self.act_quantizer_idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:]) * 1e-9
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])*1e-9
                # n_bitops=n_macs*bitw*bita*1e-9
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)*1e-9
                # n_bitops = n_macs * bitw * bita * 1e-9
                
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
            list_conv.append(n_macs)
            list_conv_idx.append([self.weight_idx_all, self.act_idx_all])
            
        def quantattnblock_hook(self,input,output): #attn block
            # b,c, *spatial = output[0].shape
            c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c *1e-9
            matmul_macs = (num_spatial ** 2) *c *1e-9
            
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2 * 1e-9
            
            list_QKV.append(matmul_macs)
            list_QKV_idx.append([self.q_idx_all, self.k_idx_all, self.v_idx_all, self.w_idx_all])
        def quantAttentionBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c *1e-9
            matmul_macs = (num_spatial ** 2) *c *1e-9
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2 *1e-9
            
            list_QKV.append(matmul_macs)
            list_QKV_idx.append([self.q_idx_all,self.k_idx_all,self.v_idx_all,self.w_idx_all])
        def quantQuantBasicTransformerBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c *1e-9
            matmul_macs = (num_spatial ** 2) *c *1e-9
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2 * 1e-9
            # matmul_ops=matmul_macs*((bitq1*bitk1+bitw1*bitv1)+(bitq2*bitk2+bitw2*bitv2))*1e-9
            list_trans.append(matmul_macs)
            list_trans_idx.append([self.attn1_q_idx_all, self.attn1_k_idx_all, self.attn1_v_idx_all, self.attn1_w_idx_all, self.attn2_q_idx_all, self.attn2_k_idx_all, self.attn2_v_idx_all, self.attn2_w_idx_all])

        handler_list=[]
        for i in range(1):#for i in range(len(use_bitwidth)):#为了效率，目前就用一步的设置了，多的不设置了
            #use_bitwidth[i]
            cnt=0
            for m in self.model.modules():
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                    if isinstance(m, QuantModule_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                        cnt += 2
                    elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3],use_bitwidth[i][cnt + 4],use_bitwidth[i][cnt + 5],use_bitwidth[i][cnt + 6],use_bitwidth[i][cnt + 7])
                        cnt += 8
                    elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                        m.set_quant_idx_cand(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1],use_bitwidth[i][cnt + 2],use_bitwidth[i][cnt + 3])
                        cnt += 4
                    else:
                        logger.info('problem')
                    # elif isinstance(m, QuantModule_one) or isinstance(m, QuantAttnBlock_one):
                    #     m.set_quant_idx(use_bitwidth[i][cnt])
                    #     cnt+=1
                    # elif isinstance(m, QuantAttentionBlock_one):
                    #     m.attention.qkv_matmul(use_bitwidth[i][cnt])
                    #     m.attention.smv_matmul(use_bitwidth[i][cnt+1])
                    #     cnt+=2
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttentionBlock_one):
                        handler=m.register_forward_hook(quantAttentionBlock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantBasicTransformerBlock_one):
                        handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                        handler_list.append(handler)
            ##for LDM
            logger.info('cnt:{}'.format(cnt))
            if hasattr(self, 'image_size') and hasattr(self, 'image_size'):
                logger.info('LDM image size')
                image_size = self.image_size
                channels = self.in_channels
            else:#for cifar
                logger.info('cifar image size')
                image_size = self.config.data.image_size
                channels = self.config.data.channels
            if self.args.task =='cifar' or self.args.task=='lsun':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
                cali_xs,cali_ts=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda())
            elif self.args.task=='image':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,1,512)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            elif self.args.task=='txt':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,77,768)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            else:
                raise ValueError('unrecognized, please adjust task')
            for handler in handler_list:
                handler.remove()
        return list_conv, list_QKV, list_trans, list_conv_idx, list_QKV_idx, list_trans_idx

    def check_act(self, use_bitwidth):
        cnt=0
        for m in self.model.modules():
            #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        assert use_bitwidth[i][m.act_idx_all] == 0
                    cnt += 2
                elif isinstance(m, (QuantAttnBlock_one,QuantAttentionBlock_one)) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        assert use_bitwidth[i][m.q_idx_all] == 0
                        assert use_bitwidth[i][m.k_idx_all] == 0
                        assert use_bitwidth[i][m.v_idx_all] == 0
                        assert use_bitwidth[i][m.w_idx_all] == 0
                    cnt += 4
                elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        assert use_bitwidth[i][m.attn1_q_idx_all] == 0
                        assert use_bitwidth[i][m.attn1_k_idx_all] == 0
                        assert use_bitwidth[i][m.attn1_v_idx_all] == 0
                        assert use_bitwidth[i][m.attn1_w_idx_all] == 0

                        assert use_bitwidth[i][m.attn2_q_idx_all] == 0
                        assert use_bitwidth[i][m.attn2_k_idx_all] == 0
                        assert use_bitwidth[i][m.attn2_v_idx_all] == 0
                        assert use_bitwidth[i][m.attn2_w_idx_all] == 0
                    cnt += 8
                else:
                    logger.info('problem')
    
    def change_act(self, use_bitwidth):
        cnt=0
        for m in self.model.modules():
            #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
            if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                if isinstance(m, QuantModule_one) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        use_bitwidth[i][m.act_idx_all] = 0
                    cnt += 2
                elif isinstance(m, (QuantAttnBlock_one,QuantAttentionBlock_one)) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        use_bitwidth[i][m.q_idx_all] = use_bitwidth[i][m.k_idx_all] = use_bitwidth[i][m.v_idx_all] = use_bitwidth[i][m.w_idx_all] = 0
                    cnt += 4
                elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                    for i in range(len(use_bitwidth)):
                        use_bitwidth[i][m.attn1_q_idx_all] = use_bitwidth[i][m.attn1_k_idx_all] = use_bitwidth[i][m.attn1_v_idx_all] = use_bitwidth[i][m.attn1_w_idx_all] = 0
                        use_bitwidth[i][m.attn2_q_idx_all] = use_bitwidth[i][m.attn2_k_idx_all] = use_bitwidth[i][m.attn2_v_idx_all] = use_bitwidth[i][m.attn2_w_idx_all] = 0
                    cnt += 8
                else:
                    logger.info('problem')

    def cal_bitops_cand_offload(self,use_bitwidth):#two dims array
        ####To Do
        list_conv2d=[]
        list_conv1d=[]
        list_linear=[]
        list_QKV=[]
        def quantmodule_hook(self,input,output):# conv family
            ins = input[0].size()
            outs = output.size()
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitw=self.weight_quantizer_list[idx].n_bits
                bita=self.act_quantizer_list[idx].n_bits
            else:
                bitw = self.weight_quantizer_list[self.weight_quantizer_idx].n_bits
                bita = self.act_quantizer_list[self.act_quantizer_idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops = n_macs * bitw * bita * 1e-9
                list_conv2d.append(n_bitops)
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops=n_macs*bitw*bita*1e-9
                list_conv1d.append(n_bitops)
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
            
        def quantattnblock_hook(self,input,output): #attn block
            # b,c, *spatial = output[0].shape 
            c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c 
            matmul_macs = (num_spatial ** 2) *c 
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitq=self.act_quantizer_q_list[idx].n_bits
                bitk=self.act_quantizer_k_list[idx].n_bits
                bitv=self.act_quantizer_v_list[idx].n_bits
                bitw=self.act_quantizer_w_list[idx].n_bits
            else:
                bitq = self.act_quantizer_q_list[self.q_quantizer_idx].n_bits
                bitk = self.act_quantizer_k_list[self.k_quantizer_idx].n_bits
                bitv = self.act_quantizer_v_list[self.v_quantizer_idx].n_bits
                bitw = self.act_quantizer_w_list[self.w_quantizer_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)

        def quantAttentionBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.attention.qkv_matmul.use_quantizer_idx
                bitq=self.attention.qkv_matmul.act_quantizer_q_list[idx].n_bits
                bitk=self.attention.qkv_matmul.act_quantizer_k_list[idx].n_bits
                bitv=self.attention.smv_matmul.act_quantizer_v_list[idx].n_bits
                bitw=self.attention.smv_matmul.act_quantizer_w_list[idx].n_bits
            else:
                q_idx = self.attention.qkv_matmul.q_quantizer_idx
                k_idx = self.attention.qkv_matmul.k_quantizer_idx
                v_idx = self.attention.smv_matmul.v_quantizer_idx
                w_idx = self.attention.smv_matmul.w_quantizer_idx
                bitq = self.attention.qkv_matmul.act_quantizer_q_list[q_idx].n_bits
                bitk = self.attention.qkv_matmul.act_quantizer_k_list[k_idx].n_bits
                bitv = self.attention.smv_matmul.act_quantizer_v_list[v_idx].n_bits
                bitw = self.attention.smv_matmul.act_quantizer_w_list[w_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)
            # if hasattr(self,'split_quantization') and self.split_quantization == True:
            #     pdb.set_trace()
        
        def quantQuantBasicTransformerBlock_hook(self, input, output):
            #check一下是不是对的捏
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx = self.attn1.use_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[idx].n_bits
            else:
                q1_idx = self.attn1.q_quantizer_idx
                k1_idx = self.attn1.k_quantizer_idx
                v1_idx = self.attn1.v_quantizer_idx
                w1_idx = self.attn1.w_quantizer_idx

                q2_idx = self.attn2.q_quantizer_idx
                k2_idx = self.attn2.k_quantizer_idx
                v2_idx = self.attn2.v_quantizer_idx
                w2_idx = self.attn2.w_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[q1_idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[k1_idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[v1_idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[w1_idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[q2_idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[k2_idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[v2_idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[w2_idx].n_bits
            matmul_ops=matmul_macs*((bitq1*bitk1+bitw1*bitv1)+(bitq2*bitk2+bitw2*bitv2))*1e-9
            list_QKV.append(matmul_ops)

        handler_list=[]
        for i in range(1):#for i in range(len(use_bitwidth)):#为了效率，目前就用一步的设置了，多的不设置了
            #use_bitwidth[i]
            cnt=0
            for m in self.model.modules():
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                    if isinstance(m, QuantModule_one) and m.split_quantization == True:
                        assert m.weight_idx_all==cnt
                        assert m.act_idx_all==cnt+1
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                        cnt += 2
                    elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx_cand(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1],use_bitwidth[i][cnt + 2],use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                        assert m.attn1_q_idx_all==cnt
                        assert m.attn1_k_idx_all==cnt+1
                        assert m.attn1_v_idx_all==cnt+2
                        assert m.attn1_w_idx_all==cnt+3

                        assert m.attn2_q_idx_all==cnt+4
                        assert m.attn2_k_idx_all==cnt+5
                        assert m.attn2_v_idx_all==cnt+6
                        assert m.attn2_w_idx_all==cnt+7
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3],use_bitwidth[i][cnt + 4],use_bitwidth[i][cnt + 5],use_bitwidth[i][cnt + 6],use_bitwidth[i][cnt + 7])
                        cnt += 8
                    else:
                        logger.info('problem')
                    # elif isinstance(m, QuantModule_one) or isinstance(m, QuantAttnBlock_one) or isinstance(m, (QuantQKMatMul_one, QuantSMVMatMul_one)):
                    #     m.set_quant_idx(use_bitwidth[i][cnt])
                    #     cnt+=1
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttentionBlock_one):
                        handler=m.register_forward_hook(quantAttentionBlock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantBasicTransformerBlock_one):
                        handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                        handler_list.append(handler)
            ##for LDM
            logger.info('cnt:{}'.format(cnt))
            if hasattr(self, 'image_size') and hasattr(self, 'image_size'):
                logger.info('LDM image size')
                image_size = self.image_size
                channels = self.in_channels
            else:#for cifar
                logger.info('cifar image size')
                image_size = self.config.data.image_size
                channels = self.config.data.channels
            #调整一下以后用来处理image和txt
            if self.args.task =='cifar' or self.args.task=='lsun':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
                cali_xs,cali_ts=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda())
            elif self.args.task=='image':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,1,512)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            elif self.args.task=='txt':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,77,768)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            else:
                raise ValueError('unrecognized')
            for handler in handler_list:
                handler.remove()
        ans=sum(list_conv2d)+sum(list_conv1d)+sum(list_linear)+sum(list_QKV)
        logger.info('conv2d: {}, conv1d: {}, linear: {}, QKV: {}, all: {}'.format(sum(list_conv2d),sum(list_conv1d),sum(list_linear),sum(list_QKV),ans))
        return ans
    
    def cal_bitops_w_a(self,_w,_a):#two dims array
        ####To Do
        list_conv2d=[]
        list_conv1d=[]
        list_linear=[]
        list_QKV=[]
        def quantmodule_hook(self,input,output):# conv family
            ins = input[0].size()
            outs = output.size()
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitw=self.weight_quantizer_list[idx].n_bits
                bita=self.act_quantizer_list[idx].n_bits
            else:
                bitw = self.weight_quantizer_list[self.weight_quantizer_idx].n_bits
                bita = self.act_quantizer_list[self.act_quantizer_idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops = n_macs * bitw * bita * 1e-9
                list_conv2d.append(n_bitops)
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops=n_macs*bitw*bita*1e-9
                list_conv1d.append(n_bitops)
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
            
        def quantattnblock_hook(self,input,output): #attn block
            # b,c, *spatial = output[0].shape 
            c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c 
            matmul_macs = (num_spatial ** 2) *c 
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitq=self.act_quantizer_q_list[idx].n_bits
                bitk=self.act_quantizer_k_list[idx].n_bits
                bitv=self.act_quantizer_v_list[idx].n_bits
                bitw=self.act_quantizer_w_list[idx].n_bits
            else:
                bitq = self.act_quantizer_q_list[self.q_quantizer_idx].n_bits
                bitk = self.act_quantizer_k_list[self.k_quantizer_idx].n_bits
                bitv = self.act_quantizer_v_list[self.v_quantizer_idx].n_bits
                bitw = self.act_quantizer_w_list[self.w_quantizer_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)

        def quantAttentionBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.attention.qkv_matmul.use_quantizer_idx
                bitq=self.attention.qkv_matmul.act_quantizer_q_list[idx].n_bits
                bitk=self.attention.qkv_matmul.act_quantizer_k_list[idx].n_bits
                bitv=self.attention.smv_matmul.act_quantizer_v_list[idx].n_bits
                bitw=self.attention.smv_matmul.act_quantizer_w_list[idx].n_bits
            else:
                q_idx = self.attention.qkv_matmul.q_quantizer_idx
                k_idx = self.attention.qkv_matmul.k_quantizer_idx
                v_idx = self.attention.smv_matmul.v_quantizer_idx
                w_idx = self.attention.smv_matmul.w_quantizer_idx
                bitq = self.attention.qkv_matmul.act_quantizer_q_list[q_idx].n_bits
                bitk = self.attention.qkv_matmul.act_quantizer_k_list[k_idx].n_bits
                bitv = self.attention.smv_matmul.act_quantizer_v_list[v_idx].n_bits
                bitw = self.attention.smv_matmul.act_quantizer_w_list[w_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)
            # if hasattr(self,'split_quantization') and self.split_quantization == True:
            #     pdb.set_trace()
        
        def quantQuantBasicTransformerBlock_hook(self, input, output):
            #check一下是不是对的捏
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx = self.attn1.use_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[idx].n_bits
            else:
                q1_idx = self.attn1.q_quantizer_idx
                k1_idx = self.attn1.k_quantizer_idx
                v1_idx = self.attn1.v_quantizer_idx
                w1_idx = self.attn1.w_quantizer_idx

                q2_idx = self.attn2.q_quantizer_idx
                k2_idx = self.attn2.k_quantizer_idx
                v2_idx = self.attn2.v_quantizer_idx
                w2_idx = self.attn2.w_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[q1_idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[k1_idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[v1_idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[w1_idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[q2_idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[k2_idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[v2_idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[w2_idx].n_bits
            matmul_ops=matmul_macs*((bitq1*bitk1+bitw1*bitv1)+(bitq2*bitk2+bitw2*bitv2))*1e-9
            list_QKV.append(matmul_ops)

        handler_list=[]
        for i in range(1):#for i in range(len(use_bitwidth)):#为了效率，目前就用一步的设置了，多的不设置了
            #use_bitwidth[i]
            cnt=0
            for m in self.model.modules():
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                    if isinstance(m, QuantModule_one) and m.split_quantization == True:
                        assert m.weight_idx_all==cnt
                        assert m.act_idx_all==cnt+1
                        m.set_quant_idx(_w,_a)
                        cnt += 2
                    elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx(_a,_a,_a,_a)
                        cnt += 4
                    elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx_cand(_a,_a,_a,_a)
                        cnt += 4
                    elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                        assert m.attn1_q_idx_all==cnt
                        assert m.attn1_k_idx_all==cnt+1
                        assert m.attn1_v_idx_all==cnt+2
                        assert m.attn1_w_idx_all==cnt+3

                        assert m.attn2_q_idx_all==cnt+4
                        assert m.attn2_k_idx_all==cnt+5
                        assert m.attn2_v_idx_all==cnt+6
                        assert m.attn2_w_idx_all==cnt+7
                        m.set_quant_idx(_a,_a,_a,_a,_a,_a,_a,_a)
                        cnt += 8
                    else:
                        logger.info('problem')
                    # elif isinstance(m, QuantModule_one) or isinstance(m, QuantAttnBlock_one) or isinstance(m, (QuantQKMatMul_one, QuantSMVMatMul_one)):
                    #     m.set_quant_idx(use_bitwidth[i][cnt])
                    #     cnt+=1
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttentionBlock_one):
                        handler=m.register_forward_hook(quantAttentionBlock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantBasicTransformerBlock_one):
                        handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                        handler_list.append(handler)
            ##for LDM
            logger.info('cnt:{}'.format(cnt))
            if hasattr(self, 'image_size') and hasattr(self, 'image_size'):
                logger.info('LDM image size')
                image_size = self.image_size
                channels = self.in_channels
            else:#for cifar
                logger.info('cifar image size')
                image_size = self.config.data.image_size
                channels = self.config.data.channels
            #调整一下以后用来处理image和txt
            if self.args.task =='cifar' or self.args.task=='lsun':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
                cali_xs,cali_ts=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda())
            elif self.args.task=='image':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,1,512)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            elif self.args.task=='txt':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,77,768)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            else:
                raise ValueError('unrecognized')
            for handler in handler_list:
                handler.remove()
        ans=sum(list_conv2d)+sum(list_conv1d)+sum(list_linear)+sum(list_QKV)
        logger.info('conv2d: {}, conv1d: {}, linear: {}, QKV: {}, all: {}'.format(sum(list_conv2d),sum(list_conv1d),sum(list_linear),sum(list_QKV),ans))
        return ans
    















    def cal_bitops_cand_offload_no_constraint(self,use_bitwidth):#two dims array
        ####To Do
        list_conv2d=[]
        list_conv1d=[]
        list_linear=[]
        list_QKV=[]
        def quantmodule_hook(self,input,output):# conv family
            ins = input[0].size()
            outs = output.size()
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitw=self.weight_quantizer_list[idx].n_bits
                bita=self.act_quantizer_list[idx].n_bits
            else:
                bitw = self.weight_quantizer_list[self.weight_quantizer_idx].n_bits
                bita = self.act_quantizer_list[self.act_quantizer_idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops = n_macs * bitw * bita * 1e-9
                list_conv2d.append(n_bitops)
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops=n_macs*bitw*bita*1e-9
                list_conv1d.append(n_bitops)
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
            
        def quantattnblock_hook(self,input,output): #attn block
            # b,c, *spatial = output[0].shape 
            c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c 
            matmul_macs = (num_spatial ** 2) *c 
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.use_quantizer_idx
                bitq=self.act_quantizer_q_list[idx].n_bits
                bitk=self.act_quantizer_k_list[idx].n_bits
                bitv=self.act_quantizer_v_list[idx].n_bits
                bitw=self.act_quantizer_w_list[idx].n_bits
            else:
                bitq = self.act_quantizer_q_list[self.q_quantizer_idx].n_bits
                bitk = self.act_quantizer_k_list[self.k_quantizer_idx].n_bits
                bitv = self.act_quantizer_v_list[self.v_quantizer_idx].n_bits
                bitw = self.act_quantizer_w_list[self.w_quantizer_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)

        def quantAttentionBlock_hook(self, input, output):
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # #matmul_macs = 2 * b * b * num_spatial
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx=self.attention.qkv_matmul.use_quantizer_idx
                bitq=self.attention.qkv_matmul.act_quantizer_q_list[idx].n_bits
                bitk=self.attention.qkv_matmul.act_quantizer_k_list[idx].n_bits
                bitv=self.attention.smv_matmul.act_quantizer_v_list[idx].n_bits
                bitw=self.attention.smv_matmul.act_quantizer_w_list[idx].n_bits
            else:
                q_idx = self.attention.qkv_matmul.q_quantizer_idx
                k_idx = self.attention.qkv_matmul.k_quantizer_idx
                v_idx = self.attention.smv_matmul.v_quantizer_idx
                w_idx = self.attention.smv_matmul.w_quantizer_idx
                bitq = self.attention.qkv_matmul.act_quantizer_q_list[q_idx].n_bits
                bitk = self.attention.qkv_matmul.act_quantizer_k_list[k_idx].n_bits
                bitv = self.attention.smv_matmul.act_quantizer_v_list[v_idx].n_bits
                bitw = self.attention.smv_matmul.act_quantizer_w_list[w_idx].n_bits
            matmul_ops=matmul_macs*(bitq*bitk+bitw*bitv)*1e-9
            list_QKV.append(matmul_ops)
            # if hasattr(self,'split_quantization') and self.split_quantization == True:
            #     pdb.set_trace()
        
        def quantQuantBasicTransformerBlock_hook(self, input, output):
            #check一下是不是对的捏
            # b,c,*spatial =output[0].shape
            c,*spatial =output[0].shape
            num_spatial = int(np.prod(spatial))
            # matmul_macs = b * (num_spatial ** 2) *c
            matmul_macs = (num_spatial ** 2) *c
            # c, *spatial = output[0].shape
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = c *  (num_spatial)**2
            if self.split_quantization == False:
                idx = self.attn1.use_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[idx].n_bits
            else:
                q1_idx = self.attn1.q_quantizer_idx
                k1_idx = self.attn1.k_quantizer_idx
                v1_idx = self.attn1.v_quantizer_idx
                w1_idx = self.attn1.w_quantizer_idx

                q2_idx = self.attn2.q_quantizer_idx
                k2_idx = self.attn2.k_quantizer_idx
                v2_idx = self.attn2.v_quantizer_idx
                w2_idx = self.attn2.w_quantizer_idx
                bitq1=self.attn1.act_quantizer_q_list[q1_idx].n_bits
                bitk1=self.attn1.act_quantizer_k_list[k1_idx].n_bits
                bitv1=self.attn1.act_quantizer_v_list[v1_idx].n_bits
                bitw1=self.attn1.act_quantizer_w_list[w1_idx].n_bits

                bitq2=self.attn2.act_quantizer_q_list[q2_idx].n_bits
                bitk2=self.attn2.act_quantizer_k_list[k2_idx].n_bits
                bitv2=self.attn2.act_quantizer_v_list[v2_idx].n_bits
                bitw2=self.attn2.act_quantizer_w_list[w2_idx].n_bits
            matmul_ops=matmul_macs*((bitq1*bitk1+bitw1*bitv1)+(bitq2*bitk2+bitw2*bitv2))*1e-9
            list_QKV.append(matmul_ops)

        handler_list=[]
        ans=0
        for i in range(len(use_bitwidth)):#for i in range(len(use_bitwidth)):#为了效率，目前就用一步的设置了，多的不设置了
            #use_bitwidth[i]
            cnt=0
            list_conv2d=[]
            list_conv1d=[]
            list_linear=[]
            list_QKV=[]
            for m in self.model.modules():
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                if isinstance(m, (QuantModule_one, QuantAttnBlock_one,QuantAttentionBlock_one, QuantBasicTransformerBlock_one)):
                    if isinstance(m, QuantModule_one) and m.split_quantization == True:
                        assert m.weight_idx_all==cnt
                        assert m.act_idx_all==cnt+1
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1])
                        cnt += 2
                    elif isinstance(m, QuantAttnBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantAttentionBlock_one) and m.split_quantization == True:
                        assert m.q_idx_all==cnt
                        assert m.k_idx_all==cnt+1
                        assert m.v_idx_all==cnt+2
                        assert m.w_idx_all==cnt+3
                        m.set_quant_idx_cand(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1],use_bitwidth[i][cnt + 2],use_bitwidth[i][cnt + 3])
                        cnt += 4
                    elif isinstance(m, QuantBasicTransformerBlock_one) and m.split_quantization == True:
                        assert m.attn1_q_idx_all==cnt
                        assert m.attn1_k_idx_all==cnt+1
                        assert m.attn1_v_idx_all==cnt+2
                        assert m.attn1_w_idx_all==cnt+3

                        assert m.attn2_q_idx_all==cnt+4
                        assert m.attn2_k_idx_all==cnt+5
                        assert m.attn2_v_idx_all==cnt+6
                        assert m.attn2_w_idx_all==cnt+7
                        m.set_quant_idx(use_bitwidth[i][cnt], use_bitwidth[i][cnt + 1], use_bitwidth[i][cnt + 2], use_bitwidth[i][cnt + 3],use_bitwidth[i][cnt + 4],use_bitwidth[i][cnt + 5],use_bitwidth[i][cnt + 6],use_bitwidth[i][cnt + 7])
                        cnt += 8
                    else:
                        logger.info('problem')
                    # elif isinstance(m, QuantModule_one) or isinstance(m, QuantAttnBlock_one) or isinstance(m, (QuantQKMatMul_one, QuantSMVMatMul_one)):
                    #     m.set_quant_idx(use_bitwidth[i][cnt])
                    #     cnt+=1
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttentionBlock_one):
                        handler=m.register_forward_hook(quantAttentionBlock_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantBasicTransformerBlock_one):
                        handler=m.register_forward_hook(quantQuantBasicTransformerBlock_hook)
                        handler_list.append(handler)
            ##for LDM
            logger.info('cnt:{}'.format(cnt))
            if hasattr(self, 'image_size') and hasattr(self, 'image_size'):
                logger.info('LDM image size')
                image_size = self.image_size
                channels = self.in_channels
            else:#for cifar
                logger.info('cifar image size')
                image_size = self.config.data.image_size
                channels = self.config.data.channels
            #调整一下以后用来处理image和txt
            if self.args.task =='cifar' or self.args.task=='lsun':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
                cali_xs,cali_ts=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda())
            elif self.args.task=='image':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,1,512)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            elif self.args.task=='txt':
                cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)), torch.randint(-2,2,(2,77,768)))
                cali_xs,cali_ts,cali_cs=cali_data
                with torch.no_grad():
                    _=self.model(cali_xs.cuda(),cali_ts.cuda(),cali_cs.cuda())
            else:
                raise ValueError('unrecognized')
            for handler in handler_list:
                handler.remove()
            ans+=sum(list_conv2d)+sum(list_conv1d)+sum(list_linear)+sum(list_QKV)
        logger.info('conv2d: {}, conv1d: {}, linear: {}, QKV: {}, all: {}'.format(sum(list_conv2d),sum(list_conv1d),sum(list_linear),sum(list_QKV),ans))
        return ans
    
    def get_all_bitwidth(self,model): # not used, if use, please test, 暂时没有调整，要用的话再调整
        for name, module in model.named_children():
            if isinstance(module,QuantModule_one):
                if module.split_quantization is True:
                    print(module.act_quantizer_list[module.act_quantizer_idx].n_bits)
                    print(module.weight_quantizer_list[module.weight_quantizer_idx].n_bits)
                else:
                    print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            elif isinstance(module,QuantAttnBlock_one):
                print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            else:
                self.get_all_bitwidth(module)
        '''
        for name, module in model.named_children():
            if isinstance(module,QuantModule_one):
                print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            elif isinstance(module,BaseQuantBlock_one):
                print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            else:
                self.get_all_bitwidth(module)
        '''

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):#not_adjusted, need adjust for txt
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock_one):
                for idx in range(len(m.attn1.act_quantizer_w_list)):
                    if sm_only:
                        m.attn1.act_quantizer_w_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_w_list[idx].running_stat = running_stat
                    else:
                        m.attn1.act_quantizer_q_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_k_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_v_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_w_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_q_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_k_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_v_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_w_list[idx].running_stat = running_stat
            if isinstance(m, QuantModule_one) and not sm_only:
                for idx in range(len(m.act_quantizer_list)):
                    m.set_running_stat(running_stat,idx)
    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock_one, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt

class QuantModel_two(nn.Module):

    def __init__(self, model: nn.Module, weight_quant_params_list, act_quant_params_list,**kwargs):#
        super().__init__()
        self.model = model
        self.sm_abit = kwargs.get('sm_abit', 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.specials = get_specials_one(act_quant_params_list[0]['leaf_param'])
        self.quant_module_refactor(self.model, weight_quant_params_list, act_quant_params_list)
        self.quant_block_refactor(self.model, weight_quant_params_list, act_quant_params_list)
        
        self.layer_num=self.get_num()

    def quant_module_refactor(self, module: nn.Module, weight_quant_params_list, act_quant_params_list):
        '''
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        '''
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)): # nn.Conv1d
                setattr(module, name, QuantModule_one(
                    child_module, weight_quant_params_list, act_quant_params_list))
                prev_quantmodule = getattr(module, name)

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(child_module, weight_quant_params_list, act_quant_params_list)

    def quant_block_refactor(self, module: nn.Module, weight_quant_params_list, act_quant_params_list):
        for name, child_module in module.named_children():
            #print(type(child_module))
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] in [QuantBasicTransformerBlock_one, QuantAttnBlock_one]:
                    setattr(module, name, self.specials[type(child_module)](child_module,
                        act_quant_params_list, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantSMVMatMul_one:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params_list, sm_abit=self.sm_abit))
                elif self.specials[type(child_module)] == QuantQKMatMul_one:
                    setattr(module, name, self.specials[type(child_module)](
                        act_quant_params_list))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module, 
                        act_quant_params_list))
            else:
                self.quant_block_refactor(child_module, weight_quant_params_list, act_quant_params_list)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                m.set_quant_state(weight_quant, act_quant)

    def cal_bitops(self,use_bitwidth,image_size,channels):#two dims array
        ####To Do
        list_conv2d=[]
        list_conv1d=[]
        list_linear=[]
        list_QKV=[]
        def quantmodule_hook(self,input,output):
            ins = input[0].size()
            outs = output.size()
            idx=self.use_quantizer_idx
            bitw=self.weight_quantizer_list[idx].n_bits
            bita=self.act_quantizer_list[idx].n_bits
            if self.idx==0: #Conv2d
                # thop method
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops = n_macs * bitw * bita * 1e-9
                list_conv2d.append(n_bitops)
                '''
                #some other method, check the same
                n_macs = (ins[1] * outs[1] *
                self.kernel_size[0] * self.kernel_size[1] *
                outs[2] * outs[3] // self.groups) * outs[0]
                n_bitops2 = n_macs * bitw * bita * 1e-9
                # print(n_bitops1)
                # print(n_bitops2)
                # assert n_bitops1 == n_bitops2
                list_conv2d.append(n_bitops2)
                '''
                
            elif self.idx==1: #Conv1d: not used
                input_size = list(input[0].shape)
                output_size = list(output.shape)
                kernel_size=self.kernel_size
                groups=self.groups
                in_c=input_size[1]
                g=groups
                n_macs=l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
                n_bitops=n_macs*bitw*bita*1e-9
                list_conv1d.append(n_bitops)
            elif self.idx==2:  #Linear:
                #thop
                total_mul = self.in_features
                num_elements = output.numel()
                def calculate_linear(total_mul, num_elements):
                    return int(total_mul * num_elements)
                n_macs=calculate_linear(total_mul, num_elements)
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                #other method
                n_macs = ins[1] * outs[1] * outs[0]
                n_bitops = n_macs * bitw * bita * 1e-9
                list_linear.append(n_bitops)
                '''
                
        def basequantblock_hook(self,input,output):
            b, c, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            matmul_macs = 2 * b * (num_spatial ** 2) * c
            idx=self.use_quantizer_idx
            bita=self.act_quantizer_q_list[idx].n_bits
            matmul_ops=matmul_macs*bita*bita*1e-9
            list_QKV.append(matmul_ops)
            
        def quantattnblock_hook(self,input,output):
            # b,c, *spatial = output[0].shape #256,16,16  疑惑？why都是这样
            # num_spatial = int(np.prod(spatial))
            # matmul_macs = 2 * b * (num_spatial ** 2) *c 
            b, *spatial = output[0].shape
            num_spatial = int(np.prod(spatial))
            matmul_macs = 2 * b * num_spatial * num_spatial
            # matmul_macs = 2 * b * b * num_spatial
            idx=self.use_quantizer_idx
            bita=self.act_quantizer_q_list[idx].n_bits
            matmul_ops=matmul_macs*bita*bita*1e-9
            list_QKV.append(matmul_ops)

        handler_list=[]
        for i in range(len(use_bitwidth)):
            #use_bitwidth[i]
            cnt=0
            for m in self.model.modules():
                #if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                if isinstance(m,(QuantModule_one,BaseQuantBlock_one)):
                    m.set_quant_idx(use_bitwidth[i][cnt])
                    cnt+=1
                    if isinstance(m,QuantModule_one):
                        handler=m.register_forward_hook(quantmodule_hook)
                        handler_list.append(handler)
                    elif isinstance(m,QuantAttnBlock_one):
                        handler=m.register_forward_hook(quantattnblock_hook)
                        handler_list.append(handler)
        image_size = image_size
        channels = channels
        cali_data = (torch.randn(2, channels, image_size, image_size), torch.randint(0, 1000, (2,)))
        cali_xs,cali_ts=cali_data
        with torch.no_grad():
            _=self.model(cali_xs.cuda(),cali_ts.cuda())
        for handler in handler_list:
            handler.remove()
        ans=sum(list_conv2d)+sum(list_conv1d)+sum(list_linear)+sum(list_QKV)
        logger.info('conv2d: {}, conv1d: {}, linear: {}, QKV: {}, all: {}'.format(sum(list_conv2d),sum(list_conv1d),sum(list_linear),sum(list_QKV),ans))
        return ans

    def set_quant_idx(self, i):
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                m.set_quant_idx(i)
    def set_quant_idx_cand(self, cand):
        cnt=0
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                m.set_quant_idx(cand[cnt])
                cnt+=1
    def get_num(self):
        cnt=0
        for m in self.model.modules():
            if isinstance(m, (QuantModule_one, BaseQuantBlock_one)):
                cnt=cnt+1
        return cnt
    def get_all_bitwidth(self,model):
        for name,module in model.named_children():
            if isinstance(module,QuantModule_one):
                print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            elif isinstance(module,BaseQuantBlock_one):
                print(module.act_quantizer_list[module.use_quantizer_idx].n_bits)
            else:
                self.get_all_bitwidth(module)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):#not_adjusted
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock_one):
                for idx in range(len(m.attn1.act_quantizer_w_list)):
                    if sm_only:
                        m.attn1.act_quantizer_w_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_w_list[idx].running_stat = running_stat
                    else:
                        m.attn1.act_quantizer_q_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_k_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_v_list[idx].running_stat = running_stat
                        m.attn1.act_quantizer_w_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_q_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_k_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_v_list[idx].running_stat = running_stat
                        m.attn2.act_quantizer_w_list[idx].running_stat = running_stat
            if isinstance(m, QuantModule_one) and not sm_only:
                for idx in range(len(m.act_quantizer_list)):
                    m.set_running_stat(running_stat,idx)
    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            if isinstance(m, (QuantBasicTransformerBlock_one, BasicTransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt

