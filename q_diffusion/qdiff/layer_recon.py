import torch
import random
import linklink as link
import logging
from qdiff.quant_layer import StraightThrough, lp_loss,QuantModule_one,UniformAffineQuantizer
from qdiff.quant_model import QuantModel_one
from qdiff.block_recon import LinearTempDecay
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_inp_oup_data
import torch.nn.functional as F
import pdb
import os

logger = logging.getLogger(__name__)
logger.info('test')


def layer_reconstruction_one_respective(model: QuantModel_one, layer: QuantModule_one, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False, cond: bool = False, is_sm: bool = False, grad_accumulation: bool = False):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    :param cond: conditional generation or not
    :param is_sm: avoid OOM when caching n^2 attention matrix when n is large
    """

    model.set_quant_state(False, False)
    layer.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        if layer.split != 0:
                for i in range(len(layer.weight_quantizer_list)):
                    layer.weight_quantizer_list[i] = AdaRoundQuantizer(uaq=layer.weight_quantizer_list[i], round_mode=round_mode,
                                                        weight_tensor=layer.org_weight.data[:, :layer.split, ...])
                    layer.weight_quantizer_0_list[i] = AdaRoundQuantizer(uaq=layer.weight_quantizer_0_list[i], round_mode=round_mode,
                                                        weight_tensor=layer.org_weight.data[:, layer.split:, ...])
        else:
            for i in range(len(layer.weight_quantizer_list)):
                layer.weight_quantizer_list[i] = AdaRoundQuantizer(uaq=layer.weight_quantizer_list[i], round_mode=round_mode,
                                                   weight_tensor=layer.org_weight.data)
        for i in range(len(layer.weight_quantizer_list)):
            layer.weight_quantizer_list[i].soft_targets = True #in Adaround
        if layer.split !=0:
            for i in range(len(layer.weight_quantizer_0_list)):
                layer.weight_quantizer_0_list[i].soft_targets = True

        # Set up optimizer
        opt_params = [layer.weight_quantizer_list[i].alpha for i in range(len(layer.weight_quantizer_list))]
        if layer.split != 0:
            for i in range(len(layer.weight_quantizer_0_list)):
                opt_params += [layer.weight_quantizer_0_list[i].alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        opt_params = [layer.act_quantizer_list[i].delta for i in range(len(layer.act_quantizer_list))]
        if layer.split != 0:
            for i in range(len(layer.act_quantizer_0_list)):
                if layer.act_quantizer_0_list[i].delta is not None:
                    opt_params += [layer.act_quantizer_0_list[i].delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode
    for jj in range(len(layer.weight_quantizer_list)):
        model.set_quant_idx(jj)
        if grad_accumulation == False:
            loss_func = LossFunction_one(layer, round_loss=loss_mode, weight=weight,
                                max_count=iters, rec_loss=rec_loss, b_range=b_range,
                                decay_start=0, warmup=warmup, p=p)
        else:
            loss_func = LossFunction_one(layer, round_loss=loss_mode, weight=weight,
                                max_count=2*iters, rec_loss=rec_loss, b_range=b_range,
                                decay_start=0, warmup=warmup, p=p)

        # Save data before optimizing the rounding
        # cached_inps, cached_outs = save_inp_oup_data(
        #     model, layer, cali_data, asym, act_quant, batch_size, keep_gpu=False, cond=cond, is_sm=is_sm)
        cached_inps, cached_outs = save_inp_oup_data(#batch_size_maybe smaller? for lsun
            model, layer, cali_data, asym, act_quant, 32, keep_gpu=False, cond=cond, is_sm=is_sm)
        if opt_mode != 'mse':
            cached_grads = save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
        else:
            cached_grads = None
        device = 'cuda'
        for i in range(iters):
            if grad_accumulation == False:
                inner_iters = 1
            else:
                inner_iters = 2
            ##logger.info('set_quant_idx_layer:{}'.format(use_quantizer))####
            idx = torch.randperm(cached_inps.size(0))[:batch_size]
            offset_ = len(idx)//inner_iters
            optimizer.zero_grad()
            for k in range(inner_iters):
                if grad_accumulation == True:
                    torch.cuda.empty_cache()
                idx_now = idx[k*offset_ : (k+1)*offset_]
                cur_inp = cached_inps[idx_now].to(device)
                cur_out = cached_outs[idx_now].to(device)
                cur_grad = cached_grads[idx_now] if opt_mode != 'mse' else None
                out_quant = layer(cur_inp)

                err = loss_func(out_quant, cur_out, cur_grad)
                err.requires_grad_(True)
                err.backward(retain_graph=True)
                if multi_gpu:
                    for p in opt_params:
                        link.allreduce(p.grad)
            optimizer.step()
            if scheduler:
                scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for i in range(len(layer.weight_quantizer_list)):
        layer.weight_quantizer_list[i].soft_targets = False
    if layer.split != 0:
        for i in range(len(layer.weight_quantizer_0_list)):
            layer.weight_quantizer_0_list[i].soft_targets = False

    # Reset original activation function
    if not include_act_func:
        layer.activation_function = org_act_func


    
class LossFunction_one:
    def __init__(self,
                 layer: QuantModule_one,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        if self.layer.split != 0:
            logger.info('split reconstruction')

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            if self.layer.split == 0:
                round_vals = self.layer.weight_quantizer_list[self.layer.use_quantizer_idx].get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                round_vals = self.layer.weight_quantizer_list[self.layer.use_quantizer_idx].get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                round_vals = self.layer.weight_quantizer_0_list[self.layer.use_quantizer_idx].get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
