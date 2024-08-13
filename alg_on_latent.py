#qdiff_cifar_mixed_offload.py
import argparse, os, gc, glob, datetime, yaml
import logging
import math
from qdiff.utils import convert_adaround
import copy
import numpy as np
import tqdm
import torch
import torch as th
import torch.nn as nn
from torch.cuda import amp
from tqdm import trange
import pdb
from PIL import Image
from pytorch_lightning import seed_everything
import random
import torch.distributed as dist
from ddim.models.diffusion import Model
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
import time
import torchvision.utils as tvu
import copy


from qdiff.quant_model import QuantModel_one
from qdiff.quant_block import BaseQuantBlock_one,QuantAttentionBlock_one
from qdiff.quant_layer import QuantModule_one
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import get_train_samples,resume_cali_model_one
from qdiff.layer_recon import layer_reconstruction_one_respective
from qdiff.block_recon import block_reconstruction_one_respective

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    create_classifier,
    classifier_defaults,
)

logger = logging.getLogger(__name__)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

def space_timesteps_one(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]    
    size_per = num_timesteps // len(section_counts)   
    extra = num_timesteps % len(section_counts)    
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)    
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)    
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    a=set(all_steps)
    b=[i for i in a]
      
      
    return a
      

  
  
  
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

rescale = lambda x: (x + 1.) / 2.





def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_np(x):
      
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample_timesteps(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
      
    return samples, intermediates

@torch.no_grad()
def convsample_dpm(model, steps, shape, eta=1.0
                    ):
    raise ValueError('not implemented')
    dpm = DPMSolverSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = dpm.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def convsample_plms(model, steps, shape, eta=1.0
                    ):
    raise ValueError('not implemented')
    plms = PLMSSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = plms.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates

@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, dpm=False, plms = False):
    log = dict()
    custom_steps = sorted(custom_steps)
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

      
    t0 = time.time()
    if vanilla:
        sample, progrow = convsample(model, shape,
                                        make_prog_row=True)
    elif plms:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_plms(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    elif dpm:
        logger.info(f'Using DPM sampling with {custom_steps} sampling steps and eta={eta}')
        sample, intermediates = convsample_dpm(model,  steps=custom_steps, shape=shape,
                                                eta=eta)
    else:
        sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                eta=eta)

    t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    logger.info(f'Throughput for this batch: {log["throughput"]}')
    return log

def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model



def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        logger.info(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step

class LatentDiffusion(object):
    def __init__(self, args, config,device=None):
        self.args = args

        assert self.args.model_ckpt != 'no'
        resume_base = self.args.model_ckpt
          
        base_configs = sorted(glob.glob(args.config))
        self.args.base = base_configs
        configs = [OmegaConf.load(cfg) for cfg in self.args.base]
        cli = OmegaConf.from_dotlist([])
        self.config = OmegaConf.merge(*configs, cli)
          
          
        self.imglogdir = os.path.join(logdir, "img")
        self.numpylogdir = os.path.join(logdir, "numpy")

        os.makedirs(self.imglogdir, exist_ok = True)
        os.makedirs(self.numpylogdir, exist_ok = True)
        self.device = device
        self.weight_list=[int(i) for i in args.weight_bit]
        self.act_list=[int(i) for i in args.act_bit]
        self.model=self.get_model(self.weight_list,self.act_list,'cifar5678',cali_ckpt=args.cali_ckpt)
          
    '''
    def set_use_timestep(self,x):
        self.use_timesteps=x
    '''
    def forward_model(self,model,cali_xs,cali_ts):
        for i in range(len(self.weight_list)):
            model.set_quant_idx(i)
            _ = model(cali_xs, cali_ts)
    def get_model(self,weight_list,act_list,ckptsavename,cali_ckpt='no'):  
        gpu = True
        eval_mode = True
        
        model, global_step = load_model(self.config, self.args.model_ckpt , gpu, eval_mode)
        logger.info(f"global step: {global_step}")
        logger.info("Switched to EMA weights")
        model.model_ema.store(model.model.parameters())
        model.model_ema.copy_to(model.model)
        if self.args.split and self.args.ptq:
            setattr(model.model.diffusion_model,"split", True)
          
          
          
        assert(self.args.cond == False)

        if self.args.ptq:
            if self.args.quant_mode == 'qdiff':
                a_scale_method = 'mse' if not args.a_min_max else 'max'
                wq_params = [{'n_bits': weight_bit, 'channel_wise': True, 'scale_method': 'max'} for weight_bit in weight_list]
                aq_params = [{'n_bits': act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act} for act_bit in act_list]
                if self.args.resume:
                    logger.info('Load with min-max quick initialization')
                    for i in range(len(wq_params)):
                        wq_params[i]['scale_method'] = 'max'
                        aq_params[i]['scale_method'] = 'max'
                if self.args.resume_w:
                    for i in range(len(wq_params)):
                        wq_params[i]['scale_method'] = 'max'
                  
                logger.info(model.model.diffusion_model)
                qnn = QuantModel_one(
                    model=model.model.diffusion_model, args=self.args,weight_quant_params_list=wq_params, act_quant_params_list=aq_params, 
                    sm_abit=self.args.sm_abit,config=self.config)
                  
                logger.info(qnn)
                qnn.to(self.device)
                qnn.eval()

                i=0
                logger.info(f"Sampling data from {self.args.cali_st} timesteps for calibration")
                sample_data = torch.load(self.args.cali_data_path)
                cali_data = get_train_samples(self.args, sample_data, custom_steps=0)
                del(sample_data)
                gc.collect()
                if self.args.resume:
                      
                      
                      
                    resume_cali_model_one(qnn, cali_ckpt, cali_data, len(self.weight_list), args.quant_act, "qdiff", cond=False)
                      
                      
                    pass
                else:
                    logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")
                    cali_xs, cali_ts = cali_data
                    if self.args.resume_w:
                        resume_cali_model_one(qnn, cali_ckpt, cali_data, len(self.weight_list), False, cond=False)
                          
                    else:
                        logger.info("Initializing weight quantization parameters")
                        qnn.set_quant_state(True, False)   
                        self.forward_model(qnn,cali_xs[:8].cuda(),cali_ts[:8].cuda())
                        logger.info("Initializing has done!")
                          
                          
                    kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size, 
                                iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                warmup=0.2, act_quant=False, opt_mode='mse')
                    
                    def recon_model(model):
                        """
                        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                        """

                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock_one)}")
                            if isinstance(module, QuantModule_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of layer {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for layer {}'.format(name))
                                    layer_reconstruction_one(qnn, module, **kwargs)
                            elif isinstance(module, BaseQuantBlock_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of block {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for block {}'.format(name))
                                    block_reconstruction_one(qnn, module, **kwargs)
                            else:
                                recon_model(module)
                    
                    def recon_model_respective(model):
                        """
                        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                        """

                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock_one)}")
                            if isinstance(module, QuantModule_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of layer {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for layer {}'.format(name))
                                    layer_reconstruction_one_respective(qnn, module, **kwargs)
                            elif isinstance(module, BaseQuantBlock_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of block {}'.format(name))
                                    continue
                                else:
                                    logger.info('Reconstruction for block {}'.format(name))
                                    block_reconstruction_one_respective(qnn, module, **kwargs)
                            else:
                                recon_model_respective(module)

                    if args.assigning_prob == True:
                        recon_model_func = recon_model
                    else:
                        recon_model_func = recon_model_respective
                    if not self.args.resume_w:
                        logger.info("Doing weight calibration")
                        recon_model_func(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=False)
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                    torch.save(qnn.state_dict(), os.path.join(self.args.logdir,ckptsavename+ "ckpt.pth"))
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            zero_data = m.zero_point.data
                            delattr(m, "zero_point")
                            m.zero_point = zero_data

                            delta_data = m.delta.data
                            delattr(m, "delta")
                            m.delta = delta_data
                    if self.args.quant_act:
                        logger.info("UNet model")
                        logger.info(model.model)                    
                        logger.info("Doing activation calibration")   
                          
                        qnn.set_quant_state(True, True)
                        with torch.no_grad():
                              
                            inds = np.random.choice(cali_xs.shape[0], 32, replace=False)
                              
                            self.forward_model(qnn,cali_xs[inds].cuda(), cali_ts[inds].cuda())
                            if self.args.running_stat:  
                                  
                                logger.info('Running stat for activation quantization')
                                qnn.set_running_stat(True)
                                for i in range(int(cali_xs.size(0) / 32)):
                                    _ = qnn(
                                        (cali_xs[i * 32:(i + 1) * 32].to(self.device), 
                                        cali_ts[i * 32:(i + 1) * 32].to(self.device)))
                                qnn.set_running_stat(False)
                        
                        kwargs = dict(
                            cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True, 
                            opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p, batch_size = self.args.cali_batch_size)   
                        recon_model_func(qnn)
                        qnn.set_quant_state(weight_quant=True, act_quant=True)   
                    logger.info("Saving calibrated quantized UNet model")
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                        elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                            if m.zero_point is not None:
                                if not torch.is_tensor(m.zero_point):
                                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                else:
                                    m.zero_point = nn.Parameter(m.zero_point)
                      
                          
                              
                      
                    torch.save(qnn.state_dict(), os.path.join(self.args.logdir,ckptsavename+ "ckpt.pth"))
                if self.args.split_quantization:
                    qnn.set_split_quantization(True)  
                model.layer_num = qnn.get_num_offload()
                qnn.layer_num=model.layer_num
                logger.info(model.layer_num)
                if self.args.deterministic_retrain:
                    assert args.cand_for_performance is not None
                    use_bitwidth=[]
                    use_bitwidth.append(eval(args.cand_for_performance)['use_bitwidth'])
                    qnn.cal_bitops(use_bitwidth)  
                    def recon_model_retrain(model):
                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock_one)}")
                            if isinstance(module, QuantModule_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of layer {}'.format(name))
                                    continue
                                else:
                                    if qnn.use_act_quant:
                                        logger.info('Reconstruction for layer {}'.format(name))
                                        layer_reconstruction_retrain(qnn, module, **kwargs)
                            elif isinstance(module, BaseQuantBlock_one):
                                if module.ignore_reconstruction is True:
                                    logger.info('Ignore reconstruction of block {}'.format(name))
                                    continue
                                else:
                                      
                                    logger.info('Reconstruction for block {}'.format(name))
                                    block_reconstruction_retrain(qnn, module, **kwargs)
                            else:
                                recon_model_retrain(module)
                      
                      
                    kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size, 
                                iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                                warmup=0.2, act_quant=False, opt_mode='mse')
                        
                    qnn.set_quant_state(True,False)
                    recon_model_retrain(qnn)
                    qnn.set_quant_state(True,True)
                    kwargs = dict(
                            cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True, 
                            opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p, batch_size = self.args.cali_batch_size)   

                    recon_model_retrain(qnn)
                    for m in qnn.model.modules():
                        if isinstance(m, AdaRoundQuantizer):
                            m.zero_point = nn.Parameter(m.zero_point)
                            m.delta = nn.Parameter(m.delta)
                        elif isinstance(m, UniformAffineQuantizer) and self.args.quant_act:
                            if m.zero_point is not None:
                                if not torch.is_tensor(m.zero_point):
                                    m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                                else:
                                    m.zero_point = nn.Parameter(m.zero_point)
                    torch.save(qnn.state_dict(), os.path.join(self.args.logdir,ckptsavename+ "ckpt.pth"))
                qnn.to(self.device)
                qnn.eval()
                if self.args.deterministic_retrain is False:
                    use_bitwidth=[]
                    use_bitwidth.append([0 for i in range(model.layer_num)])
                    list_conv, list_QKV, list_trans, list_conv_idx, list_QKV_idx, list_trans_idx = qnn.cal_bitops_offload(use_bitwidth)
                      
                      
                    dict_conv=dict()  
                    dict_QKV=dict()  
                    dict_trans=dict()  
                    def set_dict(dict_,list_,list_idx):
                        for i in range(len(list_)):
                            dict_[i]=list_idx[i]
                    set_dict(dict_conv, list_conv, list_conv_idx)
                    set_dict(dict_QKV, list_QKV, list_QKV_idx)
                    set_dict(dict_trans, list_trans, list_trans_idx)
                    with open(os.path.join(args.logdir,'data.txt'),'w') as f:
                        f.write(str(list_conv))
                        f.write('\n')
                        f.write(str(dict_conv))
                        f.write('\n')
                        f.write(str(list_QKV))
                        f.write('\n')
                        f.write(str(dict_QKV))
                        f.write('\n')
                        f.write(str(list_trans))
                        f.write('\n')
                        f.write(str(dict_trans))
                        f.write('\n')
                model.model.diffusion_model = qnn
        return model
    def set_use_timestep(self, use_bitwidth):
        raise ValueError('not implemented')
    def sample_image(self, cand = None, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
        nums =None, nplog=None):
        if args.save_memory:
            batch_size=25
        eta = self.args.eta
        diffusion_model = self.model.model.diffusion_model
        model = self.model
        if nums is None:
            logger.info('sample num_images')
            total_n_samples = self.args.num_images
        else:
            logger.info('sample given number')
            total_n_samples = nums

        if vanilla:
            logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
        else:
            logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


        tstart = time.time()
          
        cnt = 0
        n_saved = 0
          
          
          
        if model.cond_stage_model is None:
            all_images = []

            logger.info(f"Running unconditional sampling for {total_n_samples} samples")
            for _ in trange(total_n_samples // batch_size, desc="Sampling Batches (unconditional)"):
                  
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=cand['timestep'],
                                                eta=eta, dpm=self.args.dpm_solver,plms = self.args.plms)
                n_saved += batch_size
                  
                all_images.extend([custom_to_np(logs["sample"])])
                if n_saved >= total_n_samples:
                    logger.info(f'Finish after generating {n_saved} samples')
                    break
            all_img = np.concatenate(all_images, axis=0)
            all_img = all_img[:total_n_samples]
              
            return all_img

        else:
            raise NotImplementedError('Currently only sampling for unconditional models supported.')

        logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    def sample_image_file(self, cand = None, batch_size=50, vanilla=False, custom_steps=None, eta=None, 
        nums = 50000, nplog=None, dpm=False,timestep_num=None):
          
        if args.save_memory:
            batch_size=25
        save_dir=os.path.join(self.args.logdir, str(timestep_num))
        os.makedirs(save_dir)
        model = self.model
        eta = self.args.eta
        if nums is None:
            logger.info('sample num_images')
            total_n_samples = self.args.num_images
        else:
            logger.info('sample given number')
            total_n_samples = nums

        if vanilla:
            logger.info(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
        else:
            logger.info(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

        if timestep_num==None:
            timestep_num=len(cand['timestep'])
        tstart = time.time()
          
        cnt = 0
        n_saved = 0
          
          
          
        if model.cond_stage_model is None:
            
            logger.info(f"Running unconditional sampling for {total_n_samples} samples")
            for _ in trange(total_n_samples // batch_size, desc="Sampling Batches (unconditional)"):
                  
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                vanilla=vanilla, custom_steps=cand['timestep'],
                                                eta=eta, dpm=self.args.dpm_solver,plms = self.args.plms)
                n_saved += batch_size
                  
                for i in range(len(logs["sample"])):
                    imgs_now=custom_to_pil(logs["sample"][i])
                    imgs_now.save(os.path.join(save_dir,f'{cnt}.png'),'PNG')
                    cnt+=1
                  
                  
                  
                  
                  
                  
                  
                  
                if n_saved >= total_n_samples:
                    logger.info(f'Finish after generating {n_saved} samples')
                    break



        else:
            raise NotImplementedError('Currently only sampling for unconditional models supported.')

        logger.info(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

        


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deterministic_retrain',action='store_true')
    parser.add_argument('--cand_for_performance',type=str,default=None)
    parser.add_argument('--timestep_for_test_sample', type=str, default='quad')
    parser.add_argument('--set_7',action='store_true')
    parser.add_argument('--get_some_performance', action='store_true')
    parser.add_argument('--only_search_weight', action='store_true')
    parser.add_argument('--target_bitops', type = float, default = None)
    parser.add_argument('--task',type=str,choices=['cifar','lsun','image','txt'])
    parser.add_argument('--offload_file', type=str, default=None)
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument('--not_attn', action = 'store_true', help = 'whether to quant qkv activation')
    parser.add_argument('--save_memory', action = 'store_true', help = 'run LDM on 24GB')
    parser.add_argument('--only_steps', action = 'store_true', help = 'only calibrate timestep not bitwidth')
    parser.add_argument('--txt2img', action = 'store_true', help = 'whether to do txt2img')
    parser.add_argument('--assigning_prob', action = 'store_true', help = 'whether to prod_reconstruction')
    parser.add_argument('--split_quantization', action = 'store_true', help = 'whether to split quantize')
    parser.add_argument('--a_min_max', action = 'store_true', help = 'act quantizers initialize with min-max (empirically helpful in some cases)')
    parser.add_argument('--model_type', type = str, default = 'trivial', help = 'which kind of model to be used')
    parser.add_argument('--model_ckpt', type = str, default = 'no', help = 'FP model path')
    parser.add_argument('--group_method', default = 'quad', help = "if timestep groupwise, how to sample groups")
    parser.add_argument('--timestep_quant_mode', default = 'no_constraint', help = "how to quant along timestep", choices = ['no_constraint', 'groupwise', 'consistent', 'groupwise_consistent'])
    parser.add_argument('--cand_file',type=str,default=None)
    parser.add_argument('--test',action="store_true",help="whether to test quant performance")
    parser.add_argument('--num_images',type=int,default=2048)  
    parser.add_argument('--origin',action="store_true")
    parser.add_argument('--cand_timestep',type=str,default='no')
    parser.add_argument('--cand_use_bitwidth',type=str,default='no')
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
      
    parser.add_argument('--ckpt_pre',type=str,default=None)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        nargs='+',type=str,default=['8','6','4']
    )
    parser.add_argument(
        "--act_bit",
        nargs='+',type=str,default=['8','6','4']
    )
    parser.add_argument(
        "--quant_mode", type=str, default="qdiff", 
        choices=["qdiff"], 
        help="quantization mode to use"
    )
    parser.add_argument(
        "--max_images", type=int, default=50000, help="number of images to sample"
    )

      
    parser.add_argument(
        "--cali_st", type=int, default=20, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=256, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=80000,   
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=20000, type=int,   
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--a_sym", action="store_true",
        help="act quantizers use symmetric quantization"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument("--split", action="store_true",
        help="split shortcut connection into two parts"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    parser.add_argument(
        "--out_path",type=str
    )
    parser.add_argument('--max_epochs',type=int,default=20)
    parser.add_argument('--select_num',type=int,default=10)
    parser.add_argument('--population_num',type=int,default=50)
    parser.add_argument('--m_prob',type=float,default=0.1)
    parser.add_argument('--crossover_num',type=int,default=25)
    parser.add_argument('--mutation_num',type=int,default=35)
    parser.add_argument('--max_fid',type=float,default=48.0)
    parser.add_argument('--thres',type=float,default=0.2)
    parser.add_argument('--init_x',type=str,default='')
      
    parser.add_argument('--use_ddim_init_x',type=bool,default=False)
    parser.add_argument('--use_ddim',type=bool,default=False)
    parser.add_argument('--time_step',type=int,default=100)

    return parser


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class EvolutionSearcher(object):

    def __init__(self, args,diffusion, time_step):
          
        self.args = args
          
        with open(self.args.offload_file,'r') as f:
            self.selection_dict=eval(f.readlines()[0])
        self.selection_list = list(self.selection_dict.keys())
        self.selection_list = [eval(i) for i in self.selection_list]
        if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
            if self.args.group_method == 'quad':
                seq=(
                    np.linspace(
                        0, np.sqrt(1000 * 0.8), self.args.time_step   
                    )
                    ** 2
                )
            elif self.args.group_method == 'uniform':
                seq = range(0,1000,1000//self.args.time_step)
            else:
                raise ValueError('not implemented! check for group_method pls')
            seq = [int(s) for s in list(seq)[1:]]
            self.time_list=[]
            for i in range(len(seq)):
                if i==0 and seq[0]!=0:
                    self.time_list.append(list(range(0,seq[0])))
                elif i!=0 and seq[i-1]!=seq[i]:
                    self.time_list.append(list(range(seq[i-1],seq[i])))
            self.time_list.append(list(range(seq[-1],1000)))
        self.diffusion=diffusion
        self.time_step = time_step
        self.flops_constrain=True
          
          
          
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
          
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.vis_dict = {}
        if self.args.ptq:
            self.layer_num=self.diffusion.model.layer_num
        else:
            self.layer_num=self.diffusion.model.layer_num=30
        logger.info('layer_num:{}'.format(self.layer_num))
        self.max_fid = args.max_fid
        self.thres = args.thres
        
          
          
          

        self.x0 = args.init_x  

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True    
        )
        config.gpu_options.allow_growth = True
        self.evaluator = Evaluator_v1(tf.Session(config=config))
        self.evaluator.warmup()
        self.ref_stats = None
          
        if 6 in self.diffusion.weight_list and self.args.target_bitops is None and self.args.ptq:
            if self.args.set_7 == False:
                index_6=self.diffusion.weight_list.index(6)
            else: 
                index_6=self.diffusion.weight_list.index(7)
            cand_now=[[index_6 for j in range(self.layer_num)]for i in range(self.time_step)]
              
            if isinstance(self.diffusion, Diffusion):
                if self.args.timestep_quant_mode == 'no_constraint':
                    self.max_bitops_down=self.diffusion.model.cal_bitops_cand_offload_no_constraint(cand_now)
                    self.max_bitops=self.max_bitops_down
                else:
                    self.max_bitops_down=self.diffusion.model.cal_bitops_cand_offload(cand_now)
                    self.max_bitops=self.max_bitops_down
            else:
                self.max_bitops_down=self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(cand_now)
                self.max_bitops=self.max_bitops_down
              
        if self.args.target_bitops is not None:
            self.max_bitops_down=self.max_bitops=self.args.target_bitops
        self.last_best_cand = None
          
          
          
          

    def reset_diffusion(self, use_timesteps):
          
        self.diffusion.set_use_timestep(use_timesteps)

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        logger.info('select ......')  
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def sample_active_subnet(self):  
          
        if isinstance(self.diffusion, Diffusion):
            original_num_steps = self.diffusion.original_num_steps
        elif isinstance(self.diffusion, LatentDiffusion):
            original_num_steps = self.diffusion.model.num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        if self.args.timestep_quant_mode =='consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
            use_bit_width_one = [random.randint(0,len(self.diffusion.weight_list)-1)for j in range(self.layer_num)]
            use_bitwidth = [copy.deepcopy(use_bit_width_one) for i in range(self.time_step)]  
        else:
            use_bitwidth=[[random.randint(0,len(self.diffusion.weight_list)-1)for j in range(self.layer_num)]for i in range(self.time_step)]
          
        if self.args.timestep_quant_mode == 'no_constraint' or self.args.timestep_quant_mode == 'consistent':
            use_timestep = [i for i in range(original_num_steps)]
            random.shuffle(use_timestep)
            use_timestep = use_timestep[:self.time_step]
        elif self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
            time_list=self.time_list
            use_timestep=[random.choice(i) for i in time_list]
          
          
        if self.args.timestep_quant_mode != 'no_constraint':
            idx_in_selection_list=0
            select_cand=self.selection_list.pop(idx_in_selection_list)
            use_bitwidth=[copy.deepcopy(select_cand) for i in range(self.time_step)]
        else:
            use_bitwidth=[]
            for ___ in range(self.time_step):
                idx_in_selection_list=random.randint(0,len(self.selection_list)-1)
                select_cand=self.selection_list[idx_in_selection_list]
                use_bitwidth.append(copy.deepcopy(select_cand))
        if args.only_steps == True:
            use_bitwidth=[[2 for i in m] for m in use_bitwidth]
        return {'timestep':use_timestep,'use_bitwidth':use_bitwidth}
    
    def is_legal_before_search(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.info('cand has visited!')  
            return False
        info['visited'] = True
        cand=eval(cand)
        if self.flops_constrain == True:
            use_bitwidth=cand['use_bitwidth']
              
            if isinstance(self.diffusion, Diffusion):
                if self.args.timestep_quant_mode =='no_constraint':
                    bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
                else:
                    bitops_now=self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
            elif isinstance(self.diffusion, LatentDiffusion):
                bitops_now=self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')  
                info['fid']=9999
                return False
          
          
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
          
        if self.flops_constrain == True:
            logger.info('cand_timestep: {}, fid: {}, bitops:{}'.format(cand['timestep'], info['fid'],info['bitops']))  
        else:
            logger.info('cand_timestep: {}, fid: {}'.format(cand['timestep'], info['fid']))  
        return True

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logger.info('cand has visited!')  
            return False
          
        info['visited'] = True
        cand=eval(cand)
        if self.flops_constrain:
            use_bitwidth=cand['use_bitwidth']
              
            if isinstance(self.diffusion, Diffusion):
                if self.args.timestep_quant_mode == 'no_constraint':
                    bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
                else:
                    bitops_now=self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
            elif isinstance(self.diffusion, LatentDiffusion):
                bitops_now=self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')  
                info['fid']=9999
                return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
          
        if self.flops_constrain == True:
            logger.info('cand_timestep: {}, fid: {}, bitops:{}'.format(cand['timestep'], info['fid'], info['bitops']))  
        else:
            logger.info('cand_timestep: {}, fid: {}'.format(cand['timestep'], info['fid']))  
          
          
          

        return True

    def get_cand_fid(self, cand=None, args=None, nums=None,index=None):
          
        t1 = time.time()
        if isinstance(self.diffusion, Diffusion):
            self.reset_diffusion(cand['timestep'])
              
            arr=self.diffusion.sample_image(cand['use_bitwidth'],nums=nums)
        else:
            self.diffusion.model.model.diffusion_model.set_quant_idx_cand(cand['use_bitwidth'][0])
            arr = self.diffusion.sample_image(cand, nums = nums)
        sample_time = time.time() - t1
        t1 = time.time()
        
        from evaluations.evaluator_v1 import cal_fid, FIDStatistics,get_statistics
        if self.ref_stats is None:
            if isinstance(self.diffusion, Diffusion):
                self.ref_stats=get_statistics(self.evaluator,'/home/Datasets/cifar10_png/cifar_train.npz')
            elif isinstance(self.diffusion, LatentDiffusion):
                if 'churches' in self.args.model_ckpt:
                    self.ref_stats=get_statistics(self.evaluator,'/home/Datasets/lsun/lsun_church_50000.npz',batch_size = 25)
                elif 'beds' in self.args.model_ckpt:
                    self.ref_stats=get_statistics(self.evaluator,'/home/Datasets/lsun/lsun_bedroom_50000.npz',batch_size = 25)
                else:
                    raise ValueError('not implemented')
        fid = cal_fid(arr, 64, self.evaluator, ref_stats=self.ref_stats)
        if index!=None:
            np.savez(os.path.join(self.args.logdir,str(index)+'.npz'),arr_0=arr)
        fid_time = time.time() - t1
        logger.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))  
        return fid

    def get_cand_fid_file(self, cand=None, args=None, nums=None,timestep_num=None):
          
        t1 = time.time()
        if isinstance(self.diffusion, Diffusion):
              
            self.reset_diffusion(cand['timestep'])
              
            arr=self.diffusion.sample_image_file(cand['use_bitwidth'],nums=nums)
        else:
            if self.args.ptq:
                self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(cand['use_bitwidth'])
                self.diffusion.model.model.diffusion_model.set_quant_idx_cand(cand['use_bitwidth'][0])
            self.diffusion.sample_image_file(cand, nums = nums,timestep_num=timestep_num)
        sample_time = time.time() - t1
        t1 = time.time()
        

    def get_random_before_search(self, num):
          
          
        logger.info('random select ........')  
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))  
        logger.info('random_num = {}'.format(len(self.candidates)))  

    def get_random(self, num):
        logger.info('random select ........')  
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))  
        logger.info('random_num = {}'.format(len(self.candidates)))  

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logger.info('cross ......')  
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k])
            cand2 = choice(self.keep_top_k[k])

            new_cand = dict()
            new_cand['timestep']=[]
            if self.args.timestep_quant_mode == 'consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
                new_cand_bit=[]
            else:
                new_cand['use_bitwidth']=[]
            cand1 = eval(cand1)
            cand2 = eval(cand2)

            length=min(len(cand1['timestep']),len(cand2['timestep']))

            for i in range(length):
                if np.random.random_sample() < 0.5:
                    new_cand['timestep'].append(cand1['timestep'][i])
                    if self.args.timestep_quant_mode != 'consistent' and self.args.timestep_quant_mode != 'groupwise_consistent':
                        new_cand['use_bitwidth'].append(copy.deepcopy(cand1['use_bitwidth'][i]))
                else:
                    new_cand['timestep'].append(cand2['timestep'][i])
                    if self.args.timestep_quant_mode != 'consistent' and self.args.timestep_quant_mode != 'groupwise_consistent':
                        new_cand['use_bitwidth'].append(copy.deepcopy(cand2['use_bitwidth'][i]))

            if new_cand['timestep'] < cand1['timestep']:
                new_cand['timestep'] += copy.deepcopy(cand1['timestep'][len(new_cand['timestep']):])
                if self.args.timestep_quant_mode != 'consistent' and self.args.timestep_quant_mode != 'groupwise_consistent':
                    new_cand['use_bitwidth'] += copy.deepcopy(cand1['use_bitwidth'][len(new_cand['use_bitwidth']):])
            
            if new_cand['timestep'] < cand2['timestep']:
                new_cand['timestep'] += copy.deepcopy(cand2['timestep'][len(new_cand['timestep']):])
                if self.args.timestep_quant_mode != 'consistent' and self.args.timestep_quant_mode != 'groupwise_consistent':
                    new_cand['use_bitwidth'] += copy.deepcopy(cand2['use_bitwidth'][len(new_cand['use_bitwidth']):])
            
            if self.args.timestep_quant_mode == 'consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
                len_layer = min(len(cand1['use_bitwidth'][0]),len(cand2['use_bitwidth'][0]))
                for i in range(len_layer):
                    if np.random.random_sample() < 0.5:
                        new_cand_bit.append(cand1['use_bitwidth'][0][i])
                    else:
                        new_cand_bit.append(cand2['use_bitwidth'][0][i])
                if len(new_cand_bit) < len(cand1['use_bitwidth'][0]):
                    new_cand_bit += copy.deepcopy(cand1['use_bitwidth'][0][len(new_cand_bit):])
                if len(new_cand_bit) < len(cand2['use_bitwidth'][0]):
                    new_cand_bit += copy.deepcopy(cand2['use_bitwidth'][0][len(new_cand_bit):])
                new_cand['use_bitwidth']=[copy.deepcopy(new_cand_bit) for i in range(len(new_cand['timestep']))]
            if self.args.only_search_weight:
                self.diffusion.model.model.diffusion_model.change_act(new_cand['use_bitwidth'])
            return new_cand

        while len(res) < cross_num:  
              
            cand = random_cross()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('cross {}/{}'.format(len(res), cross_num))  

        logger.info('cross_num = {}'.format(len(res)))  
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logger.info('mutation ......')  
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)
            if isinstance(self.diffusion, Diffusion):
                original_num_steps = self.diffusion.original_num_steps
            else:
                original_num_steps = self.diffusion.model.num_timesteps
            if self.args.timestep_quant_mode =='groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                time_list = self.time_list
            else:
                candidates = []
                for i in range(original_num_steps):
                    if i not in cand['timestep']:
                        candidates.append(i)

            for i in range(len(cand['timestep'])):
                if np.random.random_sample() < m_prob:
                    if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                        new_c = random.choice(time_list[i])
                    else:
                        new_c = random.choice(candidates)
                        new_index = candidates.index(new_c)
                        del(candidates[new_index])
                    cand['timestep'][i] = new_c
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent' and len(candidates) == 0:    
                        break
            if self.args.timestep_quant_mode =='consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
                cand_bit_list=copy.deepcopy(cand['use_bitwidth'][0])
                for j in range(len(cand_bit_list)):
                    if np.random.random_sample()< m_prob:
                        cand_bit_list[j]=random.randint(0,len(self.diffusion.weight_list)-1)
                cand['use_bitwidth']=[copy.deepcopy(cand_bit_list) for i in range(len(cand['timestep']))]
            else:
                for i in range(len(cand['use_bitwidth'])):
                    for j in range(len(cand['use_bitwidth'][i])):
                        if np.random.random_sample()< m_prob:
                            cand['use_bitwidth'][i][j]=random.randint(0,len(self.diffusion.weight_list)-1)
            if self.args.only_steps == True:
                cand['use_bitwidth'] = [[2 for i in m] for m in cand['use_bitwidth']]
            if self.args.only_search_weight:
                self.diffusion.model.model.diffusion_model.change_act(cand['use_bitwidth'])
            return cand

        while len(res) < mutation_num:  
              
            cand = random_func()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.info('mutation {}/{}'.format(len(res), mutation_num))  

        logger.info('mutation_num = {}'.format(len(res)))  
        return res

    def mutate_init_x(self, x0, mutation_num, m_prob):
        logger.info('mutation x0 ......')  
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)
            if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                time_list = self.time_list
            else:
                candidates = []
                if isinstance(self.diffusion, Diffusion):
                    for i in range(self.diffusion.original_num_steps):
                        if i not in cand['timestep']:
                            candidates.append(i)
                elif isinstance(self.diffusion, LatentDiffusion):
                    for i in range(self.diffusion.model.num_timesteps):
                        if i not in cand['timestep']:
                            candidates.append(i)

            for i in range(len(cand['timestep'])):
                if np.random.random_sample() < m_prob:
                    if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                        new_c = random.choice(time_list[i])
                    else:
                        new_c = random.choice(candidates)
                        new_index = candidates.index(new_c)
                        del(candidates[new_index])
                    cand['timestep'][i] = new_c
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent' and len(candidates) == 0:    
                        break
            if self.args.timestep_quant_mode == 'consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
                cand_bit_list=copy.deepcopy(cand['use_bitwidth'][0])
                for i in range(len(cand_bit_list)):
                    if np.random.random_sample()<m_prob:
                        cand_bit_list[i]=random.randint(0,len(self.diffusion.weight_list)-1)
                cand['use_bitwidth']=[copy.deepcopy(cand_bit_list) for i in range(len(cand['timestep']))]
            else:
                for i in range(len(cand['use_bitwidth'])):
                    for j in range(len(cand['use_bitwidth'][i])):
                        if np.random.random_sample()<m_prob:
                            cand['use_bitwidth'][i][j]=random.randint(0,len(self.diffusion.weight_list)-1)
            if self.args.only_steps == True:
                cand['use_bitwidth'] = [[2 for i in m] for m in cand['use_bitwidth']]
            if self.args.only_search_weight:
                self.diffusion.model.model.diffusion_model.change_act(cand['use_bitwidth'])
            return cand

        while len(res) < mutation_num:  
              
            cand = random_func()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logger.info('mutation x0 {}/{}'.format(len(res), mutation_num))  

        logger.info('mutation_num = {}'.format(len(res)))  
        return res

    def get_fid(self,timestep=None,bit_width=None,index=None):
        cand=dict()
        if isinstance(self.diffusion, Diffusion):
            num_timesteps = self.diffusion.num_timesteps
        elif isinstance(self.diffusion, LatentDiffusion):
            num_timesteps = self.diffusion.model.num_timesteps
        if timestep is None:
              
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), self.args.time_step
                )
                ** 2
            )
            cand['timestep']=[int(s) for s in list(seq)]
        else:
            cand['timestep']=timestep
        cand['use_bitwidth']=bit_width
          
        if isinstance(self.diffusion, Diffusion):
            if self.args.timestep_quant_mode =='no_constraint':
                bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(bit_width)
            else:
                bitops_now=self.diffusion.model.cal_bitops_cand_offload(bit_width)
        elif isinstance(self.diffusion, LatentDiffusion):
            bitops_now=self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(bit_width)
        logger.info('bitops:{}'.format(bitops_now))
        if isinstance(self.diffusion, Diffusion):
            nums_image = 50000
        else:
            nums_image = 50000   
        fid=self.get_cand_fid(cand,nums=nums_image,index=index)
        logger.info('timestep:{}, use_bitwidth:{}, fid:{}'.format(cand['timestep'],bit_width,fid))
    
    def get_fid_file(self,cand,timestep_num):
        timestep=cand['timestep']
        bit_width=cand['use_bitwidth']

        if isinstance(self.diffusion, Diffusion):
            num_timesteps = self.diffusion.num_timesteps
        elif isinstance(self.diffusion, LatentDiffusion):
            num_timesteps = self.diffusion.model.num_timesteps
        cand['use_bitwidth']=bit_width
        if timestep is None:
            if self.args.timestep_for_test_sample is 'quad':
                seq = (
                    np.linspace(
                        0, np.sqrt(num_timesteps * 0.8), timestep_num
                    )
                    ** 2
                )
            else:
                seq = np.arange(0, 1000, 1000/timestep_num)
            cand['timestep']=[int(s) for s in list(seq)]
            cand['use_bitwidth']=[[2 for i in range(self.layer_num)] for j in range(timestep_num)]
        else:
            cand['timestep']=timestep
          
        if timestep is None:
            nums_image=50000
        else:
            nums_image=50000
        self.get_cand_fid_file(cand,nums=nums_image,timestep_num=timestep_num)
          

    def search(self):
          
        logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(  
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        
        if self.args.use_ddim_init_x is False:  
            self.get_random_before_search(self.population_num)
        else:
            if self.args.cand_file is None:
                if isinstance(self.diffusion, Diffusion):
                    steps = self.diffusion.original_num_steps
                else:
                    steps = self.diffusion.model.num_timesteps
                if self.args.use_ddim:
                    timestep_respacing = 'ddim'
                else:
                    timestep_respacing = ''
                timestep_respacing += str(self.args.time_step)
                init_x = space_timesteps_one(steps, timestep_respacing)
                init_x=[i for i in init_x]
                init_x = sorted(init_x)
                if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                    init_x = [i[0] for i in self.time_list]
                  
                  
                init_cand=dict()
                init_cand['timestep']=init_x
                if self.args.set_7 == True:
                    init_cand['use_bitwidth']=[[1 for j in range(self.layer_num)] for i in range(len(init_x))]
                else:
                    init_cand['use_bitwidth']=[[2 for j in range(self.layer_num)] for i in range(len(init_x))]
                def get_each_bitops():
                    for i in range(len(self.diffusion.weight_list)):
                        use_bitwidth=[[i for j in range(self.layer_num)] for jj in range(len(init_x))]
                        logger.info('bitwidth {}:'.format(self.diffusion.weight_list[i]))
                          
                        if isinstance(self.diffusion, Diffusion):
                            if self.args.timestep_quant_mode == 'no_constraint':
                                self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
                            else:
                                self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
                        elif isinstance(self.diffusion, LatentDiffusion):
                            self.diffusion.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
                get_each_bitops()
                  
                self.is_legal_before_search(str(init_cand))
                  
                self.candidates.append(str(init_cand))
                  
                  
                  
                  
                  

                  
                  

                self.get_random_before_search(self.population_num//2)
                res = self.mutate_init_x(x0=str(init_cand), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
                self.candidates += res
            else:
                raise ValueError('not adjusted')
                f=open(self.args.cand_file,'r')
                a=f.readlines()
                f.close()
                a=eval(a[0])
                for cand in a:
                    cand=str(cand)
                    self.vis_dict[cand]={}
                    info=self.vis_dict[cand]
                    info['visited'] = True
                    cand=eval(cand)
                    use_bitwidth=cand['use_bitwidth']
                    if isinstance(self.diffusion, Diffusion):
                        info['bitops']=self.diffusion.model.cal_bitops(use_bitwidth)
                    else:
                        info['bitops']=self.diffusion.model.model.diffusion_model.cal_bitops(use_bitwidth)
                    info['fid']=self.get_cand_fid(args=self.args,cand=cand)
                    self.candidates.append(str(cand))
        while self.epoch < self.max_epochs:
            logger.info('epoch = {}'.format(self.epoch))  

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logger.info('epoch = {} : top {} result'.format(  
                self.epoch, len(self.keep_top_k[50])))
            txt_path=os.path.join(logdir,'best_cand.txt')
            with open(txt_path,'w') as f:
                for i, cand in enumerate(self.keep_top_k[50]):
                    logger.info('No.{} {} fid = {}'.format(  
                        i + 1, cand, self.vis_dict[cand]['fid']))
                    f.write('No.{} {} fid = {}'.format(  
                        i + 1, cand, self.vis_dict[cand]['fid']))
            
            self.last_best_cand = self.keep_top_k[50][0]
            

            if self.epoch + 1 == self.max_epochs:
                break    

              
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            self.candidates = mutation

            cross_cand = self.get_cross(self.select_num, self.crossover_num)
            self.candidates += cross_cand

            self.get_random(self.population_num)   

            self.epoch += 1
        with open(os.path.join(logdir,'vis_dict.txt'),'w') as f:
            f.write(str(self.vis_dict))
        for iiii in range(10):
            cand_best = self.keep_top_k[10][iiii]
            cand_best = eval(cand_best)
            self.get_fid(cand_best['timestep'],cand_best['use_bitwidth'],index=iiii)

    def get_all_performance(self):
        for i in range(len(self.diffusion.weight_list)):
            use_bitwidth=[[i for _ in range(self.diffusion.model.layer_num)]for __ in range(self.args.time_step)]
            self.get_fid(bit_width=use_bitwidth)
    
    def get_images(self):
          
          
          
        
          
          
          
        
          

          
          
          

          
          
          

          

          
        cand = eval(self.args.cand_for_performance)
        if cand['use_bitwidth'] != None:
            tmp_bitwidth=copy.deepcopy(cand['use_bitwidth'])
            cand['use_bitwidth'] = [copy.deepcopy(tmp_bitwidth) for i in range(self.args.time_step)]
        self.get_fid_file(cand=cand,timestep_num=self.args.time_step)
            


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()
      
      
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

      
    seed_everything(args.seed)

      
    logdir = os.path.join(args.logdir, "samples", args.out_path)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(75 * "=")
    logger.info(f"Host {os.uname()[1]}")
    logger.info("logging to:")
    imglogdir = os.path.join(logdir, "img")
    args.image_folder = imglogdir

    os.makedirs(imglogdir)
    logger.info(logdir)
    logger.info(75 * "=")
    if args.model_type == 'trivial':
        runner1 = Diffusion(args, config)
    elif args.model_type == 'ldm':
        runner1 = LatentDiffusion(args, config)
    elif args.model_type == 'guided_diffusion':
        raise ValueError('not implemented')
    elif args.model_type == 'stable_diffusion':
        raise ValueError('not implemented')
    else:
        raise ValueError(str(args.model_type)+' not implemented')
    searcher = EvolutionSearcher(args, diffusion=runner1, time_step=args.time_step)
      
      

    if args.test:
        searcher.get_all_performance()
    elif args.get_some_performance:
        searcher.get_images()
    elif args.cand_timestep != 'no':   
          
        cand_ind_bitwidth = [eval(args.cand_use_bitwidth)[0]]
          
          
          
        logger.info(str(cand_ind_bitwidth))
        l_bit_width = []
        runner1.model.set_quant_idx_cand(cand_ind_bitwidth[0])
        runner1.model.get_allocation(l_bit_width)
          
        cnt = 0
        weight_l = []
        act_l = []

        while(cnt<len(l_bit_width)):
            print(cnt)
            if l_bit_width[cnt] == -10:  
                tmp1=act_l.pop()
                tmp2=act_l.pop()
                weight_l.append(tmp2)
                act_l.append(tmp1)
            else:
                act_l.append(l_bit_width[cnt])
            cnt+=1
        
        weight_l = [runner1.weight_list[m] for m in weight_l]
        act_l = [runner1.weight_list[m] for m in act_l]
        
          
        logger.info(str(l_bit_width))
        '''
        cand_ind_bitwidth = eval(args.cand_use_bitwidth)[0]
        cand_timestep = eval(args.cand_timestep)
        cand_use_bitwidth = [copy.deepcopy(cand_ind_bitwidth) for mm in cand_timestep]
        searcher.get_fid(cand_timestep, cand_use_bitwidth)
        '''
    else:
        searcher.search()
    
