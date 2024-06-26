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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda import amp
from imwatermark import WatermarkEncoder
import pdb
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
from sklearn.ensemble import RandomForestClassifier
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.quant_model import QuantModel_one
from qdiff.quant_block import BaseQuantBlock_one,QuantBasicTransformerBlock_one
from qdiff.quant_layer import QuantModule_one
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples,resume_cali_model_one
from qdiff.layer_recon import layer_reconstruction_one, layer_reconstruction_one_respective
from qdiff.block_recon import block_reconstruction_one, block_reconstruction_one_respective
from ldm.data.build_dataloader import build_dataloader
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
from pytorch_fid.inception import InceptionV3
from torch.utils import data
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, tuple) else choice(tuple(x))

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img    


from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

from transformers import AutoFeatureExtractor
from contextlib import nullcontext
from torch import autocast
from torchvision.utils import make_grid 
from einops import rearrange
from itertools import islice
from tqdm import tqdm, trange
from PIL import Image

rescale = lambda x: (x + 1.) / 2.








def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model




def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def get_activations(data, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):

    model.eval()

    if batch_size > data.shape[0]:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = data.shape[0]

    pred_arr = np.empty((data.shape[0], dims))
    start_idx = 0

    for i in range(0, data.shape[0], batch_size):
        if i + batch_size > data.shape[0]:
            batch = data[i:, :, :, :]
        else:
            batch = data[i:i+batch_size, :, :, :]
        batch = batch.to(device)
        

        with torch.no_grad():
            pred = model(batch)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]
    
    return pred_arr


def calculate_activation_statistics(datas, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    act = get_activations(datas, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(data1, ref_mu, ref_sigma, batch_size, device, dims, num_workers=1):
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    
    model = InceptionV3([block_idx]).cuda()

    m1, s1 = calculate_activation_statistics(data1, model, batch_size,
                                            dims, device, num_workers)
    
    fid_value = calculate_frechet_distance(m1, s1, ref_mu, ref_sigma)

    return fid_value



class LatentDiffusion(object):
    def __init__(self, opt):
        self.opt = opt
        self.config = OmegaConf.load(f"{opt.config}")

        self.original_num_steps = 1000

        self.device = 'cuda:0'
        self.weight_list=[int(i) for i in opt.weight_bit]
        self.act_list=[int(i) for i in opt.act_bit]
        self.sampler = self.get_model(self.weight_list,self.act_list,'cifar5678',cali_ckpt=opt.cali_ckpt)
        self.wm = "StableDiffusionV1"
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', self.wm.encode('utf-8'))


    def forward_model(self,model,cali_xs,cali_ts,cali_cs):
        for i in range(len(self.weight_list)):
            model.set_quant_idx(i)
            _ = model(cali_xs, cali_ts,cali_cs)
    def get_model(self,weight_list,act_list,ckptsavename,cali_ckpt='no'):
        gpu = True
        eval_mode = True
        self.config = OmegaConf.load(f"{opt.config}")
        self.config['dataloader']['batch_size']=self.opt.n_samples
        
        self.model = load_model_from_config(self.config, f"{opt.ckpt}")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model = self.model.cuda()
        assert(self.opt.cond == True)
        if opt.plms:
            self.sampler = PLMSSampler(self.model)
        elif opt.dpm_solver:
            self.sampler = DPMSolverSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)
        if self.opt.ptq:
            if self.opt.split:
                setattr(self.sampler.model.model.diffusion_model, "split", True)
            if self.opt.quant_mode == 'qdiff':
                a_scale_method = 'mse' if not self.opt.a_min_max else 'max'
                wq_params = [{'n_bits': weight_bit, 'channel_wise': True, 'scale_method': 'max'} for weight_bit in self.weight_list]
                aq_params = [{'n_bits': act_bit, 'symmetric': self.opt.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': self.opt.quant_act} for act_bit in self.act_list]
                if self.opt.resume:
                    logger.info('Load with min-max quick initialization')
                    for i in range(len(wq_params)):
                        wq_params[i]['scale_method'] = 'max'
                        aq_params[i]['scale_method'] = 'max'
                if self.opt.resume_w:
                    for i in range(len(wq_params)):
                        wq_params[i]['scale_method'] = 'max'
                
                
                
                qnn = QuantModel_one(
                    model = self.sampler.model.model.diffusion_model, args=self.opt,weight_quant_params_list=wq_params, act_quant_params_list=aq_params, 
                    sm_abit=self.opt.sm_abit,config=self.config)
                
                logger.info(qnn)
                
                qnn.cuda()
                qnn.eval()
                if self.opt.no_grad_ckpt:
                    logger.info('Not use gradient checkpointing for transformer blocks')
                    qnn.set_grad_ckpt(False)
                if self.opt.resume:
                    cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
                    resume_cali_model_one(qnn, self.opt.cali_ckpt, cali_data, len(self.weight_list), self.opt.quant_act, "qdiff", cond=self.opt.cond)
                else:
                    logger.info(f"Sampling data from {self.opt.cali_st} timesteps for calibration")
                    sample_data = torch.load(self.opt.cali_data_path)
                    cali_data = get_train_samples(self.opt, sample_data, self.opt.ddim_steps)
                    del(sample_data)
                    gc.collect()
                    logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")

                    cali_xs, cali_ts, cali_cs = cali_data

                    if self.opt.resume_w:
                        resume_cali_model_one(qnn, self.opt.cali_ckpt, cali_data, len(self.weight_list), False, cond=self.opt.cond)
                    else:
                        logger.info("Initializing weight quantization parameters")
                        qnn.set_quant_state(True, False) 
                        self.forward_model(qnn,cali_xs[:2].cuda(),cali_ts[:2].cuda(),cali_cs[:2].cuda())
                        
                        logger.info("Initializing has done!")
                        
                        
                    kwargs = dict(cali_data=cali_data, batch_size=self.opt.cali_batch_size, 
                        iters=self.opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                        warmup=0.2, act_quant=False, opt_mode='mse', cond = self.opt.cond, grad_accumulation = self.opt.grad_accumulation)
                    
                    def recon_model(model):
                        """
                        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                        """

                        for name, module in model.named_children():
                            logger.info(f"{name} {isinstance(module, BaseQuantBlock_one)}")
                            if name == 'output_block':
                                logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                                in_recon_done = True
                                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                            if name.isdigit() and int(name) >= 9:
                                logger.info(f"Saving temporary checkpoint at {name}...")
                                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
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
                            if name == 'output_block':
                                logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                                in_recon_done = True
                                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                            if name.isdigit() and int(name) >= 9:
                                logger.info(f"Saving temporary checkpoint at {name}...")
                                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
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
                    if opt.assigning_prob == True:
                        recon_model_func = recon_model
                    else:
                        recon_model_func = recon_model_respective
                    if not self.opt.resume_w:
                        logger.info("Doing weight calibration")
                        
                        qnn.set_quant_state(weight_quant=True, act_quant=False)
                    if self.opt.quant_act:
                        logger.info("UNet model")
                        logger.info(self.model.model)                    
                        logger.info("Doing activation calibration")   
                        
                        qnn.set_quant_state(True, True)
                        with torch.no_grad():
                            
                            
                            batch_tmp = 2
                            inds = np.random.choice(cali_xs.shape[0], batch_tmp, replace=False)
                            
                            self.forward_model(qnn,cali_xs[inds].cuda(), cali_ts[inds].cuda(),cali_cs[inds].cuda())
                            if self.opt.running_stat:
                                logger.info('Running stat for activation quantization')
                                qnn.set_running_stat(True)
                                for i in range(int(cali_xs.size(0) / batch_tmp)):
                                    self.forward_model(qnn,
                                        cali_xs[i * batch_tmp:(i + 1) * batch_tmp].cuda(),
                                        cali_ts[i * batch_tmp:(i + 1) * batch_tmp].cuda(),
                                        cali_cs[i * batch_tmp:(i + 1) * batch_tmp].cuda()
                                    )
                                qnn.set_running_stat(False)
                        
                        kwargs = dict(
                            cali_data=cali_data, batch_size=self.opt.cali_batch_size, iters=self.opt.cali_iters_a, act_quant=True, 
                            opt_mode='mse', lr=self.opt.cali_lr, p=self.opt.cali_p, cond=self.opt.cond, grad_accumulation = self.opt.grad_accumulation)   
                        
                        qnn.set_quant_state(weight_quant=True, act_quant=True)   
                    logger.info("Saving calibrated quantized UNet model")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                    
                        
                            
                    
                    
                if self.opt.split_quantization:
                    qnn.set_split_quantization(True)
                    qnn.layer_num = qnn.get_num_offload()
                    
                    qnn.cuda()
                    qnn.eval()
                    self.layer_num = qnn.layer_num
                use_bitwidth=[]
                use_bitwidth.append([2 for i in range(qnn.layer_num)])
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
                with open(os.path.join(opt.logdir,'data.txt'),'w') as f:
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
                
                
                
                
                
                
                
                
                

                
                
                
                
                self.sampler.model.model.diffusion_model = qnn
        
        self.dataloader_info = build_dataloader(self.config, opt)
        self.batch_size = opt.n_samples
        if opt.dpm_solver:
            tmp_sampler = DPMSolverSampler(self.model)
            from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
            ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
            dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
            skip_type = "time_uniform"
            t_0 = 1. / dpm_solver.noise_schedule.total_N  
            t_T = dpm_solver.noise_schedule.T  
            full_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=1000, device='cpu')
            full_timesteps=torch.flip(full_timesteps,[0])
            init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=opt.time_step, device='cpu')
            self.dpm_params = dict()
            full_timesteps = list(full_timesteps)
            self.dpm_params['full_timesteps'] = [full_timesteps[i].item() for i in range(len(full_timesteps))]
            init_timesteps = list(init_timesteps)
            self.dpm_params['init_timesteps'] = [init_timesteps[i].item() for i in range(len(init_timesteps))]
        else:
            self.dpm_params = None

        
        
        
        

        
        
        
        
        
        
        
        
        
        
        

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(outpath)) - 1
        sampling_file = os.path.join(outpath, "sampling_config.yaml")
        sampling_conf = vars(opt)
        with open(sampling_file, 'a+') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        if opt.verbose:
            logger.info("UNet model")
            logger.info(model.model)

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

        self.precision_scope = autocast if opt.precision=="autocast" else nullcontext
        return self.sampler

    
    def sample_image(self, cand,model=None, logdir=None, batch_size=None, vanilla=False, custom_steps=None, eta=None, 
        nums=50000, nplog=None, dpm=False):
        if model is None:
            model=self.sampler.model
        if batch_size==None:
            batch_size=self.opt.n_samples
        timestep_to_use=cand['timestep']
        if logdir is None:
            logdir=self.opt.logdir
        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=self.device)
        precision_scope = autocast if self.opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    
                    
                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        prompts = batch['text']
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = self.sampler.sample_timesteps(S=timestep_to_use,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        for x_sample in x_checked_image_torch:
                            all_samples.append(x_sample.cpu().numpy())
                        if len(all_samples) > nums:
                            logging.info('samples: ' + str(len(all_samples)))
                            break
        all_samples = np.array(all_samples)
        all_samples = torch.Tensor(all_samples)
        return all_samples
                        
                        
                        
                        
                        
                        
                        

                        
                        

                    
                    
                    
                    
                    

                    
                    
                    
                    
                    
                    
    def sample_image_file(self, cand,model=None, logdir=None, batch_size=None, vanilla=False, custom_steps=None, eta=None, 
        nums=50000, nplog=None, dpm=False):
        if model is None:
            model=self.sampler.model
        if batch_size==None:
            batch_size=self.opt.n_samples
        timestep_to_use=cand['timestep']
        if logdir is None:
            logdir=self.opt.logdir
        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=self.device)
        precision_scope = autocast if self.opt.precision=="autocast" else nullcontext
        cnt=0
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    
                    
                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        prompts = batch['text']
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = self.sampler.sample_timesteps(S=timestep_to_use,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        sample_path=os.path.join(self.opt.logdir,'img')
                        for x__ in x_checked_image_torch:
                            x__=255. * rearrange(x__.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x__.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{cnt:05}.png"))
                            cnt=cnt+1
                        if cnt > nums:
                            break
        all_samples = np.array(all_samples)
        all_samples = torch.Tensor(all_samples)
        return all_samples
    

    
    
    
    
    
    
    


    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--get_some_performance',action="store_true")
    parser.add_argument('--deterministic_retrain',action='store_true')
    parser.add_argument('--cand_for_performance',type=str,default=None)
    parser.add_argument('--vis_dict',type=str,default=None)
    parser.add_argument('--offload_file',type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--grad_accumulation', action = 'store_true')
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="task/txt"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
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
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )

    
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
        choices=["qdiff","linear","squant"], 
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
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
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

    def __init__(self, args,diffusion, time_step,dpm_params):
        
        self.epsilon=1e-3
        self.args = args
        with open('/root/tmp/data_latest.txt','r') as f:
            self.selection_dict=eval(f.readlines()[0])
        self.selection_list = list(self.selection_dict.keys())
        self.selection_list = [eval(i) for i in self.selection_list]
        if self.args.vis_dict != None:
            with open(args.vis_dict,'r') as f:
                a_a=eval(f.readlines()[0])
            a_a=list(a_a.keys())
            a_a=[eval(i)['use_bitwidth'][0] for i in a_a]
            self.selection_list=[x for x in self.selection_list if x not in a_a]
        if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
            if self.args.group_method == 'quad':
                seq = (
                    np.linspace(
                        0, np.sqrt(1000), self.args.time_step+2
                    )
                    ** 2
                )
            elif self.args.group_method == 'uniform':
                raise ValueError('not implemented')
                seq = range(0,1000,1000/(self.args.time_step+1))
            else:
                raise ValueError('not implemented! check for group_method pls')
            seq = [int(s) for s in list(seq)[1:-1]]
            self.time_list=[]
            for i in range(len(seq)):
                if i==0:
                    self.time_list.append(list(range(0,seq[0])))
                else:
                    self.time_list.append(list(range(seq[i-1],seq[i])))
            self.time_list.append(list(range(seq[-1],1000)))
            self.seq=seq
        self.diffusion=diffusion
        self.time_step = time_step
        self.flops_constrain=True
        self.ddim_discretize='quad'
        
        
        
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
            self.layer_num=self.diffusion.layer_num
            logger.info('layer_num:{}'.format(self.layer_num))
        else:
            self.layer_num=self.diffusion.layer_num=30
            self.diffusion.num_timesteps=1000
        self.max_fid = args.max_fid
        self.thres = args.thres

        self.x0 = args.init_x
        self.dpm_params = dpm_params

        if 6 in self.diffusion.weight_list and self.args.ptq:
            index_6=self.diffusion.weight_list.index(6)
            cand_now=[[index_6 for j in range(self.layer_num)]for i in range(self.time_step)]
            self.max_bitops_down=self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(cand_now)
            self.max_bitops=self.max_bitops_down
        self.last_best_cand = None
        self.ref_mu=np.load('/root/autodl-tmp/coco2014_mu.npy')
        self.ref_sigma=np.load('/root/autodl-tmp/coco2014_sigma.npy')

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
        if self.args.timestep_quant_mode == 'no_constraint' or self.args.timestep_quant_mode == 'consistent':
            raise ValueError('not implemented')
            use_timestep = [i for i in range(original_num_steps)]
            random.shuffle(use_timestep)
            use_timestep = use_timestep[:self.time_step]
        elif self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent': 
            time_list=self.time_list
            use_timestep=[copy.deepcopy(self.dpm_params['full_timesteps'][random.choice(i)]) for i in time_list]
        if self.args.timestep_quant_mode != 'no_constraint':
            idx_in_selection_list=0
            if len(self.selection_list) == 0:
                with open('/root/tmp/data_latest.txt','r') as f:
                    self.selection_dict=eval(f.readlines()[0])
                self.selection_list = list(self.selection_dict.keys())
                self.selection_list = [eval(i) for i in self.selection_list]
            
            select_cand=self.selection_list.pop(idx_in_selection_list)
            use_bitwidth=[copy.deepcopy(select_cand) for i in range(self.time_step+1)]
        else:
            raise ValueError('not implemented')
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
        
        if self.flops_constrain == True and self.args.ptq:
            use_bitwidth=cand['use_bitwidth']
            bitops_now=self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')
                info['fid']=9999
                return False
        
        
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
        
        if self.flops_constrain == True and self.args.ptq:
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
        if self.flops_constrain and self.args.ptq:
            use_bitwidth=cand['use_bitwidth']
            bitops_now=self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')
                info['fid']=9999
                return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
        
        if self.flops_constrain == True and self.args.ptq:
            logger.info('cand_timestep: {}, fid: {}, bitops:{}'.format(cand['timestep'], info['fid'], info['bitops']))
        else:
            logger.info('cand_timestep: {}, fid: {}'.format(cand['timestep'], info['fid']))
        
        
        

        return True

    def get_cand_fid(self, cand=None, args=None, nums=None):
        
        t1 = time.time()
        if nums==None:
            nums=self.args.num_images
        if self.args.ptq:
            self.diffusion.sampler.model.model.diffusion_model.set_quant_idx_cand(cand['use_bitwidth'][0])
        
        
        
        
        
        
        
        
        
        
        
        
        

        arr=self.diffusion.sample_image(cand, nums = nums)
        sample_time = time.time() - t1
        t1 = time.time()
        
        
        
        
        
        
        
        
        
        
        

        fid = calculate_fid(data1=arr,ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=320, dims=2048, device='cuda')
        logger.info('FID: '+str(fid))


        fid_time = time.time() - t1
        logger.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid

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
        max_iters = cross_num * 1000

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

            
            if self.args.timestep_quant_mode =='groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                
                time_list = self.time_list
            
            
            
            
            

            for i in range(len(cand['timestep'])):
                if np.random.random_sample() < m_prob:
                    if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                        
                        new_c = copy.deepcopy(self.dpm_params['full_timesteps'][random.choice(time_list[i])])
                    else:
                        raise ValueError('not implemented')
                        new_c = random.choice(candidates)
                        new_index = candidates.index(new_c)
                        del(candidates[new_index])
                    cand['timestep'][i] = new_c
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent':
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
        max_iters = mutation_num * 10000

        def random_func():
            cand = x0
            cand = eval(cand)
            if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                
                time_list=self.time_list
            
            
            
            
            


            for i in range(len(cand['timestep'])):
                if np.random.random_sample() < m_prob:
                    if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                        
                        new_c = copy.deepcopy(self.dpm_params['full_timesteps'][random.choice(time_list[i])])
                    else:
                        raise ValueError('not implemented')
                        new_c = random.choice(candidates)
                        new_index = candidates.index(new_c)
                        del(candidates[new_index])
                    cand['timestep'][i] = new_c
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent':
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

    def get_fid(self,timestep=None,bit_width=None):
        cand=dict()
        if timestep is None:
            
            seq = (
                np.linspace(
                    0, 1, self.args.time_step
                )
                ** 2
            )
            cand['timestep']=[int(s) for s in list(seq)]
        else:
            cand['timestep']=timestep
        cand['use_bitwidth']=bit_width
        if self.args.ptq:
            bitops_now=self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(bit_width)
            logger.info('bitops:{}'.format(bitops_now))
        fid=self.get_cand_fid(cand,nums=50000)
        logger.info('timestep:{}, use_bitwidth:{}, fid:{}'.format(cand['timestep'],bit_width,fid))
    
    def get_fid_file(self,cand,timestep_num):
        timestep=cand['timestep']
        if timestep is None:
            
            seq = (
                np.linspace(
                    0, np.sqrt(800), self.args.time_step+1
                )
                ** 2
            )
            seq=[int(__) for __ in seq]
            cand['timestep']=[self.dpm_params['full_timesteps'][__] for __ in seq]
            cand['use_bitwidth']=[[2 for i in range(self.layer_num)] for j in range(timestep_num)]
        
        
        
        
        
        
        
        
        
        
        
        
        if self.args.ptq:
            self.diffusion.model.model.diffusion_model.set_quant_idx_cand(cand['use_bitwidth'][0])
        fid=self.diffusion.sample_image_file(cand,nums=30000)
        
        
    def search(self):
        
        logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        if self.args.use_ddim_init_x is False:
            self.get_random_before_search(self.population_num)
        else:
            if self.args.cand_file is None:
                
                if self.args.use_ddim:
                    timestep_respacing = 'ddim'
                else:
                    timestep_respacing = ''
                timestep_respacing += str(self.args.time_step)
                
                
                                                        
                if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
                    
                    init_x=[]
                    for i in range(len(self.time_list)):
                        init_x.append(self.dpm_params['full_timesteps'][self.time_list[i][0]])
                
                
                
                init_cand=dict()
                init_cand['timestep']=list(init_x)
                init_cand['use_bitwidth']=[[2 for j in range(self.layer_num)] for i in range(len(init_x))]
                def get_each_bitops():
                    for i in range(len(self.diffusion.weight_list)):
                        use_bitwidth=[[i for j in range(self.layer_num)] for jj in range(len(init_x))]
                        logger.info('bitwidth {}:'.format(self.diffusion.weight_list[i]))
                        self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
                if self.args.ptq:
                    get_each_bitops()
                
                self.is_legal_before_search(str(init_cand))
                
                self.candidates.append(str(init_cand))
                
                
                
                
                
                self.get_random_before_search(self.population_num//2)
                res = self.mutate_init_x(x0=str(init_cand), mutation_num=self.population_num - self.population_num // 2 - 1, m_prob=0.1)
                self.candidates += res
            else:
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
                    info['bitops']=self.diffusion.sampler.model.model.diffusion_model.cal_bitops_cand_offload(use_bitwidth)
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
            txt_path=os.path.join(self.args.logdir,'best_cand.txt')
            
            with open(txt_path,'w') as f:
                for i, cand in enumerate(self.keep_top_k[50]):
                    logger.info('No.{} {} fid = {}'.format(
                        i + 1, cand, self.vis_dict[cand]['fid']))
                    f.write('No.{} {} fid = {}'.format(
                        i + 1, cand, self.vis_dict[cand]['fid']))
            with open(os.path.join(self.args.logdir,'vis_dict.txt'),'w') as f:
                f.write(str(self.vis_dict))
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
        for iiii in range(10):
            cand_best = self.keep_top_k[10][iiii]
            cand_best = eval(cand_best)
            self.get_fid(cand_best['timestep'],cand_best['use_bitwidth'])

    def continue_search(self):
        
        logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        steps = self.diffusion.model.num_timesteps
        with open(self.args.vis_dict,'r') as f:
            vis_dict=eval(f.readlines()[0])
        self.vis_dict=copy.deepcopy(vis_dict)
        for i in vis_dict.keys():
            vis_dict[i]=vis_dict[i]['fid']
        from operator import itemgetter
        vis_dict = dict(sorted(vis_dict.items(), key=itemgetter(1)))
        cands=list(vis_dict.keys())
        self.keep_top_k[10]=cands[:10]
        self.keep_top_k[50]=cands[:50]
        
        
        
        
        
        
        while self.epoch < self.max_epochs:
            logger.info('epoch = {}'.format(self.epoch))
            if self.epoch!=0:
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
        for i in range(2,len(self.diffusion.weight_list)):
            use_bitwidth=[[i for _ in range(self.diffusion.layer_num)]for __ in range(self.args.time_step)]
            self.get_fid(bit_width=use_bitwidth)
    
    def get_some_image(self):
        cand_4={'timestep':None,'use_bitwidth':None}
        
        
        t1=time.time()
        self.get_fid_file(cand=cand_4, timestep_num=4)
        print(time.time()-t1)
        
        
    def get_images(self):
        cand = eval(self.args.cand_for_performance)
        if cand['use_bitwidth'] != None:
            tmp_bitwidth=copy.deepcopy(cand['use_bitwidth'])
            cand['use_bitwidth'] = [copy.deepcopy(tmp_bitwidth) for i in range(self.args.time_step)]
        self.get_fid_file(cand=cand,timestep_num=self.args.time_step)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    n_gpu = torch.cuda.device_count()
    device = torch.device('cuda:0' if n_gpu > 0 else 'cpu')
    device_ids = list(range(n_gpu))
    parser = get_parser()
    opt = parser.parse_args()
    
    
    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "task/txt/txt2img-samples-laion400m"
    os.makedirs(opt.outdir, exist_ok = True)
    seed_everything(opt.seed)
    outpath = os.path.join(opt.outdir, opt.out_path)
    os.makedirs(outpath)
    opt.logdir = outpath
    log_path = os.path.join(outpath, "run.log")
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
    imglogdir = os.path.join(outpath, "img")
    opt.image_folder = imglogdir

    os.makedirs(imglogdir)
    logger.info(outpath)
    logger.info(75 * "=")
    
    assert(opt.cond)

    runner1 = LatentDiffusion(opt)

    searcher = EvolutionSearcher(opt, diffusion=runner1, time_step=opt.time_step,dpm_params=runner1.dpm_params)
    
    

    if opt.get_some_performance:
        
        searcher.get_images()
    elif opt.cand_timestep != 'no': 
        
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
    