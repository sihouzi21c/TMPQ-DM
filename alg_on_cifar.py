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
from qdiff.block_recon import block_reconstruction_one_respective,block_reconstruction_retrain

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

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

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
        section_counts = [int(x) for x in section_counts.split(",")]  # [10, 20]
    size_per = num_timesteps // len(section_counts) # 500  1000
    extra = num_timesteps % len(section_counts)  # 0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)  # 500
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)  # (500 - 1) / (10 - 1) = 55.4
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    a=set(all_steps)
    b=[i for i in a]
    #print(a)
    #print(b)
    return a
    #return list(set(all_steps))

class Diffusion(object):
    def __init__(self, args, config,device=None):#init for everything
        self.args = args
        self.config = config
        config.split_shortcut = self.args.split
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device
        self.weight_list=[int(i) for i in args.weight_bit]
        self.act_list=[int(i) for i in args.act_bit]
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )#[1000,]
        self.betas = torch.from_numpy(betas).float()
        self.betas = self.betas.to(self.device)
        betas = self.betas
        self.num_timesteps = betas.shape[0]
        self.original_num_steps=self.num_timesteps
        ##print(betas.shape)##[1000]
        ##exit()##
        self.alphas = 1.0 - betas
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), self.alphas_cumprod[:-1]], dim=0
        )
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = self.posterior_variance.clamp(min=1e-20).log()
        self.model=self.get_model(self.weight_list,self.act_list,'cifar5678',cali_ckpt=args.cali_ckpt)
        ##self.model_list=[self.model1,self.model1]
    
    def set_use_timestep(self,x):
        self.use_timesteps=x
    def forward_model(self,model,cali_xs,cali_ts):
        for i in range(len(self.weight_list)):
            model.set_quant_idx(i)
            _ = model(cali_xs, cali_ts)
    def get_model(self,weight_list,act_list,ckptsavename,cali_ckpt='no'):#return : model(qnn)
        model = Model(self.config)
        if cali_ckpt is not 'no':
            print('cali_ckpt:{}'.format(cali_ckpt))
            print('weight bit:{}'.format(weight_list))
            print('act_bit:{}'.format(act_list))
        if self.config.data.dataset == "CIFAR10":
            name = "cifar10"
        elif self.config.data.dataset == "LSUN":
            name = f"lsun_{self.config.data.category}"
        else:
            raise ValueError
        ckpt = get_ckpt_path(f"ema_{name}")
        logger.info("Loading checkpoint {}".format(ckpt))
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        model.to(self.device)
        model.eval()
        assert(self.args.cond == False)
        if self.args.quant_mode == 'qdiff':

            wq_params = [{'n_bits': weight_bit, 'channel_wise': True, 'scale_method': 'max'} for weight_bit in weight_list]
            aq_params = [{'n_bits': act_bit, 'symmetric': args.a_sym, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': args.quant_act} for act_bit in act_list]
            #if self.args.resume:
            #    logger.info('Load with min-max quick initialization')
            #    wq_params['scale_method'] = 'max'
            #    aq_params['scale_method'] = 'max'
            #if self.args.resume_w:
            #    wq_params['scale_method'] = 'max'
            if self.args.not_attn:
                qnn = QuantModel_one(
                    model=model, args=self.args,weight_quant_params_list=wq_params, act_quant_params_list=aq_params, 
                    sm_abit=self.args.sm_abit,config=self.config, not_attn = True)
            else:
                qnn = QuantModel_one(
                    model=model, args=self.args,weight_quant_params_list=wq_params, act_quant_params_list=aq_params, 
                    sm_abit=self.args.sm_abit,config=self.config, not_attn = False)
            #qnn.layer_num
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
            else:
                logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape}")
                cali_xs, cali_ts = cali_data
                if self.args.resume_w:
                    resume_cali_model_one(qnn, cali_ckpt, cali_data, len(self.weight_list), False, cond=False)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    self.forward_model(qnn,cali_xs[:8].cuda(),cali_ts[:8].cuda())
                    logger.info("Initializing has done!")
                kwargs = dict(cali_data=cali_data, batch_size=self.args.cali_batch_size, 
                            iters=self.args.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                            warmup=0.2, act_quant=False, opt_mode='mse')
           
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
                
                recon_model_func = recon_model_respective
                if not self.args.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model_func(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                if self.args.quant_act:
                    logger.info("UNet model")
                    logger.info(model)                    
                    logger.info("Doing activation calibration")   
                    # Initialize activation quantization parameters
                    qnn.set_quant_state(True, True)
                    with torch.no_grad():
                        inds = np.random.choice(cali_xs.shape[0], 32, replace=False)
                        #_ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda())
                        self.forward_model(qnn,cali_xs[inds].cuda(), cali_ts[inds].cuda())
                        if self.args.running_stat:##now not used, try using? improve not much
                            #print('wo cao yao zuo ma')#
                            logger.info('Running stat for activation quantization')
                            qnn.set_running_stat(True)
                            for i in range(int(cali_xs.size(0) / 32)):
                                _ = qnn(
                                    (cali_xs[i * 32:(i + 1) * 32].to(self.device), 
                                    cali_ts[i * 32:(i + 1) * 32].to(self.device)))
                            qnn.set_running_stat(False)
                    kwargs = dict(
                        cali_data=cali_data, iters=self.args.cali_iters_a, act_quant=True, 
                        opt_mode='mse', lr=self.args.cali_lr, p=self.args.cali_p)   
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
            qnn.layer_num = qnn.get_num_offload()
            model = qnn
        use_bitwidth=[]
        use_bitwidth.append([0 for i in range(model.layer_num)])
        list_conv, list_QKV, list_trans, list_conv_idx, list_QKV_idx, list_trans_idx = model.cal_bitops_offload(use_bitwidth)
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
        model.to(self.device)
        if self.args.verbose:
            logger.info("quantized model")
            logger.info(model)
        
        model.eval()
        return model
    
    def sample_image(self, use_bit_width, nums=None, batch_size=128): #To Do: use_bit_width is two dims
        config = self.config
        if nums is None:
            logger.info('sample num_images')
            total_n_samples = self.args.num_images
        else:
            logger.info('sample given number')
            total_n_samples = nums
        #n_rounds = math.ceil((total_n_samples - img_id) / batch_size)
        n_rounds = math.ceil((total_n_samples) / batch_size)
        all_images=[]
        ##all_labels=[]
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image_new(x, use_bit_width,self.model)
                #x = inverse_data_transform(config, x)#[-1,1] turn to [0,1]
                x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                x = x.permute(0, 2, 3, 1)
                x = x.contiguous()
                gathered_samples = x.unsqueeze(0)
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            arr = np.concatenate(all_images, axis=0)
            return arr          

    def sample_image_new(self,x,use_bit_width,model,last=True):#get a batch of image
        '''
        if self.args.skip_type == "uniform":
            skip = self.num_timesteps // self.args.time_step
            seq = range(0, self.num_timesteps, skip)
        '''

        from ddim.functions.denoising import generalized_steps_mix,generalized_steps

        betas = self.betas
        seq=[]
        for i in range(self.original_num_steps):
            if i in self.use_timesteps:
                seq.append(i)
        ##assert len(seq) == len(self.use_timesteps) #there maybe use_timesteps duplicate
        if self.args.origin:
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), len(self.use_timesteps)
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        if self.args.origin:
            print("origin")
            print(seq)
            xs=generalized_steps(x,seq,model,betas,eta=self.args.eta, args=self.args)##ddim
        else:
            xs = generalized_steps_mix(#eta=0 , same as p_sample_loop in guided_diffusion
                x, seq, use_bit_width,model, betas, eta=self.args.eta, args=self.args)
        x = xs
        if last:
            x = x[0][-1]
        return x

    def sample_image_file(self, use_bit_width, nums=None, batch_size=128,timesteps_num=None): #To Do: use_bit_width is two dims
        config = self.config
        os.makedirs(os.path.join(self.args.logdir,str(timesteps_num)))
        #img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        #logger.info(f"starting from image {img_id}")
        if nums is None:
            logger.info('sample num_images')
            total_n_samples = self.args.num_images
        else:
            logger.info('sample given number')
            total_n_samples = nums
        #n_rounds = math.ceil((total_n_samples - img_id) / batch_size)
        n_rounds = math.ceil((total_n_samples) / batch_size)
        all_images=[]
        ##all_labels=[]
        img_id=0
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
        with torch.no_grad():
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                with amp.autocast(enabled=False):
                    x = self.sample_image_new(x, use_bit_width,self.model)
                x = inverse_data_transform(config, x)#[-1,1] turn to [0,1]
                if img_id+x.shape[0]>total_n_samples:
                    n = self.args.max_images - img_id
                for i in range(n):
                    tvu.save_image(
                        x[i], os.path.join(self.args.logdir,str(timesteps_num) ,f"{img_id}.png")
                    )
                    img_id += 1


from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf

rescale = lambda x: (x + 1.) / 2.
   


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
    parser.add_argument('--num_images',type=int,default=2048)#2048
    parser.add_argument('--origin',action="store_true")
    parser.add_argument('--cand_timestep',type=str,default='no')
    parser.add_argument('--cand_use_bitwidth',type=str,default='no')
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    #parser.add_argument('--ckpt_list',type=str,default='["no","no"]')
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

    # qdiff specific configs
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
        "--cali_iters", type=int, default=80000, #20000
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=20000, type=int, #5000
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
    ##parser.add_argument('--ref_path',type=str,default='')
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
        #for offload
        with open(self.args.offload_file,'r') as f:
            self.selection_dict=eval(f.readlines()[0])
        self.selection_list = list(self.selection_dict.keys())
        self.selection_list = [eval(i) for i in self.selection_list]
        if self.args.timestep_quant_mode == 'groupwise' or self.args.timestep_quant_mode == 'groupwise_consistent':
            if self.args.group_method == 'quad':
                seq=(
                    np.linspace(
                        0, np.sqrt(1000 * 0.8), self.args.time_step # set threshold, N-->N+1 steps
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
        #self.bitnum=[8,6]
        # self.cfg = cfg
        ## EA hyperparameters
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        ## tracking variable 
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
        
        # self.RandomForestClassifier = RandomForestClassifier(n_estimators=40)
        # self.rf_features = []
        # self.rf_lebal = []

        self.x0 = args.init_x

        from evaluations.evaluator_v1 import Evaluator_v1
        import tensorflow.compat.v1 as tf
        config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
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
            #for offload：cal_bitops_cand-->cal_bitops_cand_offload
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
        logger.info('select ......')##
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def sample_active_subnet(self):#return timestep:,use_bitwidth:
        original_num_steps = self.diffusion.original_num_steps
        use_timestep = [i for i in range(original_num_steps)]
        if self.args.timestep_quant_mode =='consistent' or self.args.timestep_quant_mode == 'groupwise_consistent':
            use_bit_width_one = [random.randint(0,len(self.diffusion.weight_list)-1)for j in range(self.layer_num)]
            use_bitwidth = [copy.deepcopy(use_bit_width_one) for i in range(self.time_step)]#一起改变
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
            logger.info('cand has visited!')##
            return False
        info['visited'] = True
        cand=eval(cand)
        if self.flops_constrain == True:
            use_bitwidth=cand['use_bitwidth']
            #for offload
            if self.args.timestep_quant_mode =='no_constraint':
                bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
            else:
                bitops_now=self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')
                info['fid']=9999
                return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
        if self.flops_constrain == True:
            logger.info('cand_timestep: {}, fid: {}, bitops:{}'.format(cand['timestep'], info['fid'],info['bitops']))####
        else:
            logger.info('cand_timestep: {}, fid: {}'.format(cand['timestep'], info['fid']))####
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
            if self.args.timestep_quant_mode == 'no_constraint':
                bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
            else:
                bitops_now=self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
            info['bitops']=bitops_now
            if(bitops_now>self.max_bitops):
                logger.info('cand out of boundary')
                info['fid']=9999
                return False
        info['fid'] = self.get_cand_fid(args=self.args, cand=cand)
        if self.flops_constrain == True:
            logger.info('cand_timestep: {}, fid: {}, bitops:{}'.format(cand['timestep'], info['fid'], info['bitops']))####
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
            self.ref_stats=get_statistics(self.evaluator,'/home/Datasets/cifar10_png/cifar_train.npz')
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
        logger.info('random select ........')####
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))####
        logger.info('random_num = {}'.format(len(self.candidates)))####

    def get_random(self, num):
        logger.info('random select ........')####
        while len(self.candidates) < num:
            cand = self.sample_active_subnet()
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.info('random {}/{}'.format(len(self.candidates), num))####
        logger.info('random_num = {}'.format(len(self.candidates)))####

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logger.info('cross ......')####
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
            logger.info('cross {}/{}'.format(len(res), cross_num))####

        logger.info('cross_num = {}'.format(len(res)))####
        return res

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logger.info('mutation ......')####
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
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent' and len(candidates) == 0:  # cand 的长度小于 candidates 的长度
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
            logger.info('mutation {}/{}'.format(len(res), mutation_num))####

        logger.info('mutation_num = {}'.format(len(res)))####
        return res

    def mutate_init_x(self, x0, mutation_num, m_prob):
        logger.info('mutation x0 ......')####
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
                for i in range(self.diffusion.original_num_steps):
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
                    if self.args.timestep_quant_mode != 'groupwise' and self.args.timestep_quant_mode != 'groupwise_consistent' and len(candidates) == 0:  # cand 的长度小于 candidates 的长度
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
            logger.info('mutation x0 {}/{}'.format(len(res), mutation_num))####

        logger.info('mutation_num = {}'.format(len(res)))####
        return res

    def get_fid(self,timestep=None,bit_width=None,index=None):
        cand=dict()
        num_timesteps = self.diffusion.num_timesteps
        if timestep is None:
            #quad
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
        if self.args.timestep_quant_mode =='no_constraint':
            bitops_now=self.diffusion.model.cal_bitops_cand_offload_no_constraint(bit_width)
        else:
            bitops_now=self.diffusion.model.cal_bitops_cand_offload(bit_width)
        logger.info('bitops:{}'.format(bitops_now))
        if isinstance(self.diffusion, Diffusion):
            nums_image = 50000
        else:
            nums_image = 50000 # 10000
        fid=self.get_cand_fid(cand,nums=nums_image,index=index)
        logger.info('timestep:{}, use_bitwidth:{}, fid:{}'.format(cand['timestep'],bit_width,fid))
    
    def get_fid_file(self,cand,timestep_num):
        timestep=cand['timestep']
        bit_width=cand['use_bitwidth']
        num_timesteps = self.diffusion.num_timesteps
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
        logger.info('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(##
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        
        if self.args.use_ddim_init_x is False:#not in
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
                        if self.args.timestep_quant_mode == 'no_constraint':
                            self.diffusion.model.cal_bitops_cand_offload_no_constraint(use_bitwidth)
                        else:
                            self.diffusion.model.cal_bitops_cand_offload(use_bitwidth)
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
            logger.info('epoch = {}'.format(self.epoch))##

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['fid'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['fid'])

            logger.info('epoch = {} : top {} result'.format(##
                self.epoch, len(self.keep_top_k[50])))
            txt_path=os.path.join(logdir,'best_cand.txt')
            with open(txt_path,'w') as f:
                for i, cand in enumerate(self.keep_top_k[50]):
                    logger.info('No.{} {} fid = {}'.format(##
                        i + 1, cand, self.vis_dict[cand]['fid']))
                    f.write('No.{} {} fid = {}'.format(##
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
    
            


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
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
    elif args.model_type == 'guided_diffusion':
        raise ValueError('not implemented')
    elif args.model_type == 'stable_diffusion':
        raise ValueError('not implemented')
    else:
        raise ValueError(str(args.model_type)+' not implemented')
    searcher = EvolutionSearcher(args, diffusion=runner1, time_step=args.time_step)

    searcher.search()
