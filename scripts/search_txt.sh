export PYTHONPATH="./q_diffusion:qdiff:ldm:taming"

CUDA_VISIBLE_DEVICES=$1 python alg_on_txt.py --prompt "a photograph of an astronaut riding a horse" \
--plms --resume --cond --ptq --weight_bit '8' '7' '6' '5' --quant_mode qdiff --quant_act --act_bit '8' '7' '6' '5' \
--cali_st 25 \
--cali_batch_size 8 --cali_n 128 --split --sm_abit 16 --cali_ckpt '/root/autodl-tmp/cifar5678ckpt.pth' \
--cali_data_path /root/autodl-fs/sd_coco-s75_sample1024_allst.pt --outdir task/txt --out_path 4steps_fix \
--config 'q_diffusion/configs/stable-diffusion/v1-inference_coco.yaml' --ckpt '/root/autodl-tmp/sd-v1-4.ckpt' \
--time_step 4 --num_images 1000 --split_quantization \
--max_epochs 20 --n_samples 3 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 --offload_file '/root/data_latest.txt' \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True --timestep_quant_mode 'groupwise_consistent' \
--use_ddim True --cali_iters 20000 --cali_iters_a 5000 --grad_accumulation --task 'txt' \