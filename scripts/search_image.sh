export PYTHONPATH="./q_diffusion:qdiff:ldm:taming"

CUDA_VISIBLE_DEVICES=$1 python ../alg_on_image.py \
--config q_diffusion/models/ldm/cin256_v2/config.yaml \
--model_ckpt q_diffusion/models/ldm/cin256_v2/model.ckpt \
--model_type 'ldm' --ptq \
--use_pretrained \
--eta 0 \
--skip_type quad --weight_bit '8' '7' '6' '5' --ckpt_pre 'ckpt/cifar_mixed_prob/cifar5678ckpt.pth' \
--quant_mode qdiff --cali_st 20 --cali_batch_size 20 --cali_n 32 --quant_act --act_bit '8' '7' '6' '5' \
--a_sym --split --cali_data_path datasets/imagenet_input_100steps.pth \
-l task/mixed --out_path mixed_imagenet_split_get_txt --cond \
--time_step 10 --num_images 2000 \
--max_epochs 20 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True --timestep_quant_mode 'consistent' \
--use_ddim True --cali_iters 10000 --cali_iters_a 2500 \
#qdiff_cifar_mixed_image_offload.py