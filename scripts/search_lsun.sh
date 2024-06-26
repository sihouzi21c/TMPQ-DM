export PYTHONPATH="./q_diffusion:qdiff:ldm:taming"

CUDA_VISIBLE_DEVICES=$1 python ../alg_on_latent.py \
--config q_diffusion/models/ldm/lsun_beds256/config.yaml \
--model_ckpt q_diffusion/models/ldm/lsun_beds256/model.ckpt \
--model_type 'ldm' \
--use_pretrained --get_some_performance \
--eta 0 \
--skip_type quad --resume --ptq --weight_bit '8' '7' '6' '5' --cali_ckpt 'ckpt/bedroom_split/cifar5678ckpt.pth' \
--quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 32 --quant_act --act_bit '8' '7' '6' '5' \
--a_sym --split --cali_data_path datasets/bedroom_sample2040_allst.pt --save_memory \
-l task/mixed --out_path bedroom_method \
--time_step 10 --num_images 1000 --offload_file 'offload/bedroom/dict.txt' \
--max_epochs 15 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True --timestep_quant_mode 'groupwise_consistent' --split_quantization --task 'lsun' \
--use_ddim True --cali_iters 20000 --cali_iters_a 5000 \
#qdiff_cifar_mixed_offload