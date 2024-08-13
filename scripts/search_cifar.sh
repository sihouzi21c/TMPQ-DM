export PYTHONPATH="./q_diffusion:qdiff:ldm:taming"

CUDA_VISIBLE_DEVICES=$1 python ../alg_on_cifar.py \
--config q_diffusion/configs/cifar10.yml \
--use_pretrained \
--eta 0 \
--skip_type quad --resume --ptq --weight_bit '8' '7' '6' '5' --ckpt_pre 'ckpt/cifar_mixed_respective/cifar5678ckpt.pth' \
--quant_mode qdiff --cali_st 20 --cali_batch_size 32 --cali_n 32 --quant_act --act_bit '8' '7' '6' '5' \
--a_sym --split --cali_data_path datasets/cifar_sd1236_sample2048_allst.pt \
-l task/mixed --out_path 10split_quantization_based_on_respective_offload_groupwise_consistent_3_3_new \
--time_step 10 --num_images 2000 \
--max_epochs 20 \
--population_num 50 \
--mutation_num 25 \
--crossover_num 10 \
--seed 0 \
--m_prob 0.25 \
--use_ddim_init_x True --timestep_quant_mode 'groupwise_consistent' --task 'cifar' \
--use_ddim True --cali_iters 20000 --cali_iters_a 5000 --split_quantization --offload_file 'offload/cifar/0.txt' \
#qdiff_cifar_mixed_offload.py