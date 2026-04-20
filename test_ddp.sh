#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 98894 samples

bash train_ddp.sh 8\
    --exp_name dmd_drug_8step_b48_2e-4 \
    --teacher_path outputs/drugs_latent2 \
    --train_diffusion \
    --nf 256 \
    --n_layers 4 \
    --latent_nf 2 \
    --normalize_factors '[1,4,10]' \
    --normalization_factor 1 \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --step_num 8 \
    --step_num_small 0 \
    --step_num_large 0 \
    --step_num_pow 1.0 \
    --step_ratio 5 \
    --n_epochs 36 \
    --batch_size 6 \
    --n_stability_samples 10000 \
    --test_epochs 1 \
    --ema_decay 0.9999 \
    --G_lr 8e-7 \
    --mu_fake_lr 32e-7 \
    --disc_lr 16e-5 \
    --gan_coeffg 0.000 \
    --gan_coefff 0.2 \
    --gan_pos -1 \
    --r1_weight 0.001 \
    --r1_sigma 0.01 \
    --Tmin 0.02 \
    --Tminpre 0.02 \
    --tmin_liftpos 6 \
    --reg_coeff 0.0 \
    --use_js True \
    --fdiv_coeff 0.1 \
    --consist_coeff 0.00 \
    --log_grad_norm \
    --no_wandb
