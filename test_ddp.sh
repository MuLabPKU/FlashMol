#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

# 98894 samples

bash train_ddp.sh 7\
    --exp_name dmd_drug_8step_b32 \
    --teacher_path outputs/drugs_latent2 \
    --train_diffusion \
    --nf 256 \
    --n_layers 4 \
    --latent_nf 2 \
    --normalize_factors '[1,4,10]' \
    --diffusion_noise_schedule polynomial_2 \
    --diffusion_noise_precision 1e-5 \
    --diffusion_loss_type l2 \
    --step_num 8 \
    --step_num_small 3 \
    --step_num_large 3 \
    --step_num_pow 1.0 \
    --step_ratio 5 \
    --n_epochs 36 \
    --batch_size 8 \
    --n_stability_samples 10000 \
    --test_epochs 5 \
    --ema_decay 0.9999 \
    --G_lr 8e-7 \
    --mu_fake_lr 32e-7 \
    --disc_lr 16e-5 \
    --gan_coeffg 0.000 \
    --gan_coefff 0.000 \
    --gan_pos -1 \
    --r1_weight 0.001 \
    --r1_sigma 0.01 \
    --Tmin 0.02 \
    --Tminpre 0.02 \
    --tmin_liftpos 6 \
    --reg_coeff 0.0 \
    --consist_coeff 0.00 \
    --log_grad_norm \
    --no_wandb
