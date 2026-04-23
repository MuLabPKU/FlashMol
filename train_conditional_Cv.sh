#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

python main_dmd.py \
  --exp_name dmd_qm9_8step_11.0_conditional_Cv \
  --teacher_path outputs/exp_cond_Cv \
  --train_diffusion \
  --step_num 8 \
  --n_epochs 11 \
  --batch_size 32 \
  --n_stability_samples 10000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 192 \
  --n_layers 9 \
  --normalize_factors '[1,4,10]' \
  --test_epochs 5 \
  --ema_decay 0.9999 \
  --latent_nf 1 \
  --gan_coeffg 0.0 \
  --gan_coefff 0.2 \
  --gan_pos -10 \
  --r1_weight 0.001 \
  --r1_sigma 0.01 \
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 8e-7 \
  --mu_fake_lr 32e-7 \
  --disc_lr 16e-5 \
  --step_num_small 3 \
  --step_num_large 3 \
  --step_num_pow 1.0 \
  --use_js True \
  --fdiv_coeff 0.1 \
  --Tmin 0.02 \
  --Tminpre 0.02 \
  --tmin_liftpos 6 \
  --conditioning Cv \
  --dataset qm9_second_half \
  --include_charges False \
  --log_grad_norm
