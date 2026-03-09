export CUDA_VISIBLE_DEVICES=4
python main_dmd.py \
  --exp_name dmd_qm9_trial_3 \
  --teacher_path outputs/qm9_latent2 \
  --resume outputs/dmd_qm9_trail_difflr_0 \
  --start_epoch 7 \
  --train_diffusion \
  --step_num 20 \
  --n_epochs 21 \
  --batch_size 64 \
  --n_stability_samples 5000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 256 \
  --n_layers 9 \
  --normalize_factors '[1,4,10]' \
  --test_epochs 5 \
  --ema_decay 0.9999 \
  --latent_nf 2 \
  --gan_coeffg 0.2 \
  --gan_coefff 1 \
  --gan_pos 7 \
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 2e-8 \
  --mu_fake_lr 8e-8 \
  --disc_lr 8e-8 \
  --tmin_liftpos 5 \
  --step_num_div_small 4 \
  --step_num_div_large 2 \
  --step_num_liftpos 10 \



export CUDA_VISIBLE_DEVICES=4
python main_dmd.py \
  --exp_name dmd_qm9_trial_3.1 \
  --teacher_path outputs/qm9_latent2 \
  --resume outputs/dmd_qm9_trail_difflr_0 \
  --start_epoch 7 \
  --train_diffusion \
  --step_num 20 \
  --n_epochs 21 \
  --batch_size 64 \
  --n_stability_samples 5000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 256 \
  --n_layers 9 \
  --normalize_factors '[1,4,10]' \
  --test_epochs 5 \
  --ema_decay 0.9999 \
  --latent_nf 2 \
  --gan_coeffg 0.2 \
  --gan_coefff 1 \
  --gan_pos 7\
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 2e-8 \
  --mu_fake_lr 8e-8 \
  --disc_lr 8e-5 \
  --tmin_liftpos 5 \
  --step_num_div_small 4 \
  --step_num_div_large 2 \
  --step_num_liftpos 10 \



export CUDA_VISIBLE_DEVICES=4
python main_dmd.py \
  --exp_name dmd_qm9_trial_3.1 \
  --teacher_path outputs/qm9_latent2 \
  --resume outputs/dmd_qm9_trail_difflr_0 \
  --start_epoch 7 \
  --train_diffusion \
  --step_num 20 \
  --n_epochs 21 \
  --batch_size 64 \
  --n_stability_samples 5000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 256 \
  --n_layers 9 \
  --normalize_factors '[1,4,10]' \
  --test_epochs 5 \
  --ema_decay 0.9999 \
  --latent_nf 2 \
  --gan_coeffg 0.02 \
  --gan_coefff 1 \
  --gan_pos 7 \
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 2e-8 \
  --mu_fake_lr 8e-8 \
  --disc_lr 8e-8 \
  --tmin_liftpos 5 \
  --step_num_div_small 4 \
  --step_num_div_large 2 \
  --step_num_liftpos 10 \