export CUDA_VISIBLE_DEVICES=1

#Window 4
python main_dmd.py \
  --exp_name dmd_qm9_trial_32step3.3 \
  --teacher_path outputs/qm9_latent2 \
  --train_diffusion \
  --step_num 32 \
  --n_epochs 26 \
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
  --gan_pos 0\
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 2e-8 \
  --mu_fake_lr 8e-8 \
  --disc_lr 8e-4 \
  --tmin_liftpos 0 \
  --step_num_div_small 4 \
  --step_num_div_large 2 \
  --step_num_liftpos 10000 \
  --Tmin 0.002

#Window 4
python main_dmd.py \
  --exp_name dmd_qm9_trial_32step3.4 \
  --teacher_path outputs/qm9_latent2 \
  --train_diffusion \
  --step_num 32 \
  --n_epochs 26 \
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
  --gan_pos 0\
  --step_ratio 5 \
  --reg_coeff 0.0 \
  --G_lr 2e-8 \
  --mu_fake_lr 8e-8 \
  --disc_lr 2e-4 \
  --tmin_liftpos 0 \
  --step_num_div_small 4 \
  --step_num_div_large 2 \
  --step_num_liftpos 10000 \
  --Tmin 0.002