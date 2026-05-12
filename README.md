# FlashMol

<p align="center">
  <strong>High-quality molecule generation in as few as four steps.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://arxiv.org/abs/2605.07020"><img src="https://img.shields.io/badge/arXiv-2605.07020-b31b1b.svg" alt="FlashMol arXiv"></a>
  <a href="https://github.com/Sylvan-Wesley/DMDMolGen"><img src="https://img.shields.io/badge/Code-FlashMol-blue.svg" alt="Code"></a>
</p>

<div align="center">
  <object data="equivariant_diffusion/training_diagram.pdf" type="application/pdf" width="100%" height="600">
    <a href="equivariant_diffusion/training_diagram.pdf"><strong>View training diagram PDF</strong></a>
  </object>
</div>

Official code release for the paper "[FlashMol: High-Quality Molecule Generation in as Few as Four Steps](https://arxiv.org/abs/2605.07020)".

FlashMol applies distribution matching distillation to latent diffusion models for fast 3D molecule generation. The repository includes training recipes for unconditional QM9 and GEOM-DRUGS generation, plus conditional QM9 generation using molecular property conditioning.

## Environment

Install the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

Note: If you want to set up an RDKit environment separately, it may be easiest to install conda and run:

```bash
conda create -c conda-forge -n my-rdkit-env rdkit
```

Then activate the environment and install the remaining packages from `requirements.txt`.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE) for the full terms.

## Train FlashMol

### For QM9

Train an 8-step DMD model using a pretrained QM9 teacher checkpoint at `outputs/qm9_latent2`.

```bash
python main_dmd.py \
  --exp_name dmd_qm9_8step_js_1e-1 \
  --teacher_path outputs/qm9_latent2 \
  --train_diffusion \
  --step_num 8 \
  --n_epochs 21 \
  --batch_size 32 \
  --n_stability_samples 10000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 256 \
  --n_layers 9 \
  --normalize_factors '[1,4,10]' \
  --test_epochs 5 \
  --ema_decay 0.9999 \
  --latent_nf 2 \
  --gan_coeffg 0.000 \
  --gan_coefff 0.2 \
  --gan_pos 0 \
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
  --consist_coeff 0.00 \
  --use_js True \
  --fdiv_coeff 0.1 \
  --Tmin 0.02 \
  --Tminpre 0.02 \
  --tmin_liftpos 6 \
  --log_grad_norm
```

### For GEOM-DRUGS

First follow the instructions in [`data/geom/README.md`](data/geom/README.md) to set up the GEOM-DRUGS data.

```bash
bash train_ddp.sh 8 \
  --exp_name dmd_drug_8step_node5_run3_13000 \
  --teacher_path outputs/drugs_latent2 \
  --train_diffusion \
  --short True \
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
  --n_epochs 1 \
  --batch_size 6 \
  --n_stability_samples 10000 \
  --test_epochs 1 \
  --ema_decay 0.9999 \
  --G_lr 5e-6 \
  --mu_fake_lr 4e-6 \
  --disc_lr 2e-4 \
  --gan_coeffg 0.000 \
  --gan_coefff 0.47 \
  --gan_pos -1 \
  --r1_weight 0.001 \
  --r1_sigma 0.01 \
  --Tmin 0.02 \
  --Tminpre 0.02 \
  --tmin_liftpos 6 \
  --reg_coeff 0.0 \
  --use_js True \
  --fdiv_coeff 0.03 \
  --consist_coeff 0.00 \
  --log_grad_norm \
  --no_wandb
```

## Evaluate FlashMol

Analyze sample quality:

```bash
python eval_analyze.py --model_path outputs/$exp_name --n_samples 10000
```

Visualize generated molecules:

```bash
python eval_sample.py --model_path outputs/$exp_name --n_samples 10000
```

Small note: If you run out of GPU memory, lower the batch size or the number of generated samples.

## Conditional Generation

### Train the Conditional FlashMol

The example below trains a conditional QM9 model for the `homo` property using the pretrained conditional teacher at `outputs/exp_cond_homo`.

```bash
python main_dmd.py \
  --exp_name dmd_qm9_32step_homo \
  --teacher_path outputs/exp_cond_homo \
  --train_diffusion \
  --step_num 32 \
  --n_epochs 11 \
  --batch_size 32 \
  --n_stability_samples 10000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 192 \
  --n_layers 9 \
  --normalize_factors '[1,8,1]' \
  --test_epochs 2 \
  --ema_decay 0.9999 \
  --latent_nf 1 \
  --gan_coeffg 0.0 \
  --gan_coefff 0.2 \
  --gan_pos 0 \
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
  --conditioning homo \
  --dataset qm9_second_half \
  --include_charges False \
  --log_grad_norm
```

The `--conditioning` argument can be set to QM9 properties such as `alpha`, `gap`, `homo`, `lumo`, `mu`, and `Cv`.

### Generate Samples for Different Property Values

```bash
python eval_conditional_qm9.py \
  --generators_path outputs/dmd_qm9_32step_homo \
  --property homo \
  --n_sweeps 10 \
  --task qualitative
```

### Evaluate with Property Classifiers

Train a property classifier:

```bash
cd qm9/property_prediction
python main_qm9_prop.py \
  --num_workers 2 \
  --lr 5e-4 \
  --property homo \
  --exp_name exp_class_homo \
  --model_name egnn
cd ../..
```

Evaluate generated samples with the trained classifier:

```bash
python eval_conditional_qm9.py \
  --generators_path outputs/dmd_qm9_32step_homo \
  --classifiers_path qm9/property_prediction/outputs/exp_class_homo \
  --property homo \
  --iterations 100 \
  --batch_size 100 \
  --task edm
```

## Citation

Please consider citing FlashMol if you find this repository helpful.

```bibtex
@misc{wei2026flashmolhighqualitymoleculegeneration,
      title={FlashMol: High-Quality Molecule Generation in as Few as Four Steps}, 
      author={Xinyuan Wei and Zian Li and Shaoheng Yan and Cai Zhou and Muhan Zhang},
      year={2026},
      eprint={2605.07020},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.07020}, 
}
```

## Acknowledgements

This repository builds on the codebases of [GeoLDM](https://github.com/MinkaiXu/GeoLDM) and [AccGeoLDM](https://github.com/rlacombe/AccGeoLDM). Thanks to the authors for their excellent open-source implementations.
