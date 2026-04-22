"""Optuna hyperparameter search for DMDMolGen.

Each trial launches a full DDP training run via train_ddp.sh,
then parses the results file for mol_stable * uniqueness.

Usage:
    python optuna_search.py --node_id 1 --n_trials 10
"""

import argparse
import os
import re
import subprocess

import optuna


PARTITIONED_NODE_IDS = (1, 3, 4, 5, 7, 8, 10)
GAN_COEFFF_MIN = 0.001
GAN_COEFFF_MAX = 1.0


def parse_results(exp_name: str):
    """Parse the last epoch from ./results/{exp_name}.txt.

    Expected line format:
        Epoch N: {'mol_stable': X, 'atm_stable': Y}, Validity: V, Uniqueness: U, Novelty: N
    Returns (mol_stable, uniqueness) or (0.0, 0.0) on failure.
    """
    path = f"./results/{exp_name}.txt"
    if not os.path.exists(path):
        print(f"Results file not found: {path}")
        return 0.0, 0.0

    last_line = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        return 0.0, 0.0

    # Parse mol_stable
    m_mol = re.search(r"'mol_stable':\s*([\d.]+)", last_line)
    # Parse uniqueness
    m_uniq = re.search(r"Uniqueness:\s*([\d.]+)", last_line)

    mol_stable = float(m_mol.group(1)) if m_mol else 0.0
    uniqueness = float(m_uniq.group(1)) if m_uniq else 0.0

    return mol_stable, uniqueness


def get_partitioned_gan_coefff_range(node_id: int):
    if node_id not in PARTITIONED_NODE_IDS:
        raise ValueError(
            f"Unsupported node_id={node_id} for partitioned GAN coeff search. "
            f"Expected one of {PARTITIONED_NODE_IDS}."
        )

    partition_width = (GAN_COEFFF_MAX - GAN_COEFFF_MIN) / len(PARTITIONED_NODE_IDS)
    partition_index = PARTITIONED_NODE_IDS.index(node_id)
    low = GAN_COEFFF_MIN + partition_index * partition_width
    if partition_index == len(PARTITIONED_NODE_IDS) - 1:
        high = GAN_COEFFF_MAX
    else:
        high = low + partition_width
    return low, high


def objective(trial: optuna.Trial, node_id: int, partition_gan_coeff_by_node: bool) -> float:
    # --- Suggest hyperparameters ---
    G_lr = trial.suggest_float("G_lr", 1e-8, 1e-5, log=True)
    mu_fake_lr = trial.suggest_float("mu_fake_lr", 1e-8, 1e-5, log=True)
    disc_lr = trial.suggest_float("disc_lr", 1e-6, 1e-3, log=True)
    if partition_gan_coeff_by_node:
        gan_coefff_low, gan_coefff_high = get_partitioned_gan_coefff_range(node_id)
    else:
        gan_coefff_low, gan_coefff_high = GAN_COEFFF_MIN, GAN_COEFFF_MAX
    gan_coefff = trial.suggest_float("gan_coefff", gan_coefff_low, gan_coefff_high)
    fdiv_coeff = trial.suggest_float("fdiv_coeff", 0.01, 1.0, log=True)
    step_num_tied = trial.suggest_int("step_num_tied", 0, 3)

    # Unique experiment name
    exp_name = f"optuna_node{node_id}_trial{trial.number}"

    # Build command matching test_ddp.sh but with searched HPs
    cmd = [
        "bash", "train_ddp.sh", "8",
        "--exp_name", exp_name,
        "--teacher_path", "outputs/drugs_latent2",
        "--train_diffusion",
        "--nf", "256",
        "--n_layers", "4",
        "--latent_nf", "2",
        "--normalize_factors", "[1,4,10]",
        "--normalization_factor", "1",
        "--diffusion_noise_schedule", "polynomial_2",
        "--diffusion_noise_precision", "1e-5",
        "--diffusion_loss_type", "l2",
        "--step_num", "8",
        "--step_num_small", str(step_num_tied),
        "--step_num_large", str(step_num_tied),
        "--step_num_pow", "1.0",
        "--step_ratio", "5",
        "--n_epochs", "1",
        "--short", "True",
        "--batch_size", "6",
        "--n_stability_samples", "10000",
        "--test_epochs", "1",
        "--ema_decay", "0.9999",
        "--G_lr", str(G_lr),
        "--mu_fake_lr", str(mu_fake_lr),
        "--disc_lr", str(disc_lr),
        "--gan_coeffg", "0.0",
        "--gan_coefff", str(gan_coefff),
        "--gan_pos", "-1",
        "--r1_weight", "0.001",
        "--r1_sigma", "0.01",
        "--Tmin", "0.02",
        "--Tminpre", "0.02",
        "--tmin_liftpos", "6",
        "--reg_coeff", "0.0",
        "--use_js", "True",
        "--fdiv_coeff", str(fdiv_coeff),
        "--consist_coeff", "0.0",
        "--log_grad_norm",
        "--no_wandb",
    ]

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | Node {node_id}")
    print(f"  G_lr={G_lr:.2e}, mu_fake_lr={mu_fake_lr:.2e}, disc_lr={disc_lr:.2e}")
    print(f"  gan_coefff={gan_coefff:.4f}, fdiv_coeff={fdiv_coeff:.4f}, step_num_tied={step_num_tied}")
    if partition_gan_coeff_by_node:
        print(f"  gan_coefff_range=[{gan_coefff_low:.6f}, {gan_coefff_high:.6f}]")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Trial {trial.number} training failed with return code {result.returncode}")
        return 0.0

    mol_stable, uniqueness = parse_results(exp_name)
    score = mol_stable * uniqueness

    print(f"\nTrial {trial.number} result: mol_stable={mol_stable:.4f}, "
          f"uniqueness={uniqueness:.4f}, score={score:.4f}")

    return score


def main():
    parser = argparse.ArgumentParser(description="Optuna HP search for DMDMolGen")
    parser.add_argument("--node_id", type=int, required=True,
                        help="Node number used for unique trial naming")
    parser.add_argument("--n_trials", type=int, default=10,
                        help="Number of trials to run on this node")
    parser.add_argument("--study_name", type=str, default="dmd_hp_search",
                        help="Optuna study name")
    parser.add_argument("--reset_study", action="store_true",
                        help="Delete the per-node Optuna database before running so trial numbering restarts at 0")
    parser.add_argument("--partition_gan_coeff_by_node", action="store_true",
                        help="Restrict gan_coefff to a per-node subrange for nodes 1,3,4,5,7,8,10")
    args = parser.parse_args()

    if args.partition_gan_coeff_by_node:
        get_partitioned_gan_coefff_range(args.node_id)

    storage_path = f"optuna_node{args.node_id}.db"
    storage = f"sqlite:///{storage_path}"

    if args.reset_study and os.path.exists(storage_path):
        os.remove(storage_path)
        print(f"Deleted existing Optuna storage: {storage_path}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    completed_trials = len([
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ])
    remaining_trials = max(0, args.n_trials - completed_trials)

    print(f"Node {args.node_id}: {completed_trials} completed trials found in {storage}")
    if remaining_trials == 0:
        print(f"Target already reached; skipping optimization (target={args.n_trials}).")
    else:
        print(f"Running {remaining_trials} additional trial(s) to reach target={args.n_trials}.")
        study.optimize(
            lambda trial: objective(trial, args.node_id, args.partition_gan_coeff_by_node),
            n_trials=remaining_trials,
        )

    print(f"\n{'='*60}")
    print(f"Node {args.node_id} now has {len(study.trials)} total trial records")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
