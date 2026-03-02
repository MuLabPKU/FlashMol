"""
Test script for one_step_sample, sample_one_step, and test() functions.
Constructs a tiny EnLatentDiffusion model from scratch (no checkpoint needed).
Run: python test_sample_and_eval.py
"""
import sys
import types

# Mock wandb before any imports that depend on it
wandb_mock = types.ModuleType('wandb')
wandb_mock.log = lambda *a, **kw: None
wandb_mock.init = lambda *a, **kw: None
wandb_mock.save = lambda *a, **kw: None
wandb_mock.Settings = lambda *a, **kw: {}
sys.modules['wandb'] = wandb_mock

import copy
import torch
import argparse
import numpy as np
from configs.datasets_config import get_dataset_info
from qm9.models import get_autoencoder
from equivariant_diffusion.en_diffusion import EnLatentDiffusion
from egnn.models import EGNN_dynamics_QM9


def make_tiny_args():
    """Minimal args to construct a tiny model."""
    args = argparse.Namespace(
        # Model architecture (tiny for testing)
        nf=32,
        n_layers=2,
        latent_nf=2,
        attention=True,
        tanh=True,
        model='egnn_dynamics',
        norm_constant=1,
        inv_sublayers=1,
        sin_embedding=False,
        normalization_factor=1,
        aggregation_method='sum',
        condition_time=True,
        # Diffusion
        diffusion_steps=50,
        diffusion_noise_schedule='polynomial_2',
        diffusion_noise_precision=1e-5,
        diffusion_loss_type='l2',
        probabilistic_model='diffusion',
        # Data
        dataset='qm9',
        remove_h=False,
        include_charges=True,
        normalize_factors=[1, 4, 1],
        kl_weight=0.01,
        # Training (for test() compatibility)
        n_report_steps=1,
        context_node_nf=0,
        conditioning=[],
        augment_noise=0,
        trainable_ae=False,
        # Misc
        cuda=False,
        no_cuda=True,
        ae_path=None,
    )
    return args


def make_tiny_model(args, dataset_info, device):
    """Build a tiny EnLatentDiffusion model from scratch."""
    # Build VAE (autoencoder)
    # We need a dummy dataloader for DistributionNodes
    from qm9.models import DistributionNodes
    histogram = dataset_info['n_nodes']
    nodes_dist = DistributionNodes(histogram)

    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    from egnn.models import EGNN_encoder_QM9, EGNN_decoder_QM9
    from equivariant_diffusion.en_diffusion import EnHierarchicalVAE

    encoder = EGNN_encoder_QM9(
        in_node_nf=in_node_nf, context_node_nf=0, out_node_nf=args.latent_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=1,
        attention=args.attention, tanh=args.tanh, mode=args.model,
        norm_constant=args.norm_constant, inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges)

    decoder = EGNN_decoder_QM9(
        in_node_nf=args.latent_nf, context_node_nf=0, out_node_nf=in_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model,
        norm_constant=args.norm_constant, inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges)

    vae = EnHierarchicalVAE(
        encoder=encoder, decoder=decoder,
        in_node_nf=in_node_nf, n_dims=3,
        latent_node_nf=args.latent_nf,
        kl_weight=args.kl_weight,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges)

    # Build dynamics for the latent diffusion
    dynamics_in_node_nf = args.latent_nf + 1  # +1 for time conditioning

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=0,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers,
        attention=args.attention, tanh=args.tanh, mode=args.model,
        norm_constant=args.norm_constant, inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method)

    model = EnLatentDiffusion(
        vae=vae,
        trainable_ae=False,
        dynamics=net_dynamics,
        in_node_nf=args.latent_nf,
        n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges)

    return model, nodes_dist


def make_masks(batch_size, n_nodes, max_n_nodes, device):
    """Create node_mask and edge_mask."""
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, :n_nodes] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(max_n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    return node_mask, edge_mask


def test_one_step_sample():
    """Test EnLatentDiffusion.one_step_sample() output shapes."""
    print("=" * 60)
    print("TEST 1: one_step_sample()")
    print("=" * 60)

    args = make_tiny_args()
    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)
    model, nodes_dist = make_tiny_model(args, dataset_info, device)
    model.to(device)
    model.eval()

    batch_size = 4
    n_nodes = 10
    max_n_nodes = dataset_info['max_n_nodes']  # 29
    node_mask, edge_mask = make_masks(batch_size, n_nodes, max_n_nodes, device)

    with torch.no_grad():
        xh = model.one_step_sample(batch_size, max_n_nodes, node_mask, edge_mask, context=None)

    # one_step_sample returns data-space xh; use VAE dims for splitting
    n_dims = model.vae.n_dims  # 3
    num_atom_types = model.vae.in_node_nf - int(model.vae.include_charges)  # 5
    expected_features = n_dims + model.vae.in_node_nf  # 3 + 6 = 9

    print(f"  xh shape: {xh.shape}")
    print(f"  Expected: [{batch_size}, {max_n_nodes}, {expected_features}]")
    assert xh.shape == (batch_size, max_n_nodes, expected_features), \
        f"Shape mismatch: {xh.shape} != ({batch_size}, {max_n_nodes}, {expected_features})"

    # Verify masked positions are zero
    masked_vals = xh[:, n_nodes:, :]
    assert torch.allclose(masked_vals, torch.zeros_like(masked_vals), atol=1e-6), \
        "Masked positions should be zero"

    # Split and check (same logic as sample_one_step wrappers)
    x = xh[:, :, :n_dims]
    one_hot = xh[:, :, n_dims:n_dims + num_atom_types]
    charges = xh[:, :, n_dims + num_atom_types:]
    print(f"  x shape: {x.shape}  (positions)")
    print(f"  one_hot shape: {one_hot.shape}  (atom types, expect {num_atom_types})")
    print(f"  charges shape: {charges.shape}  (charges)")

    assert not torch.any(torch.isnan(xh)), "NaN detected in one_step_sample output!"

    print("  PASSED\n")
    return model, nodes_dist


def test_one_step_sample_latent(model):
    """Test EnLatentDiffusion.one_step_sample_latent() output shapes."""
    print("=" * 60)
    print("TEST 2: one_step_sample_latent()")
    print("=" * 60)

    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)
    batch_size = 4
    n_nodes = 10
    max_n_nodes = dataset_info['max_n_nodes']
    node_mask, edge_mask = make_masks(batch_size, n_nodes, max_n_nodes, device)

    with torch.no_grad():
        z0 = model.one_step_sample_latent(batch_size, max_n_nodes, node_mask, edge_mask, context=None)

    n_dims = model.n_dims  # 3
    latent_nf = model.in_node_nf  # 2
    expected_latent_features = n_dims + latent_nf

    print(f"  z0 shape: {z0.shape}")
    print(f"  Expected: [{batch_size}, {max_n_nodes}, {expected_latent_features}]")
    assert z0.shape == (batch_size, max_n_nodes, expected_latent_features), \
        f"Shape mismatch: {z0.shape} != ({batch_size}, {max_n_nodes}, {expected_latent_features})"

    # Verify masked positions are zero
    masked_vals = z0[:, n_nodes:, :]
    assert torch.allclose(masked_vals, torch.zeros_like(masked_vals), atol=1e-6), \
        "Masked positions should be zero"

    assert not torch.any(torch.isnan(z0)), "NaN detected in one_step_sample_latent output!"

    # Verify z0 can be fed into corrupt() and score() (the training pipeline)
    T = model.T
    Tmin = max(1, int(0.02 * T))
    Tmax = int(0.98 * T)
    t_int = torch.randint(Tmin, Tmax, (batch_size, 1), device=device).float()
    noise_t = t_int / T

    z_t = model.corrupt(noise_t, z0, batch_size, max_n_nodes, node_mask, edge_mask, context=None)
    print(f"  corrupt(z0) → z_t shape: {z_t.shape}  (should match z0)")
    assert z_t.shape == z0.shape, f"corrupt output shape mismatch: {z_t.shape}"

    s, mu = model.score(noise_t, z_t, batch_size, max_n_nodes, node_mask, edge_mask, context=None)
    print(f"  score(z_t) → s shape: {s.shape}, mu shape: {mu.shape}")
    assert s.shape == z_t.shape, f"score output shape mismatch: {s.shape}"
    assert not torch.any(torch.isnan(s)), "NaN detected in score output!"

    print("  PASSED\n")


def test_sample_one_step_wrapper(model, nodes_dist):
    """Test the sample_one_step wrapper from train_dmd.py."""
    print("=" * 60)
    print("TEST 3: sample_one_step() wrapper (train_dmd.py)")
    print("=" * 60)

    from train_dmd import sample_one_step

    args = make_tiny_args()
    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)

    batch_size = 4
    nodesxsample = nodes_dist.sample(batch_size)
    print(f"  nodesxsample: {nodesxsample.tolist()}")

    model.eval()
    one_hot, charges, x, node_mask = sample_one_step(
        args, device, model, dataset_info, prop_dist=None, nodesxsample=nodesxsample)

    max_n_nodes = dataset_info['max_n_nodes']
    num_atom_types = len(dataset_info['atom_decoder'])  # 5

    print(f"  x shape: {x.shape}  → expected [{batch_size}, {max_n_nodes}, 3]")
    print(f"  one_hot shape: {one_hot.shape}  → expected [{batch_size}, {max_n_nodes}, {num_atom_types}]")
    print(f"  charges shape: {charges.shape}")
    print(f"  node_mask shape: {node_mask.shape}")

    assert x.shape == (batch_size, max_n_nodes, 3)
    assert one_hot.shape == (batch_size, max_n_nodes, num_atom_types)
    assert node_mask.shape == (batch_size, max_n_nodes, 1)

    assert not torch.any(torch.isnan(x)), "NaN in x!"
    assert not torch.any(torch.isnan(one_hot)), "NaN in one_hot!"

    print("  PASSED\n")


def test_eval_test_function(model, nodes_dist):
    """Test the test() function from train_dmd.py with a dummy dataloader."""
    print("=" * 60)
    print("TEST 4: test() function (NLL evaluation)")
    print("=" * 60)

    from train_dmd import test as test_fn

    args = make_tiny_args()
    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)

    # Create a dummy batch
    batch_size = 4
    max_n_nodes = dataset_info['max_n_nodes']
    n_nodes = 10
    num_classes = len(dataset_info['atom_decoder'])

    positions = torch.randn(batch_size, max_n_nodes, 3) * 0.1
    # Zero out masked positions
    atom_mask = torch.zeros(batch_size, max_n_nodes)
    atom_mask[:, :n_nodes] = 1
    positions = positions * atom_mask.unsqueeze(2)
    # Remove mean (required for equivariance)
    from equivariant_diffusion.utils import remove_mean_with_mask
    node_mask_3d = atom_mask.unsqueeze(2)
    positions = remove_mean_with_mask(positions, node_mask_3d)

    one_hot = torch.zeros(batch_size, max_n_nodes, num_classes)
    one_hot[:, :n_nodes, 0] = 1  # all Carbon

    charges = torch.zeros(batch_size, max_n_nodes, 1)
    charges[:, :n_nodes, 0] = 6  # Carbon charge

    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag = ~torch.eye(max_n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask = (edge_mask * diag).view(batch_size * max_n_nodes * max_n_nodes, 1)

    dummy_data = {
        'positions': positions,
        'atom_mask': atom_mask,
        'edge_mask': edge_mask,
        'one_hot': one_hot,
        'charges': charges,
    }

    # Wrap in a list to act as a dataloader
    dummy_loader = [dummy_data]

    model.eval()
    try:
        nll = test_fn(args=args, loader=dummy_loader, epoch=0,
                      eval_model=model, device=device, dtype=torch.float32,
                      property_norms=None, nodes_dist=nodes_dist, partition='Test')
        print(f"  NLL value: {nll:.4f}")
        assert not np.isnan(nll), "NaN in NLL!"
        print("  PASSED\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_analyze_and_save(model, nodes_dist):
    """Test analyze_and_save from train_dmd.py (without wandb)."""
    print("=" * 60)
    print("TEST 5: analyze_and_save()")
    print("=" * 60)

    from train_dmd import analyze_and_save

    args = make_tiny_args()
    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)

    model.eval()
    try:
        validity_dict = analyze_and_save(
            epoch=0, model_sample=model, nodes_dist=nodes_dist,
            args=args, device=device, dataset_info=dataset_info,
            prop_dist=None, n_samples=4, batch_size=4)
        print(f"  Validity dict: {validity_dict}")
        print("  PASSED\n")
    except Exception as e:
        print(f"  FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    return True


def test_eval_analyze_sample_one_step(model, nodes_dist):
    """Test sample_one_step from eval_analyze.py."""
    print("=" * 60)
    print("TEST 6: sample_one_step() wrapper (eval_analyze.py)")
    print("=" * 60)

    from eval_analyze import sample_one_step

    args = make_tiny_args()
    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)

    batch_size = 4
    nodesxsample = nodes_dist.sample(batch_size)

    model.eval()
    one_hot, charges, x, node_mask = sample_one_step(
        args, device, model, dataset_info, prop_dist=None, nodesxsample=nodesxsample)

    max_n_nodes = dataset_info['max_n_nodes']
    num_atom_types = len(dataset_info['atom_decoder'])

    print(f"  x shape: {x.shape}  → expected [{batch_size}, {max_n_nodes}, 3]")
    print(f"  one_hot shape: {one_hot.shape}  → expected [{batch_size}, {max_n_nodes}, {num_atom_types}]")
    print(f"  charges shape: {charges.shape}")
    print(f"  node_mask shape: {node_mask.shape}")

    assert x.shape == (batch_size, max_n_nodes, 3)
    assert one_hot.shape == (batch_size, max_n_nodes, num_atom_types)
    assert not torch.any(torch.isnan(x)), "NaN in x!"
    assert not torch.any(torch.isnan(one_hot)), "NaN in one_hot!"

    print("  PASSED\n")


if __name__ == '__main__':
    print("\nDMD Sample & Eval Test Suite")
    print("=" * 60)
    print("Building tiny model (no checkpoint needed)...\n")

    model, nodes_dist = test_one_step_sample()
    test_one_step_sample_latent(model)
    test_sample_one_step_wrapper(model, nodes_dist)
    test_eval_test_function(model, nodes_dist)
    test_analyze_and_save(model, nodes_dist)
    test_eval_analyze_sample_one_step(model, nodes_dist)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
