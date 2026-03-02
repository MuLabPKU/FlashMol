"""
Debug script for EnLatentDiffusion — focused on DMD training numerical health.
Tests: score magnitudes, d_s scale, z_fake_e vs x_e scale, NaN detection,
       gradient flow, L_dmd / L_reg values.

Run: python debug_en_diffusion.py
"""
import sys, types, copy
wandb_mock = types.ModuleType('wandb')
wandb_mock.log = lambda *a, **kw: None
wandb_mock.init = lambda *a, **kw: None
wandb_mock.save = lambda *a, **kw: None
wandb_mock.Settings = lambda *a, **kw: {}
sys.modules['wandb'] = wandb_mock

import torch
import argparse
import numpy as np
from configs.datasets_config import get_dataset_info
from equivariant_diffusion.en_diffusion import EnLatentDiffusion, EnHierarchicalVAE
from egnn.models import EGNN_dynamics_QM9, EGNN_encoder_QM9, EGNN_decoder_QM9
from qm9.models import DistributionNodes
from equivariant_diffusion.utils import remove_mean_with_mask


# ─── helpers ─────────────────────────────────────────────────────────────────

def sep(title=""):
    w = 64
    if title:
        print(f"\n{'─'*4} {title} {'─'*(w-6-len(title))}")
    else:
        print("─" * w)


def stat(name, t):
    """Print min/max/mean/std/nan of a tensor."""
    t = t.detach().float()
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    print(f"  {name:30s}  min={t.min():.3e}  max={t.max():.3e}"
          f"  mean={t.mean():.3e}  std={t.std():.3e}"
          f"{'  *** NaN ***' if nan else ''}"
          f"{'  *** Inf ***' if inf else ''}")
    return nan or inf


def make_args(T=500, nf=32, n_layers=2, latent_nf=2):
    return argparse.Namespace(
        nf=nf, n_layers=n_layers, latent_nf=latent_nf,
        attention=True, tanh=True, model='egnn_dynamics',
        norm_constant=1, inv_sublayers=1, sin_embedding=False,
        normalization_factor=1, aggregation_method='sum',
        condition_time=True,
        diffusion_steps=T,
        diffusion_noise_schedule='polynomial_2',
        diffusion_noise_precision=1e-5,
        diffusion_loss_type='l2',
        probabilistic_model='diffusion',
        dataset='qm9', remove_h=False, include_charges=True,
        normalize_factors=[1, 4, 1], kl_weight=0.01,
        n_report_steps=1, context_node_nf=0, conditioning=[],
        augment_noise=0, trainable_ae=False,
        cuda=False, no_cuda=True, ae_path=None,
    )


def build_model(args, dataset_info, device):
    in_node_nf = len(dataset_info['atom_decoder']) + int(args.include_charges)
    nodes_dist = DistributionNodes(dataset_info['n_nodes'])

    encoder = EGNN_encoder_QM9(
        in_node_nf=in_node_nf, context_node_nf=0, out_node_nf=args.latent_nf,
        n_dims=3, device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(),
        n_layers=1, attention=args.attention, tanh=args.tanh,
        mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges)

    decoder = EGNN_decoder_QM9(
        in_node_nf=args.latent_nf, context_node_nf=0, out_node_nf=in_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers, attention=args.attention, tanh=args.tanh,
        mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        include_charges=args.include_charges)

    vae = EnHierarchicalVAE(
        encoder=encoder, decoder=decoder, in_node_nf=in_node_nf, n_dims=3,
        latent_node_nf=args.latent_nf, kl_weight=args.kl_weight,
        norm_values=args.normalize_factors, include_charges=args.include_charges)

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=args.latent_nf + 1, context_node_nf=0, n_dims=3,
        device=device, hidden_nf=args.nf, act_fn=torch.nn.SiLU(),
        n_layers=args.n_layers, attention=args.attention, tanh=args.tanh,
        mode=args.model, norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers, sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method)

    model = EnLatentDiffusion(
        vae=vae, trainable_ae=False, dynamics=net_dynamics,
        in_node_nf=args.latent_nf, n_dims=3,
        timesteps=args.diffusion_steps,
        noise_schedule=args.diffusion_noise_schedule,
        noise_precision=args.diffusion_noise_precision,
        loss_type=args.diffusion_loss_type,
        norm_values=args.normalize_factors,
        include_charges=args.include_charges)

    return model, nodes_dist


def make_masks(B, n_nodes, max_n_nodes, device):
    node_mask = torch.zeros(B, max_n_nodes)
    node_mask[:, :n_nodes] = 1
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag = ~torch.eye(max_n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask = (edge_mask * diag).view(B * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)
    return node_mask, edge_mask


def make_real_latent(model, B, n_nodes, max_n_nodes, node_mask, edge_mask, device):
    """Simulate what train_epoch does: encode a random real batch."""
    in_nf = model.vae.in_node_nf
    num_classes = in_nf - int(model.include_charges)
    x = torch.randn(B, max_n_nodes, 3, device=device) * 0.3
    x = x * node_mask
    x = remove_mean_with_mask(x, node_mask)
    one_hot = torch.zeros(B, max_n_nodes, num_classes, device=device)
    one_hot[:, :n_nodes, 0] = 1
    charges = torch.zeros(B, max_n_nodes, 1, device=device)
    xh = torch.cat([x, one_hot, charges], dim=2)
    with torch.no_grad():
        x_e = model.encode(xh, node_mask, edge_mask, None)
    return x_e


# ─── TEST 1: sigma at various t ───────────────────────────────────────────────

def test_sigma_schedule(model):
    sep("TEST 1: sigma(t) at key timesteps")
    T = model.T
    fractions = [0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.98]
    dummy = torch.zeros(1, 1, 1)
    rows = []
    for frac in fractions:
        t_int = max(1, int(frac * T))
        t = torch.tensor([[t_int / T]])
        gamma_t = model.gamma(t)
        sigma_t = model.sigma(gamma_t, dummy).item()
        denom = (sigma_t + 1e-8) ** 2
        rows.append((frac, t_int, sigma_t, 1/denom, denom))
    print(f"  {'t/T':>6}  {'t_int':>6}  {'sigma_t':>12}  {'1/sigma_t^2':>14}  {'(sigma+1e-8)^2':>16}")
    for frac, t_int, sigma_t, inv_denom, denom in rows:
        print(f"  {frac:>6.2f}  {t_int:>6d}  {sigma_t:>12.6f}  {inv_denom:>14.2e}  {denom:>16.6e}")


# ─── TEST 2: score magnitudes ─────────────────────────────────────────────────

def test_score_magnitudes(mu_real, mu_fake, B, n_nodes, max_n_nodes, node_mask, edge_mask, device, T):
    sep("TEST 2: score magnitudes at Tmin=0.02*T vs Tmin=0.20*T")
    Tmin_old = max(1, int(0.02 * T))
    Tmin_new = max(1, int(0.20 * T))

    for label, tmin in [("Tmin=0.02*T", Tmin_old), ("Tmin=0.20*T", Tmin_new)]:
        t_int = torch.full((B, 1), tmin, dtype=torch.float, device=device)
        noise_t = t_int / T

        z_fake_e = mu_real.one_step_sample_latent(B, max_n_nodes, node_mask, edge_mask, None)
        z_fake_t = mu_real.corrupt(noise_t, z_fake_e, B, max_n_nodes, node_mask, edge_mask, None)

        with torch.no_grad():
            s_real, _ = mu_real.score(noise_t, z_fake_t, B, max_n_nodes, node_mask, edge_mask, None)
            s_fake, _ = mu_fake.score(noise_t, z_fake_t, B, max_n_nodes, node_mask, edge_mask, None)

        d_s = (s_fake - s_real)
        print(f"\n  [{label}]  t_int={tmin}")
        bad = stat("  s_real", s_real)
        bad |= stat("  s_fake", s_fake)
        bad |= stat("  d_s (score diff)", d_s)
        L_dmd_val = (d_s * z_fake_e).sum(dim=[1, 2]).mean()
        print(f"  {'L_dmd (raw)':30s}  {L_dmd_val.item():.4e}")
        if bad:
            print("  *** NaN/Inf detected above ***")


# ─── TEST 3: z_fake_e vs x_e scale ───────────────────────────────────────────

def test_latent_scales(model, x_e, node_mask, edge_mask, B, max_n_nodes, device):
    sep("TEST 3: z_fake_e scale vs real x_e scale")
    z_fake_e = model.one_step_sample_latent(B, max_n_nodes, node_mask, edge_mask, None)

    xe_sq = x_e.pow(2).sum(dim=[1, 2]).mean().item()
    zf_sq = z_fake_e.pow(2).sum(dim=[1, 2]).mean().item()
    L_reg = (zf_sq - xe_sq) ** 2
    print(f"  x_e  ||·||² mean over batch : {xe_sq:.4f}")
    print(f"  z_fake_e ||·||² mean        : {zf_sq:.4f}")
    print(f"  L_reg = (z² - x²)²          : {L_reg:.4f}")
    stat("  x_e", x_e)
    stat("  z_fake_e", z_fake_e)


# ─── TEST 4: gradient flow through z_fake_e ──────────────────────────────────

def test_gradient_flow(G, mu_real, mu_fake, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, T):
    sep("TEST 4: gradient flow — does grad reach G.dynamics?")
    Tmin = max(1, int(0.20 * T))
    Tmax = int(0.98 * T)
    t_int = torch.randint(Tmin, Tmax, (B, 1), device=device).float()
    noise_t = t_int / T

    z_fake_e = G.one_step_sample_latent(B, max_n_nodes, node_mask, edge_mask, None)
    z_fake_t = mu_real.corrupt(noise_t, z_fake_e, B, max_n_nodes, node_mask, edge_mask, None)

    with torch.no_grad():
        s_real, _ = mu_real.score(noise_t, z_fake_t, B, max_n_nodes, node_mask, edge_mask, None)

    z_fake_t_d = z_fake_t.detach()
    s_fake, _ = mu_fake.score(noise_t, z_fake_t_d, B, max_n_nodes, node_mask, edge_mask, None)

    d_s = (s_fake - s_real).detach()
    L_dmd = (d_s * z_fake_e).sum(dim=[1, 2]).mean()

    L_dmd.backward()

    has_grad = False
    no_grad_params = []
    for name, p in G.dynamics.named_parameters():
        if p.grad is not None:
            has_grad = True
        else:
            no_grad_params.append(name)

    print(f"  L_dmd = {L_dmd.item():.4e}")
    print(f"  G.dynamics receives gradient: {has_grad}")
    if not has_grad:
        print("  *** GRADIENT DOES NOT REACH G.dynamics — broken! ***")
    if no_grad_params:
        print(f"  Params with no grad ({len(no_grad_params)}): {no_grad_params[:3]} ...")

    # Grad norms per param
    total_gnorm = 0.0
    for p in G.dynamics.parameters():
        if p.grad is not None:
            total_gnorm += p.grad.norm().item() ** 2
    total_gnorm = total_gnorm ** 0.5
    print(f"  Total gradient norm (before clip): {total_gnorm:.4e}")
    print(f"  (clip threshold is 5.0 → ratio: {total_gnorm/5.0:.1f}x)")


# ─── TEST 5: NaN propagation through score ───────────────────────────────────

def test_nan_propagation(model, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, T):
    sep("TEST 5: NaN propagation — inject NaN into z_fake_e")
    Tmin = max(1, int(0.20 * T))
    t_int = torch.full((B, 1), Tmin, dtype=torch.float, device=device)
    noise_t = t_int / T

    # Normal run
    z_fake_e = model.one_step_sample_latent(B, max_n_nodes, node_mask, edge_mask, None)
    z_fake_t = model.corrupt(noise_t, z_fake_e, B, max_n_nodes, node_mask, edge_mask, None)
    s, mu = model.score(noise_t, z_fake_t, B, max_n_nodes, node_mask, edge_mask, None)
    bad = stat("  score (clean input)", s)
    if not bad:
        print("  Clean input → no NaN in score ✓")

    # Inject NaN into first feature dim (index 3 = latent feature, not position)
    # Position dims [0,1,2] are checked by assert_mean_zero; inject into feature dim instead
    z_poisoned = z_fake_e.clone()
    z_poisoned[0, 0, 3] = float('nan')  # feature dim, safe from mean-zero assert
    z_t_poisoned = model.corrupt(noise_t, z_poisoned, B, max_n_nodes, node_mask, edge_mask, None)
    s_p, _ = model.score(noise_t, z_t_poisoned, B, max_n_nodes, node_mask, edge_mask, None)
    bad2 = stat("  score (NaN-poisoned input)", s_p)
    if bad2:
        print("  NaN propagates through score — EGNN reset may mask but not cure this")


# ─── TEST 6: corrupt → score round-trip consistency ──────────────────────────

def test_corrupt_score_consistency(model, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, T):
    sep("TEST 6: corrupt → score(z0=z_fake_e) diffusion loss check")
    Tmin = max(1, int(0.20 * T))
    Tmax = int(0.98 * T)
    t_int = torch.randint(Tmin, Tmax, (B, 1), device=device).float()
    noise_t = t_int / T

    z_fake_e = model.one_step_sample_latent(B, max_n_nodes, node_mask, edge_mask, None).detach()
    z_fake_t = model.corrupt(noise_t, z_fake_e, B, max_n_nodes, node_mask, edge_mask, None)

    s, mu, L_diff = model.score(noise_t, z_fake_t, B, max_n_nodes, node_mask, edge_mask, None, z_fake_e)
    print(f"  L_fake_diffusion = {L_diff.item():.4f}")
    bad = stat("  eps_t (via score)", s)
    if not bad:
        print("  No NaN in diffusion loss forward pass ✓")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\nEnLatentDiffusion — DMD Debug Suite")
    sep()

    device = torch.device('cpu')
    dataset_info = get_dataset_info('qm9', False)

    # Use T=500 to match real training
    args = make_args(T=500, nf=32, n_layers=2, latent_nf=2)
    print("Building tiny model (T=500, nf=32, n_layers=2, latent_nf=2) ...")

    mu_real, nodes_dist = build_model(args, dataset_info, device)
    mu_real.eval()
    for p in mu_real.parameters():
        p.requires_grad_(False)

    G = copy.deepcopy(mu_real)
    G.train()
    for p in G.vae.parameters():
        p.requires_grad_(False)
    for p in G.dynamics.parameters():
        p.requires_grad_(True)

    mu_fake = copy.deepcopy(mu_real)
    mu_fake.train()
    for p in mu_fake.parameters():
        p.requires_grad_(True)

    B, n_nodes, max_n_nodes = 4, 10, dataset_info['max_n_nodes']
    node_mask, edge_mask = make_masks(B, n_nodes, max_n_nodes, device)
    x_e = make_real_latent(mu_real, B, n_nodes, max_n_nodes, node_mask, edge_mask, device)

    test_sigma_schedule(mu_real)
    test_score_magnitudes(mu_real, mu_fake, B, n_nodes, max_n_nodes, node_mask, edge_mask, device, args.diffusion_steps)
    test_latent_scales(G, x_e, node_mask, edge_mask, B, max_n_nodes, device)
    test_gradient_flow(G, mu_real, mu_fake, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, args.diffusion_steps)
    test_nan_propagation(mu_real, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, args.diffusion_steps)
    test_corrupt_score_consistency(mu_real, B, max_n_nodes, n_nodes, node_mask, edge_mask, device, args.diffusion_steps)

    sep()
    print("Debug suite complete.\n")
