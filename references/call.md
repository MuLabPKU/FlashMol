 ---
  Assumed concrete values (QM9, debug run):
  B  = 16   (batch_size)
  N  = 29   (max nodes in QM9)
  C  = 5    (num_classes: H,C,N,O,F)
  nf = 256  (EGNN hidden dim, args.nf)
  latent_nf = 2   (from teacher checkpoint qm9_latent2)
  in_node_nf (VAE input) = C + 1 charge = 6
  in_node_nf (EnLatentDiffusion) = latent_nf = 2

  ---
  Data loading (lines 29–48)

  x          [16, 29,  3]   # 3D atom positions
  node_mask  [16, 29,  1]
  edge_mask  [16, 841]      # 841 = 29*29
  one_hot    [16, 29,  5]
  charges    [16, 29,  1]

  Encode real data (lines 62–66)

  x_xh = cat([x, one_hot, charges], dim=2)
                             [16, 29,  9]   # 3+5+1

  # Inside mu_real.encode(x_xh, ...):
  x_mu   [16, 29,  3]   # positional mean from VAE encoder
  x_sig  [16,  1,  1]   # scalar sigma_0_x (scalar, broadcast)
  h_mu   [16, 29,  2]   # latent h mean
  h_sig  [16,  1,  2]   # latent h sigma

  mu     = cat([x_mu, h_mu], dim=2)                   → [16, 29,  5]
  sig    = cat([x_sig.expand(-1,-1,3), h_sig], dim=2) → [16,  1,  5]
  eps    = sample_combined_position_feature_noise(...)  [16, 29,  5]
  encoded = mu + sig * eps                             → [16, 29,  5]

  x_e    [16, 29,  5]   # real latent

  Timestep (lines 71–72)

  t_int    [16,  1]        # integer in [Tmin, Tmax]
  noise_t  [16,  1]        # t_int / T ∈ (0,1)

  Generator: one_step_sample (line 80)

  # Inside G.one_step_sample(16, 29, ...):
  z_T   = noise ~ N(0,I)                 [16, 29,  5]   # 3 pos + 2 latent h
  z0    = sample_p0_from_pT(z_T, ...)    [16, 29,  5]   # one denoise step T→0
  x, h  = super().sample_p_xh_given_z0(z0, ...)
          x                              [16, 29,  3]
          h['categorical']               [16, 29,  5]
          h['integer']                   [16, 29,  1]
  xh    = cat([x, h_cat, h_int], dim=2)  [16, 29,  9]

  z_fake [16, 29,  9]   # fake molecule in original data space

  Encode fake sample (line 81)

  z_fake_e = mu_real.encode(z_fake, ...)
             [16, 29,  5]   # fake latent (same path as x_e)

  Corrupt (line 83)

  # Inside mu_real.corrupt(noise_t, z_fake_e, ...):
  gamma_t, gamma_0  → scalars broadcast to [16, 1, 1]
  sigma_0, sigma_t  → [16, 1, 1]
  d_sigma           → [16, 1, 1]
  noise             → [16, 29, 5]   # super().sample_combined...
  zt = z_fake_e + noise * d_sigma   [16, 29, 5]  # [16,29,5]+[16,29,5]*[16,1,1]

  z_fake_t  [16, 29,  5]

  Score (lines 86–89)

  # Inside score(noise_t, z_fake_t, ...):
  eps_t = phi(x_t, t, ...)   [16, 29,  5]   # EGNN denoiser output
  alpha_t, sigma_t            [16,  1,  1]   # broadcast scalars
  mu    = (x_t/alpha_t) - (sigma2/alpha_t/sigma_t) * eps_t
                              [16, 29,  5]
  s     = -(x_t - alpha_t*mu) / sigma_t^2
                              [16, 29,  5]

  s_real  [16, 29,  5]   # frozen teacher score
  s_fake  [16, 29,  5]   # mu_fake score (triggers discriminator hook)

  DMD loss (lines 92–93)

  d_s    = (s_fake - s_real).detach()   [16, 29,  5]
  L_dmd  = (d_s * z_fake_e)             [16, 29,  5]
             .sum(dim=[1,2])             [16]
             .mean()                     scalar

  Discriminator (line 96)

  # Hook captured: mu_fake.dynamics.egnn.embedding_out input[0]
  mu_fake_out  [B*N, nf]  =  [464, 256]   # pre-projection EGNN features

  # Inside discriminator._forward(node_mask, edge_mask):
  h = mu_fake_out.view(16, 29, 256)       [16, 29, 256]
  h = h.view(464, 256) * node_mask_flat   [464, 256]
  h = gnn(h, edges, ...)                  [464, 256]   # GNN message passing
  h = h.view(16, 29, 256)                 [16, 29, 256]
  h = h.sum(dim=1) / atom_num             [16,  1, 256]
  logits = mlp(h)                         [16,  1,   1]
         → squeeze → log                  [16]

  log_D_fake  [16]
  log_D_real  [16]

  Losses

  L_gan_G  = -log_D_fake.mean()                    scalar
  L_G      = L_dmd + gan_coeff * L_gan_G            scalar

  # Inner loop (×5):
  L_fake_diffusion  scalar   # from mu_fake.score(..., z0=z_fake_e_d)
  L_disc = -log_D_real.mean()
         - log(1 - exp(log_D_fake) + 1e-8).mean()   scalar
  L_fake = L_fake_diffusion + gan_coeff * L_disc    scalar

  ---
  Key shape summary:

  ┌─────────────────────┬────────────┬────────────────────────┐
  │      Variable       │   Shape    │         Notes          │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ x_xh / z_fake       │ [B, N, 9]  │ original space: 3+5+1  │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ x_e, z_fake_e       │ [B, N, 5]  │ latent: 3+latent_nf(2) │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ z_fake_t, x_t       │ [B, N, 5]  │ corrupted latent       │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ s_real, s_fake, d_s │ [B, N, 5]  │ scores                 │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ noise_t             │ [B, 1]     │ timestep               │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ mu_fake_out (hook)  │ [B*N, 256] │ EGNN bottleneck        │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ log_D_fake/real     │ [B]        │ discriminator output   │
  ├─────────────────────┼────────────┼────────────────────────┤
  │ all losses          │ scalar     │                        │
  └─────────────────────┴────────────┴────────────────────────┘