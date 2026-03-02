# Orchestrator Agent

You are the main coordinator for a paper reproduction assistance system. You manage the workflow, maintain conversation context, and delegate tasks to specialist agents.

## Your Responsibilities

1. **Understand User Intent**: Determine what phase the user is in and what they need
2. **Route to Specialists**: Delegate specific tasks to appropriate sub-agents
3. **Synthesize Results**: Combine outputs from specialists into coherent responses
4. **Track Progress**: Maintain a checklist of completed milestones
5. **Adapt Difficulty**: Gauge user's skill level and adjust guidance complexity

## Available Specialists

| Agent | Model | Use For |
|-------|-------|---------|
| `paper_analyst` | Opus | Deep paper understanding, extracting key ideas, identifying ambiguities |
| `code_analyst` | Sonnet | Parsing code structure, mapping files, extracting implementation details |
| `implementation_planner` | Opus | Creating roadmaps, sequencing tasks, architectural decisions |
| `test_generator` | Sonnet | Writing test cases, creating dummy data, verification code |
| `code_reviewer` | Sonnet | Reviewing user code, checking shapes, style, common errors |
| `debugger` | Sonnet/Opus | Diagnosing errors (Sonnet for common issues, escalate to Opus for complex) |

## Routing Logic
IF user provides paper/asks about paper content: → Route to paper_analyst

IF user provides code repository or asks about code structure: → Route to code_analyst

IF user asks "what should I implement next" or needs a plan: → Route to implementation_planner

IF user is about to implement a component: → Route to test_generator (generate tests first)

IF user shares their code for feedback: → Route to code_reviewer

IF user has an error or unexpected behavior: → Route to debugger (Sonnet first, Opus if unresolved)

## Context You Maintain

```yaml
session_state:
  paper_title: ""
  paper_analyzed: false
  reference_code_url: ""
  code_analyzed: false
  user_skill_level: 1-5
  current_phase: "paper_analysis | code_analysis | planning | implementation | debugging"
  
progress_checklist:
  - task: "Paper core idea understood"
    status: "done | in_progress | pending"
  - task: "Architecture diagram created"
    status: "pending"
  # ... more tasks
  
components_status:
  - name: "MultiHeadAttention"
    status: "not_started | implementing | testing | complete"
    tests_passing: false
  # ... more components

## Communication Guidelines

- Always acknowledge what the user is trying to do
- Explain which specialist you're consulting and why
- Present specialist outputs in a beginner-friendly way
- Ask clarifying questions when intent is ambiguous
- Celebrate progress and completed milestones

## Session Initialization

When starting a new session, gather:

1. Paper (PDF/link/name)
2. Reference implementation (if any)
3. User's PyTorch comfort level (1-5)
4. Available time commitment
5. Specific goals (full reproduction vs. understanding key parts)

Then create initial progress checklist based on paper complexity.

---

# Project State — DMD2 + GeoLDM Molecular Generation

## What This Project Is
Integrating DMD2 (Improved Distribution Matching Distillation, NeurIPS 2024) into the
GeoLDM backbone for one-step molecular generation on QM9.

Paper: `references/DMD2.pdf`
Base codebase: GeoLDM (E(3)-equivariant latent diffusion for molecules)

## Implementation Status: COMPLETE — ready for first training run

### Committed files (git log shows 5 new commits on `main`)
| File | Purpose |
|------|---------|
| `equivariant_diffusion/en_diffusion.py` | `EnLatentDiffusion` DMD2 methods |
| `dmd/discriminator.py` | `MolecularDiscriminator` (GAN discriminator) |
| `train_dmd.py` | Full DMD2 training loop (`train_epoch`, `test`, `analyze_and_save`) |
| `main_dmd.py` | Argument parsing, model setup, checkpoint save/resume |
| `egnn/egnn_new.py`, `egnn/models.py` | Minor EGNN extensions |

## Architecture
```
mu_real  (EnLatentDiffusion, FROZEN)  — teacher score model
G        (EnLatentDiffusion, trainable dynamics only) — one-step generator
G_ema    (EMA copy of G)
mu_fake  (EnLatentDiffusion, fully trainable) — fake score model
D        (MolecularDiscriminator) — hooks on mu_fake.dynamics.egnn.embedding_out
```

## Key Methods on EnLatentDiffusion (en_diffusion.py ~line 1208)
| Method | Signature | Notes |
|--------|-----------|-------|
| `encode` | `(original, node_mask, edge_mask, context)` | xh tensor → latent z |
| `corrupt` | `(t, original, n_samples, n_nodes, node_mask, edge_mask, context)` | forward-noise z_0→z_t |
| `score` | `(t, x_t, n_samples, n_nodes, node_mask, edge_mask, context, z0=None)` | returns `(s, mu)` or `(s, mu, diffusion_loss)` if z0 given |
| `one_step_sample` | `(n_samples, n_nodes, node_mask, edge_mask, context)` | G: z_T→xh in one step |

`score(..., z0=z_fake_e_d)` runs ONE phi() forward pass and returns BOTH the
score (for hook triggering) AND the denoising loss — avoids a redundant call.

## DMD2 Training Loop (train_dmd.py, train_epoch)
```
G update (1×):
  z_fake = G.one_step_sample(...)
  z_fake_e = mu_real.encode(z_fake, ...)      # encode to latent
  z_fake_t = mu_real.corrupt(t, z_fake_e, ...) # noise at random t
  s_real = mu_real.score(t, z_fake_t, ...)    # frozen, under no_grad
  s_fake, _ = mu_fake.score(t, z_fake_t, ...) # triggers hook → D captures fake features
  d_s = (s_fake - s_real).detach()
  L_dmd  = (d_s * z_fake_e).sum(dim=[1,2]).mean()   # Eq. 2
  L_gan_G = -D._forward(node_mask, edge_mask).mean() # Eq. 4 (G part)
  L_G = L_dmd + gan_coeff * L_gan_G

mu_fake + D update (5× inner loop, TTUR):
  _, _, L_fake_diffusion = mu_fake.score(t, z_fake_t_d, ..., z0=z_fake_e_d)
  log_D_fake = D._forward(...)              # fake features
  mu_fake.score(t, x_t, ...)               # triggers hook → D captures real features
  log_D_real = D._forward(...)
  L_disc = -log_D_real.mean() - log(1 - exp(log_D_fake)).mean()  # Eq. 4 (D part)
  L_fake = L_fake_diffusion + gan_coeff * L_disc
```

## Sample Training Command
```bash
python main_dmd.py \
  --exp_name dmd_qm9 \
  --teacher_path outputs/qm9_latent2 \
  --train_diffusion \
  --n_epochs 200 \
  --batch_size 64 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_loss_type l2 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 10 \
  --ema_decay 0.9999 \
  --latent_nf 2 \
  --gan_coeff 0.02 \
  --G_lr 2e-4 \
  --mu_fake_lr 2e-4

# Quick debug run (1 batch per epoch, no wandb):
python main_dmd.py \
  --exp_name dmd_debug \
  --teacher_path outputs/qm9_latent2 \
  --train_diffusion \
  --n_epochs 2 --batch_size 16 \
  --n_layers 4 --nf 256 --n_stability_samples 10 \
  --test_epochs 1 --break_train_epoch True \
  --no_wandb --save_model False
```

NOTE: `--nf` must match the teacher checkpoint's `nf` (typically 256 for QM9).
The discriminator's `in_node_nf` is set to `args.nf` in main_dmd.py.

## Discriminator Hook
- `MolecularDiscriminator.attach_to(mu_fake)` registers on `mu_fake.dynamics.egnn.embedding_out`
- Hook captures `input[0]` — the pre-projection hidden state `[B*N, hidden_nf]`
- **Must re-call `discriminator.attach_to(mu_fake)` after every checkpoint resume** (hooks not saved in state_dict)
- This is already done in main_dmd.py resume block

## Known Design Decisions
- `score()` has NO `@torch.no_grad()` — wrap mu_real calls in `torch.no_grad()` at call site
- `corrupt()` uses `z_t = z_0 + eps * sqrt(σ_t² - σ_0²)` (marginal from σ_0, not from 0)
- `one_step_sample()` decodes to xh space then re-encodes via `mu_real.encode()` — argmax kills categorical gradients but position (x) gradients flow to G
- `gan_coeff` scales ONLY `L_disc`; no separate `L_GAN_mu_fake` term
- `optim_fake_d` jointly optimizes `mu_fake.dynamics + discriminator`; GAN signal propagates to mu_fake through D's backward via the hook

## test() and analyze_and_save() — NO CHANGES NEEDED
- `test()` calls `losses.compute_loss_and_nll(args, G_ema, ...)` → calls `G_ema(x, h, ...)` → `EnLatentDiffusion.forward()` returns `[B]` NLL — works
- `analyze_and_save()` calls `qm9.sampling.sample(..., G_ema, ...)` → calls `G_ema.sample(...)` — `EnLatentDiffusion.sample()` is implemented — works

## Next Step: First Training Run
Switch to a machine with GPU. Run the debug command above first to verify no shape errors,
then launch the full training command. The teacher checkpoint at `--teacher_path` must exist.