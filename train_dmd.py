import wandb
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask, \
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
import time
import torch
import torch.nn.functional as F
from equivariant_diffusion import utils as diffusion_utils


def train_epoch(args, loader, epoch, mu_real, G, G_ema, G_dp, mu_fake, discriminator,
                ema, device, dtype, property_norms, nodes_dist, gradnorm_queue,
                dataset_info, prop_dist, optim_G, optim_fake, optim_d, gan_coeffg, gan_coefff,
                reg_coeff, step_ratio, step_num):

    T = mu_real.T
    if epoch <= args.tmin_liftpos :
        Tmin = max(1, int(args.Tminpre * T)) # This used to be 0.8 in previous training parts
    else :
        Tmin = max(1, int(args.Tmin * T))
    Tmax = int(0.98 * T)

    if epoch <= args.gan_pos :
        gan_coeffg = 0

    G_dp.train()
    G.train()
    mu_fake.train()
    loss_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)
        if args.data_augmentation:
            x = utils.random_rotation(x).detach()

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        bs_data, n_data, num_feat = x.shape

        # ================================================================
        # Encode real data once (no grad needed, mu_real is frozen).
        # x_xh: [B, N, n_dims + in_node_nf] concatenated real molecule.
        # ================================================================
        with torch.no_grad():
            x_xh = torch.cat([x, one_hot, charges if charges.dim() == 3
                               else charges.unsqueeze(2)], dim=2) if args.include_charges \
                else torch.cat([x, one_hot], dim=2)
            x_e = mu_real.encode(x_xh, node_mask, edge_mask, context)   # [B, N, latent_nf]
            # another choice is to use encode_to_latent_space
        latent_nf = x_e.shape[-1]

        # ================================================================
        # Sample timestep t ∈ [Tmin, Tmax] (normalised to [0,1]).
        # ================================================================
        t_int = torch.randint(Tmin, Tmax, (bs_data, 1), device=device).float()
        noise_t = t_int / T                                               # [B, 1]

        # ================================================================
        # GENERATOR UPDATE
        # 1. Sample z_fake from G (gradient flows back through here).
        # 2. Encode to latent, corrupt, compute scores.
        # 3. DMD loss + GAN generator loss → update G.
        # ================================================================
        if step_num == 1 :
            z_fake_e = G.one_step_sample_latent(bs_data, n_data, node_mask, edge_mask, context)
        else :
            # Select which step to backprop through BEFORE generating,
            # so only 1 step keeps its computation graph (saves ~(step_num-1)x GPU memory).
            if args.step_num_liftpos is not None :
                if epoch <= args.step_num_liftpos :
                    z_t_hat = torch.randint(args.step_num_large, step_num, (1,)).item()
                else :
                    z_t_hat = torch.randint(args.step_num_small, step_num, (1,)).item() 
            else :
                z_t_hat = torch.randint(step_low(args.start_epoch, epoch, 
                                        args.n_epochs, args.step_num_small, 
                                        args.step_num_large, args.step_num_pow), step_num, (1,)).item() 
            
            if args.t_coupling and z_t_hat <= args.step_num_large - 2 :
                noise_t[noise_t < args.t_coupling_coeff] = args.t_coupling_coeff

            z_fake_e = G.few_step_sample_latent(
                step_num, bs_data, n_data, node_mask, edge_mask, context, selected_step=z_t_hat)

        z_fake_t = mu_real.corrupt(noise_t, z_fake_e, bs_data, n_data, node_mask, edge_mask, context)

        z_fake_e_d = z_fake_e.detach()
        z_fake_t_d = z_fake_t.detach()
        x_e_d      = x_e.detach()

        # ================================================================
        # MU_FAKE + DISCRIMINATOR UPDATE  (step_ratio inner steps per 1 G step)
        # mu_fake is trained to denoise fake samples (diffusion loss).
        # D is trained to distinguish real from fake mu_fake features.
        # gan_coeff scales L_disc only; no separate mu_fake GAN term.
        # ================================================================

        discriminator.detach_hook = False   # Do not detach hook features during mu_fake/D update
        for _ in range(step_ratio):
            # mu_fake forward on fake z_t: hook captures fake bottleneck features + diffusion loss
            # in one forward pass (z0=z_fake_e_d recovers the noise used by corrupt()).
            L_fake_diffusion = mu_fake.score(noise_t, z_fake_t_d, bs_data, n_data,
                                                   node_mask, edge_mask, context, z_fake_e_d)
            L_fake_diffusion = soft_clamp(L_fake_diffusion)
            logit_D_fake = discriminator._forward(node_mask, edge_mask)     # log D(fake) [B]

            # mu_fake forward on real x_t → hook captures real bottleneck features
            x_t = mu_real.corrupt(noise_t, x_e_d, bs_data, n_data, node_mask, edge_mask, context)
            mu_fake.score(noise_t, x_t, bs_data, n_data, node_mask, edge_mask, context)
            logit_D_real = discriminator._forward(node_mask, edge_mask)     # log D(real) [B]

            # D loss: -log D(real) - log(1 - D(fake))   [gan_coeff scales the adversarial term]
            L_disc = F.softplus(-logit_D_real).mean() \
                    + F.softplus(logit_D_fake).mean()
            
            L_disc = soft_clamp(L_disc, 5)
            L_fake_diffusion = soft_clamp(L_fake_diffusion, 5)

            L_fake = L_fake_diffusion + gan_coefff * L_disc

            if torch.isnan(L_fake) or torch.isinf(L_fake):
                print(f'Warning: L_fake is {L_fake.item()}, skipping mu_fake update at iter {i}.')
                continue

            optim_fake.zero_grad()
            optim_d.zero_grad()
            L_fake.backward()
            torch.nn.utils.clip_grad_norm_(mu_fake.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optim_fake.step()
            optim_d.step()


        with torch.no_grad():
            s_real = mu_real.score(noise_t, z_fake_t, bs_data, n_data, node_mask, edge_mask, context)
        # mu_fake forward on z_fake_t: triggers hook → discriminator.mu_fake_out = fake features
        # Keep hook features on graph so L_gan_G gradient flows back to G.
        discriminator.detach_hook = False
        s_fake = mu_fake.score(noise_t, z_fake_t, bs_data, n_data, node_mask, edge_mask, context)

        # DMD loss: stop-grad on score difference, keep grad on z_fake_e (flows to G)
        latent_nf = s_fake.shape[-1]
        d_s = (s_fake - s_real).detach()
        L_dmd = (d_s * z_fake_t).sum(dim=[1, 2]).mean() / (latent_nf * n_data) # In loss of diffusion it is divided by a denom

        # Latent scale regularization: penalize z_fake_e variance mismatch with real latents
        L_reg = (z_fake_e.pow(2).sum(dim=[1, 2]).mean()
                 - x_e.detach().pow(2).sum(dim=[1, 2]).mean()).pow(2)

        # GAN generator loss: G wants D to classify fake as real → maximise log D(fake)
        logit_fake = discriminator._forward(node_mask, edge_mask)       # [B]
        L_gan_G = F.softplus(-logit_fake).mean()

        if args.clamp :
            L_dmd = soft_clamp(L_dmd, 10)
            L_gan_G = soft_clamp(L_gan_G, 10)

        weighting_factor = (z_fake_e - s_real).abs().mean(dim=[0, 1, 2], keepdim=True)

        L_G = L_dmd + gan_coeffg * L_gan_G + reg_coeff * L_reg
        L_G = L_G / weighting_factor

        if torch.any(torch.isnan(z_fake_e)) or torch.any(z_fake_e.abs() > 50):
            print(f"z_fake_e stats: min={z_fake_e.min():.2f}, max={z_fake_e.max():.2f}")
            print(f"z_fake_e coord range: {z_fake_e[:,:,:3].abs().max():.2f}")
            # check for collapsed atoms
            x_coords = z_fake_e[:, :, :3]  # [B, N, 3]
            dists = torch.cdist(x_coords, x_coords)  # [B, N, N]
            dists = dists + torch.eye(dists.shape[1], device=dists.device) * 1e6  # mask diagonal
            min_dist = dists.min()
            print(f"Min pairwise distance: {min_dist:.6f}")

        if torch.isnan(L_G) or torch.isinf(L_G):
            print(f'Warning: L_G is {L_G.item()}, skipping G update at iter {i}.')
        elif any(torch.isnan(p.grad).any() for p in mu_fake.dynamics.parameters() if p.grad is not None):
            print(f"NaN grad detected at iter {i}, skipping mu_fake update")
            continue  # skip this inner step
        else:
            optim_G.zero_grad()
            L_G.backward()
            if args.clip_grad:
                utils.gradient_clipping(G, gradnorm_queue)
            optim_G.step()

        # ================================================================
        # EMA update on G
        # ================================================================
        if ema is not None:
            ema.update_model_average(G_ema, G)

        loss_epoch.append(L_G.item())

        if i % args.n_report_steps == 0:
            with torch.no_grad():
                acc_real = (logit_D_real > 0).float().mean().item()
                acc_fake = (logit_D_fake < 0).float().mean().item()
                acc_d = (acc_real + acc_fake) / 2
            print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                  f"L_G: {L_G.item():.4f}, L_dmd: {L_dmd.item():.4f}, "
                  f"L_gan_G: {L_gan_G.item():.4f}, L_reg: {L_reg.item():.4f}, "
                  f"L_fake_diffusion: {L_fake_diffusion.item():.4f}, L_disc: {L_disc.item():.4f}, "
                  f"D_acc: {acc_d:.2f} (real:{acc_real:.2f}/fake:{acc_fake:.2f})")

        if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) \
                and not (epoch == 0 and i == 0) and args.train_diffusion:
            start = time.time()
            if len(args.conditioning) > 0:
                save_and_sample_conditional(args, device, G_ema, prop_dist, dataset_info, epoch=epoch)
            sample_different_sizes_and_save(G_ema, nodes_dist, args, device, dataset_info,
                                            prop_dist, epoch=epoch)
            print(f'Sampling took {time.time() - start:.2f} seconds')

            vis.visualize(f"outputs/{args.exp_name}/epoch_{epoch}_{i}", dataset_info=dataset_info, wandb=wandb)
            if len(args.conditioning) > 0:
                vis.visualize("outputs/%s/epoch_%d/conditional/" % (args.exp_name, epoch),
                                    dataset_info, wandb=wandb, mode='conditional')

        if args.break_train_epoch:
            break
    wandb.log({"Train Epoch Loss": np.mean(loss_epoch) if loss_epoch else float('nan')}, commit=True) # record the loss for every epoch


def encode_to_latent_space(model, x, h, node_mask, edge_mask, context):
    # Encode data to latent space.
    z_x_mu, z_x_sigma, z_h_mu, z_h_sigma = model.vae.encode(x, h, node_mask, edge_mask, context)
    # Compute fixed sigma values.
    t_zeros = torch.zeros(size=(x.size(0), 1), device=x.device)
    model.gamma.to('cuda')
    gamma_0 = model.inflate_batch_array(model.gamma(t_zeros), x)
    sigma_0 = model.sigma(gamma_0, x)

    # Infer latent z.
    z_xh_mean = torch.cat([z_x_mu, z_h_mu], dim=2)
    diffusion_utils.assert_correctly_masked(z_xh_mean, node_mask)
    z_xh_sigma = sigma_0
    z_xh = model.vae.sample_normal(z_xh_mean, z_xh_sigma, node_mask)
    z_xh = z_xh.detach()  # Always keep the encoder fixed.
    diffusion_utils.assert_correctly_masked(z_xh, node_mask)

    z_x = z_xh[:, :, :model.n_dims]
    z_h = z_xh[:, :, model.n_dims:]
    diffusion_utils.assert_mean_zero_with_mask(z_x, node_mask)
    # Make the data structure compatible with the EnVariationalDiffusion compute_loss().
    z_h = {'categorical': torch.zeros(0).to(z_h), 'integer': z_h}

    return z_x, z_h


def denoise_step(model, z_t, alpha_t, sigma_t, t, node_mask, edge_mask, context):
    """Tweedie denoising: x̂_0 = (z_t - σ_t·ε_pred) / α_t"""
    return (z_t / alpha_t) - (model.phi(z_t, t, node_mask, edge_mask, context) * (sigma_t / alpha_t))


def sample_one_step(args, device, model, dataset_info, prop_dist=None, nodesxsample=torch.tensor([10])):
    """One-step sampling for DMD-trained models via model.one_step_sample()."""
    max_n_nodes = dataset_info['max_n_nodes']
    assert int(torch.max(nodesxsample)) <= max_n_nodes
    batch_size = len(nodesxsample)

    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
    node_mask = node_mask.unsqueeze(2).to(device)

    if args.context_node_nf > 0:
        context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    with torch.no_grad():
        if args.step_num == 1 :
            xh = model.one_step_sample(batch_size, max_n_nodes, node_mask, edge_mask, context)
        else :
            xh = model.few_step_sample(args.step_num, batch_size, max_n_nodes, node_mask, edge_mask, context)

    # Split data-space xh using VAE dimensions (not latent-space num_classes).
    n_dims = model.vae.n_dims
    num_atom_types = model.vae.in_node_nf - int(model.vae.include_charges)
    x = xh[:, :, :n_dims]
    one_hot = xh[:, :, n_dims:n_dims + num_atom_types]
    charges = xh[:, :, n_dims + num_atom_types:]

    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    assert_correctly_masked(one_hot.float(), node_mask)

    return one_hot, charges, x, node_mask


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test'):
    from qm9 import losses
    eval_model.eval()
    with torch.no_grad():
        loss_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            if args.augment_noise > 0:
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            loss, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist, x, h,
                                                     node_mask, edge_mask, context)
            loss_epoch += loss.item() * batch_size
            n_samples += batch_size
            if i % args.n_report_steps == 0:
                print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"NLL: {loss_epoch/n_samples:.2f}")

    return loss_epoch / n_samples


def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)
    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, dataset_info, id_from, name='chain')
    return one_hot, charges, x


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample_one_step(args, device, model, dataset_info,
                                                         prop_dist=prop_dist,
                                                         nodesxsample=nodesxsample)
        print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/',
                          one_hot, charges, x, dataset_info, batch_size * counter, name='molecule')


def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=10, batch_size=100):
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    for i in range(int(n_samples / batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample_one_step(args, device, model_sample, dataset_info,
                                                         prop_dist=prop_dist,
                                                         nodesxsample=nodesxsample)
        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    wandb.log(validity_dict, commit=False)
    if rdkit_tuple is not None:
        wandb.log({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]}, commit=False)
    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)
    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch),
        one_hot, charges, x, dataset_info, id_from, name='conditional', node_mask=node_mask)
    return one_hot, charges, x

def soft_clamp(x, limit=15.0, temperature=1.0) :
    return limit * torch.tanh(x / (limit * temperature))

def step_low(start_epoch, cur_epoch, total_epoch, step_num_small, step_num_large, power=0.75) :
    """Progressively lower the z_t_hat lower bound from step_num_large to step_num_small.
    step_num_large = large lower bound (safe, used at start).
    step_num_small = small lower bound (aggressive, used at end).
    power > 1: stays at large longer (conservative); power < 1: drops to small faster."""
    epoch_train = total_epoch - start_epoch
    if epoch_train <= 0:
        return step_num_small
    progress = min((cur_epoch - start_epoch) / epoch_train, 1.0)
    result = step_num_large - (step_num_large - step_num_small) * (progress ** power)
    return max(int(result)+1, step_num_small)