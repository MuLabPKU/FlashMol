# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import utils
import argparse
from qm9 import dataset
from qm9.models import get_model, get_autoencoder, get_latent_diffusion
import os
from equivariant_diffusion.utils import assert_correctly_masked, assert_mean_zero_with_mask
import torch
import pickle
import qm9.visualizer as vis
from qm9.analyze import check_stability
from os.path import join
from configs.datasets_config import get_dataset_info
from qm9.utils import prepare_context, compute_mean_mad


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def sample_one_step(args, device, generative_model, dataset_info,
                    prop_dist=None, nodesxsample=torch.tensor([10]), context=None,
                    fix_noise=False):
    """One-step / few-step sampling for DMD-trained models."""
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
        if context is None:
            context = prop_dist.sample_batch(nodesxsample)
        context = context.unsqueeze(1).repeat(1, max_n_nodes, 1).to(device) * node_mask
    else:
        context = None

    step_num = getattr(args, 'step_num', 1)
    if step_num == 1:
        xh = generative_model.one_step_sample(
            batch_size, max_n_nodes, node_mask, edge_mask, context,
            fix_noise=fix_noise)
    else:
        xh = generative_model.few_step_sample(
            step_num, batch_size, max_n_nodes, node_mask, edge_mask, context,
            fix_noise=fix_noise)

    # Split data-space xh using VAE dimensions.
    n_dims = generative_model.vae.n_dims
    num_atom_types = generative_model.vae.in_node_nf - int(generative_model.vae.include_charges)
    x = xh[:, :, :n_dims]
    one_hot = xh[:, :, n_dims:n_dims + num_atom_types]
    charges = xh[:, :, n_dims + num_atom_types:]

    assert_correctly_masked(x, node_mask)
    assert_mean_zero_with_mask(x, node_mask)
    assert_correctly_masked(one_hot.float(), node_mask)
    if args.include_charges:
        assert_correctly_masked(charges.float(), node_mask)

    return one_hot, charges, x, node_mask


def sample_different_sizes_and_save(args, eval_args, device, generative_model,
                                    nodes_dist, prop_dist, dataset_info,
                                    n_samples=10):
    nodesxsample = nodes_dist.sample(n_samples)
    one_hot, charges, x, node_mask = sample_one_step(
        args, device, generative_model, dataset_info,
        prop_dist=prop_dist, nodesxsample=nodesxsample)

    vis.save_xyz_file(
        join(eval_args.model_path, 'eval/molecules/'), one_hot, charges, x,
        id_from=0, name='molecule', dataset_info=dataset_info,
        node_mask=node_mask)


def sample_only_stable_different_sizes_and_save(
        args, eval_args, device, generative_model, nodes_dist, prop_dist,
        dataset_info, n_samples=10, n_tries=50):
    assert n_tries > n_samples

    nodesxsample = nodes_dist.sample(n_tries)
    one_hot, charges, x, node_mask = sample_one_step(
        args, device, generative_model, dataset_info,
        prop_dist=prop_dist, nodesxsample=nodesxsample)

    counter = 0
    for i in range(n_tries):
        num_atoms = int(node_mask[i:i+1].sum().item())
        atom_type = one_hot[i:i+1, :num_atoms].argmax(2).squeeze(0).cpu().detach().numpy()
        x_squeeze = x[i:i+1, :num_atoms].squeeze(0).cpu().detach().numpy()
        mol_stable = check_stability(x_squeeze, atom_type, dataset_info)[0]

        num_remaining_attempts = n_tries - i - 1
        num_remaining_samples = n_samples - counter

        if mol_stable or num_remaining_attempts <= num_remaining_samples:
            if mol_stable:
                print('Found stable mol.')
            vis.save_xyz_file(
                join(eval_args.model_path, 'eval/molecules/'),
                one_hot[i:i+1], charges[i:i+1], x[i:i+1],
                id_from=counter, name='molecule_stable',
                dataset_info=dataset_info,
                node_mask=node_mask[i:i+1])
            counter += 1

            if counter >= n_samples:
                break


def save_and_sample_fixed_noise(args, eval_args, device, generative_model,
                                nodes_dist, prop_dist, dataset_info,
                                id_from=0, num_chains=100):
    """Generate molecules with fix_noise=True for visualization chains.

    With fixed noise, each molecule in the batch shares the same initial noise,
    so variation comes only from the node count — useful for visual comparison.
    """
    for i in range(num_chains):
        target_path = f'eval/chain_{i}/'

        nodesxsample = nodes_dist.sample(1)
        one_hot, charges, x, node_mask = sample_one_step(
            args, device, generative_model, dataset_info,
            prop_dist=prop_dist, nodesxsample=nodesxsample,
            fix_noise=True)

        vis.save_xyz_file(
            join(eval_args.model_path, target_path), one_hot, charges, x,
            dataset_info, id_from, name='chain', node_mask=node_mask)

        vis.visualize_chain_uncertainty(
            join(eval_args.model_path, target_path), dataset_info,
            spheres_3d=True)

    return one_hot, charges, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default="outputs/edm_1",
                        help='Specify model path')
    parser.add_argument(
        '--n_tries', type=int, default=10,
        help='N tries to find stable molecule for gif animation')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of atoms in molecule for gif animation')
    parser.add_argument('--epoch', type=int, default=-1,
                        help='Choose which epoch to test')
    parser.add_argument('--step_num', type=int, default=None,
                        help='Number of denoising steps (default: use value from checkpoint, or 1)')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    epoch_num = eval_args.epoch
    if epoch_num == -1:
        pickle_name = 'args.pickle'
    else:
        pickle_name = f'args_{epoch_num}.pickle'

    with open(join(eval_args.model_path, pickle_name), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    # Override step_num from command line (falls back to pickle value, then 1)
    args.step_num = eval_args.step_num if eval_args.step_num is not None else getattr(args, 'step_num', 1)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    generative_model, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    # Load checkpoint — same priority as eval_analyze.py
    if epoch_num == -1:
        candidates = ['G_ema.npy', 'G.npy', 'generative_model_ema.npy', 'generative_model.npy']
    else:
        candidates = [f'G_ema_{epoch_num}.npy', f'G_{epoch_num}.npy',
                      f'generative_model_ema_{epoch_num}.npy', f'generative_model_{epoch_num}.npy']

    fn = None
    for c in candidates:
        if os.path.exists(join(eval_args.model_path, c)):
            fn = c
            break
    if fn is None:
        raise FileNotFoundError(f"No model checkpoint found in {eval_args.model_path}. Tried: {candidates}")

    print(f"Loading model from: {fn}")
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    print('Sampling handful of molecules.')
    sample_different_sizes_and_save(
        args, eval_args, device, generative_model, nodes_dist, prop_dist,
        dataset_info=dataset_info, n_samples=30)

    print('Sampling stable molecules.')
    sample_only_stable_different_sizes_and_save(
        args, eval_args, device, generative_model, nodes_dist, prop_dist,
        dataset_info=dataset_info, n_samples=10, n_tries=2*10)

    print('Visualizing molecules.')
    vis.visualize(
        join(eval_args.model_path, 'eval/molecules/'), dataset_info,
        max_num=100, spheres_3d=True)

    print('Sampling visualization chains.')
    save_and_sample_fixed_noise(
        args, eval_args, device, generative_model, nodes_dist, prop_dist,
        dataset_info=dataset_info, num_chains=eval_args.n_tries)


if __name__ == "__main__":
    main()
