# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
from os.path import join, basename
from qm9 import dataset
from qm9.models import get_optim, get_model, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_dmd import train_epoch, test, analyze_and_save
from dmd.discriminator import MolecularDiscriminator

'''
python main_progdistill.py --n_epochs 30 --n_stability_samples 10 --diffusion_noise_schedule polynomial_2 
--diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 64 --nf 256
--n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 10 --ema_decay 0.9999 --train_diffusion
--latent_nf 2 --exp_name $student_name --teacher_path outputs/$teacher_model
'''

parser = argparse.ArgumentParser(description='DMDMolGen')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Teacher-student args
parser.add_argument('--teacher_path', type=str, default='outputs/qm9_latent2',
                    help='Path to teacher model')
parser.add_argument('--teacher_epoch', type=int, default=None,
                    help='Specific epoch to load teacher from (default: load best model)')
parser.add_argument('--student_path', type=str, default=None,
                    help='Path to a pre-trained student checkpoint to initialize G/G_ema from. '
                         'Ignored when --resume is set.')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

# GAN args
parser.add_argument('--gan_coeffg', type=float, default=0)
parser.add_argument('--gan_coefff', type=float, default=0.02)
parser.add_argument('--reg_coeff', type=float, default=0)
parser.add_argument('--step_ratio', type=int, default=5)
parser.add_argument('--step_num', type=int, default=10)

# VAE args
parser.add_argument('--latent_nf', type=int, default=4,
                    help='number of latent features')
parser.add_argument('--kl_weight', type=float, default=0.01,
                    help='weight of KL term in ELBO')

parser.add_argument('--model', type=str, default='egnn_dynamics',
                    help='our_dynamics | schnet | simple_dynamics | '
                         'kernel_dynamics | egnn_dynamics |gnn_dynamics')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--G_lr', type=float, default=2e-4)
parser.add_argument('--mu_fake_lr', type=float, default=2e-4)
parser.add_argument('--disc_lr', type=float, default=2e-4)
parser.add_argument('--tmin_liftpos', type=int, default=10)
parser.add_argument('--Tmin', type=float, default=0.2)
parser.add_argument('--step_num_div_small', type=int, default=4)
parser.add_argument('--step_num_div_large', type=int, default=2)
parser.add_argument('--step_num_liftpos', type=int, default=10000)
parser.add_argument('--gan_pos', type=int, default=7)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')
# EGNN args -->
parser.add_argument('--n_layers', type=int, default=6,
                    help='number of layers')
parser.add_argument('--inv_sublayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--nf', type=int, default=128,
                    help='number of layers')
parser.add_argument('--tanh', type=eval, default=True,
                    help='use tanh in the coord_mlp')
parser.add_argument('--attention', type=eval, default=True,
                    help='use attention in the EGNN')
parser.add_argument('--norm_constant', type=float, default=1,
                    help='diff/(|diff| + norm_constant)')
parser.add_argument('--sin_embedding', type=eval, default=False,
                    help='whether using or not the sin embedding')
# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=1)
parser.add_argument('--wandb_usr', type=str)
parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
parser.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=10)
parser.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=500,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--visualize_every_batch', type=int, default=1e8,
                    help="Can be used to visualize multiple times per epoch")
parser.add_argument('--normalization_factor', type=float, default=1,
                    help="Normalize the sum aggregation of EGNN")
parser.add_argument('--aggregation_method', type=str, default='sum',
                    help='"sum" or "mean"')
args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()
args.wandb_usr = utils.get_wandb_username(args.wandb_usr)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

# ============ RESUME CHECKPOINT LOADING ============
# Load this BEFORE teacher/model creation to get correct hyperparameters
if args.resume is not None:
    # Save command-line args that should override checkpoint
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr
    normalization_factor = args.normalization_factor
    aggregation_method = args.aggregation_method
    total_epoch = args.n_epochs
    test_epochs = args.test_epochs
    coeffg = args.gan_coeffg
    coefff = args.gan_coefff
    G_lr = args.G_lr
    mu_fake_lr = args.mu_fake_lr
    disc_lr = args.disc_lr
    gan_pos = args.gan_pos
    tmin_liftpos = args.tmin_liftpos
    step_num_div_large = args.step_num_div_large
    step_num_div_small = args.step_num_div_small
    step_num_liftpos = args.step_num_liftpos
    Tmin = args.Tmin

    # Save teacher_path if user wants to change teacher during resume
    teacher_path_override = args.teacher_path

    # Load checkpoint args
    try:
        with open(join(args.resume, 'args.pickle'), 'rb') as f:
            args = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Checkpoint directory '{args.resume}' does not contain 'args.pickle'. "
            f"Cannot resume training without saved arguments."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint args from '{args.resume}/args.pickle': {e}"
        )

    # Restore overrides
    args.resume = resume
    args.break_train_epoch = False
    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    args.n_epochs = total_epoch
    args.test_epochs = test_epochs
    args.gan_coefff = coefff
    args.gan_coeffg = coeffg
    args.G_lr = G_lr
    args.mu_fake_lr = mu_fake_lr
    args.disc_lr = disc_lr
    args.gan_pos = gan_pos
    args.tmin_liftpos = tmin_liftpos
    args.step_num_div_large = step_num_div_large
    args.step_num_div_small = step_num_div_small
    args.step_num_liftpos = step_num_liftpos
    args.Tmin = Tmin

    # Handle teacher_path: use override if provided, else use saved value
    if teacher_path_override is not None:
        args.teacher_path = teacher_path_override
        print(f"WARNING: Resuming with different teacher: {teacher_path_override}")
    # else: keep args.teacher_path from checkpoint

    # Backward compatibility
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    print(f"Resuming from: {resume}")
    print(f"Starting at epoch: {start_epoch}")
    print(args)

if args.teacher_path is not None:
    teacher_exp_name = basename(args.teacher_path)

    # Load teacher args (epoch-specific or best)
    if args.teacher_epoch is not None:
        teacher_args_file = join(args.teacher_path, f'args_{args.teacher_epoch}.pickle')
    else:
        teacher_args_file = join(args.teacher_path, 'args.pickle')

    with open(teacher_args_file, 'rb') as f:
        teacher_args = pickle.load(f)

    teacher_args.exp_name = teacher_exp_name

    # Careful with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = normalization_factor
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = aggregation_method

    #print(f" #### Teacher model: ####\n{args}")
    #print(teacher_args)

utils.create_folders(args)
# print(args)


# Wandb config
if args.no_wandb:
    mode = 'disabled'
else:
    mode = 'online' if args.online else 'offline'
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_dmd', 'config': args,
          'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
wandb.init(**kwargs)
wandb.save('*.txt')

# Retrieve QM9 dataloaders
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

data_dummy = next(iter(dataloaders['train']))

if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf

# Create Latent Diffusion Model or Autoencoder
# Initialize teacher as None (will be set if needed)
teacher = None

if args.train_diffusion:
    # If we have a teacher checkpoint, load its args first to get correct diffusion_steps
    if args.teacher_path is not None:
        teacher_exp_name = basename(args.teacher_path)

        # Determine which checkpoint files to load based on teacher_epoch
        if args.teacher_epoch is not None:
            # Load from specific epoch
            args_file = join(args.teacher_path, f'args_{args.teacher_epoch}.pickle')
            model_file = join(args.teacher_path, f'generative_model_ema_{args.teacher_epoch}.npy')
            print(f"Loading teacher from epoch {args.teacher_epoch}")
        else:
            # Load best model (no epoch number)
            args_file = join(args.teacher_path, 'args.pickle')
            model_file = join(args.teacher_path, 'generative_model_ema.npy')
            print("Loading teacher from best checkpoint")

        # Load teacher args
        try:
            with open(args_file, 'rb') as f:
                teacher_args = pickle.load(f)
                teacher_args.exp_name = teacher_exp_name
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Teacher args file not found: {args_file}\n"
                f"Make sure the teacher checkpoint exists at the specified epoch."
            )

        # Sync diffusion_steps from teacher so saved args.pickle matches model weights
        args.diffusion_steps = teacher_args.diffusion_steps

        # Create teacher with its ORIGINAL diffusion_steps from checkpoint
        teacher, nodes_dist, prop_dist = get_latent_diffusion(teacher_args, device, dataset_info, dataloaders['train'])

        # Load the checkpoint (now sizes will match)
        try:
            model_state_dict = torch.load(model_file)
            teacher.load_state_dict(model_state_dict)
            print(f"Successfully loaded teacher model from: {model_file}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Teacher model file not found: {model_file}\n"
                f"Make sure the teacher checkpoint exists at the specified epoch."
            )

        # mu_real is identical to the teacher
        mu_real = teacher
        mu_real.eval()
        for p in mu_real.parameters() :
            p.requires_grad_(False)

        # G starts from the teacher and only needs the dynamics (encoder/decoder are only used at inference)
        G = copy.deepcopy(teacher)
        G.train()
        for p in G.vae.parameters() :
            p.requires_grad_(False)
        for p in G.dynamics.parameters() :
            p.requires_grad_(True)

        # mu_fake only needs the dynamics since we train on latent space
        mu_fake = copy.deepcopy(teacher)
        mu_fake.train()
        for p in mu_fake.parameters():
            p.requires_grad_(True)

        # Discriminator: classification head tapping mu_fake's EGNN bottleneck.
        # in_node_nf must equal args.nf (mu_fake's EGNN hidden_nf — what the hook captures).
        # attach_to() registers a forward hook on mu_fake.egnn.embedding_out.
        discriminator = MolecularDiscriminator(
            in_node_nf=args.nf,  # must match mu_fake's EGNN hidden_nf
            n_dims=3,
            device=device)
        discriminator.attach_to(mu_fake)  # hook registration — must redo after resume

    else:
        # Progressive distillation requires a teacher model
        raise ValueError(
            "DMD requires a teacher model. "
            "Please specify --teacher_path with a path to a trained model checkpoint. "
            "If you want to train from scratch, use main_qm9.py instead."
        )
else:
    raise NotImplementedError("DMD only trains the dynamics")

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
mu_real = mu_real.to(device)
G = G.to(device)
mu_fake = mu_fake.to(device)

optim_G = torch.optim.AdamW(G.dynamics.parameters(), lr=args.G_lr, amsgrad=True, weight_decay=1e-12)
optim_fake = torch.optim.AdamW(mu_fake.dynamics.parameters(), lr=args.mu_fake_lr, amsgrad=True, weight_decay=1e-12)
optim_d = torch.optim.AdamW(discriminator.parameters(), lr=args.disc_lr, amsgrad=True, weight_decay=1e-12)

# print(model)

gradnorm_queue = utils.Queue()
# Start with conservative value to prevent early gradient explosion
# This will be flushed out after ~50 iterations
gradnorm_queue.add(10.0)  # Reduced from 3000


def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def compute_loss_and_nll(args, teacher_model, student_model, nodes_dist, x, h, node_mask, edge_mask, context):
    bs, n_nodes, n_dims = x.size()

    if args.probabilistic_model == 'diffusion':
        edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

        assert_correctly_masked(x, node_mask)

        # Here x is a position tensor, and h is a dictionary with keys
        # 'categorical' and 'integer'.
        nll = student_model (x, h, node_mask, edge_mask, context)

        N = node_mask.squeeze(2).sum(1).long()

        log_pN = nodes_dist.log_prob(N)

        assert nll.size() == log_pN.size()
        nll = nll - log_pN

        # Average over batch.
        nll = nll.mean(0)

        reg_term = torch.tensor([0.]).to(nll.device)
        mean_abs_z = 0.
    else:
        raise ValueError(args.probabilistic_model)

    return nll, reg_term, mean_abs_z


def save_dmd_checkpoint(args, epoch, G, G_ema, mu_fake, discriminator, optim_G, optim_fake, optim_d, suffix=''):
    """Save all DMD model states. suffix='' for best, suffix='_N' for periodic."""
    out = 'outputs/%s' % args.exp_name
    args.current_epoch = epoch + 1
    utils.save_model(G,                 f'{out}/G{suffix}.npy')
    utils.save_model(mu_fake,  f'{out}/mu_fake{suffix}.npy')
    utils.save_model(discriminator,     f'{out}/discriminator{suffix}.npy')
    utils.save_model(optim_G,           f'{out}/optim_G{suffix}.npy')
    utils.save_model(optim_fake,        f'{out}/optim_fake{suffix}.npy')
    utils.save_model(optim_d,           f'{out}/optim_d{suffix}.npy')
    if G_ema is not None:
        utils.save_model(G_ema,         f'{out}/G_ema{suffix}.npy')
    with open(f'{out}/args{suffix}.pickle', 'wb') as f:
        pickle.dump(args, f)

def main():
    # ============ LOAD MODEL STATES FROM CHECKPOINT ============
    if args.resume is not None:
        # --- G ---
        G_ema_path = join(args.resume, 'G_ema.npy')
        G_path     = join(args.resume, 'G.npy')
        try:
            G.load_state_dict(torch.load(G_ema_path, map_location=device))
            print(f"Loaded G from: {G_ema_path}")
        except FileNotFoundError:
            G.load_state_dict(torch.load(G_path, map_location=device))
            print(f"Loaded G from: {G_path}")

        # --- mu_fake ---
        mu_fake_path = join(args.resume, 'mu_fake.npy')
        try:
            mu_fake.load_state_dict(torch.load(mu_fake_path, map_location=device))
            print(f"Loaded mu_fake from: {mu_fake_path}")
        except FileNotFoundError:
            print(f"WARNING: {mu_fake_path} not found, mu_fake starts from teacher weights")

        # --- discriminator ---
        disc_path = join(args.resume, 'discriminator.npy')
        try:
            discriminator.load_state_dict(torch.load(disc_path, map_location=device))
            print(f"Loaded discriminator from: {disc_path}")
        except FileNotFoundError:
            print("WARNING: discriminator.npy not found, starting with fresh discriminator weights")
        # Hooks are not saved in state_dict — must re-register after every load
        discriminator.attach_to(mu_fake)

        # --- optimizers ---
        try:
            optim_G.load_state_dict(torch.load(join(args.resume, 'optim_G.npy'), map_location=device))
            print("Loaded optim_G state from checkpoint")
        except FileNotFoundError:
            print("WARNING: optim_G.npy not found, starting with fresh optimizer state")

        try:
            optim_fake.load_state_dict(torch.load(join(args.resume, 'optim_fake.npy'), map_location=device))
            print("Loaded optim_fake state from checkpoint")
        except FileNotFoundError:
            print("WARNING: optim_fake.npy not found, starting with fresh optimizer state")

        try:
            optim_d.load_state_dict(torch.load(join(args.resume, 'optim_d.npy'), map_location=device))
            print("Loaded optim_d state from checkpoint")
        except FileNotFoundError:
            print("WARNING: optim_d.npy not found, starting with fresh optimizer state")

        # Override learning rates from command line (load_state_dict restores old lr)
        for pg in optim_G.param_groups:
            pg['lr'] = args.G_lr
        for pg in optim_fake.param_groups:
            pg['lr'] = args.mu_fake_lr
        for pg in optim_d.param_groups:
            pg['lr'] = args.disc_lr
        print(f"Overriding lr: G_lr={args.G_lr}, mu_fake_lr={args.mu_fake_lr}, disc_lr={args.disc_lr}")

        print(f"Successfully resumed from epoch {args.start_epoch}")

    # Load G (and G_ema) from a pre-trained student checkpoint.
    # Only applies when NOT resuming — resume already loads G directly.
    if args.student_path is not None and args.resume is None:
        G_ema_path = join(args.student_path, 'G_ema.npy')
        G_path     = join(args.student_path, 'G.npy')
        try:
            G.load_state_dict(torch.load(G_ema_path, map_location=device))
            print(f"Loaded student G from: {G_ema_path}")
        except FileNotFoundError:
            try:
                G.load_state_dict(torch.load(G_path, map_location=device))
                print(f"Loaded student G from: {G_path}")
            except FileNotFoundError:
                print(f"WARNING: No G checkpoint found at {args.student_path}, "
                      f"keeping teacher initialization for G.")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        G_dp = torch.nn.DataParallel(G.cpu())
        G_dp = G_dp.cuda()
    else:
        G_dp = G

    # Initialize EMA over G only (mu_fake does not need EMA).
    if args.ema_decay > 0:
        G_ema = copy.deepcopy(G)
        ema   = flow_utils.EMA(args.ema_decay)

        if args.resume is not None:
            try:
                G_ema.load_state_dict(torch.load(join(args.resume, 'G_ema.npy'), map_location=device))
                print("Loaded G_ema state from checkpoint")
            except FileNotFoundError:
                print("WARNING: G_ema.npy not found, initialising EMA from current G weights")

        G_ema_dp = torch.nn.DataParallel(G_ema) if args.dp and torch.cuda.device_count() > 1 else G_ema
    else:
        ema    = None
        G_ema  = G
        G_ema_dp = G_dp

    if args.train_diffusion and teacher is None:
        raise RuntimeError(
            "Cannot start training: teacher model is None but train_diffusion=True. "
            "Please check teacher_path configuration."
        )

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch,
                    mu_real=mu_real, G=G, G_ema=G_ema, G_dp=G_dp,
                    mu_fake=mu_fake, discriminator=discriminator,
                    ema=ema, device=device, dtype=dtype,
                    property_norms=property_norms, nodes_dist=nodes_dist,
                    dataset_info=dataset_info, gradnorm_queue=gradnorm_queue,
                    optim_G=optim_G, optim_fake=optim_fake, optim_d=optim_d, prop_dist=prop_dist,
                    gan_coefff=args.gan_coefff, gan_coeffg=args.gan_coeffg, reg_coeff=args.reg_coeff, step_ratio=args.step_ratio, step_num=args.step_num)

        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            # Collect all test-epoch metrics into a single dict, then log once.
            epoch_metrics = {}

            if isinstance(G, en_diffusion.EnVariationalDiffusion):
                epoch_metrics.update(G.log_info())

            if not args.break_train_epoch and args.train_diffusion:
                analyze_and_save(args=args, epoch=epoch, model_sample=G_ema,
                                 nodes_dist=nodes_dist, dataset_info=dataset_info,
                                 device=device, prop_dist=prop_dist,
                                 n_samples=args.n_stability_samples)

            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch,
                           eval_model=G_ema_dp, partition='Val', device=device,
                           dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch,
                            eval_model=G_ema_dp, partition='Test', device=device,
                            dtype=dtype, nodes_dist=nodes_dist,
                            property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val  = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    # Best checkpoint — no epoch suffix
                    save_dmd_checkpoint(args, epoch, G, G_ema, mu_fake,
                                        discriminator, optim_G, optim_fake, optim_d, suffix='')

            # Periodic checkpoint — epoch-numbered
            if args.save_model:
                save_dmd_checkpoint(args, epoch, G, G_ema, mu_fake,
                                    discriminator, optim_G, optim_fake, optim_d, suffix=f'_{epoch}')
                print(f'Saved periodic checkpoint for epoch {epoch}')

            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            epoch_metrics.update({
                "Val loss": nll_val,
                "Test loss": nll_test,
                "Best cross-validated test loss": best_nll_test,
            })
            wandb.log(epoch_metrics, commit=True)


if __name__ == "__main__":
    main()
