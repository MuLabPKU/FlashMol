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
from train_progdistill import train_epoch, test, analyze_and_save

'''
python main_progdistill.py --n_epochs 30 --n_stability_samples 10 --diffusion_noise_schedule polynomial_2 
--diffusion_noise_precision 1e-5 --diffusion_steps 500 --diffusion_loss_type l2 --batch_size 64 --nf 256 
--n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 10 --ema_decay 0.9999 --train_diffusion
--latent_nf 2 --exp_name $student_name --teacher_path outputs/$teacher_model
'''

parser = argparse.ArgumentParser(description='ProgDistillatsion')
parser.add_argument('--exp_name', type=str, default='debug_10')

# Teacher-student args
parser.add_argument('--teacher_path', type=str, default='outputs/qm9_latent2',
                    help='Path to teacher model')
parser.add_argument('--teacher_epoch', type=int, default=None,
                    help='Specific epoch to load teacher from (default: load best model)')

# Latent Diffusion args
parser.add_argument('--train_diffusion', action='store_true', 
                    help='Train second stage LatentDiffusionModel model')
parser.add_argument('--ae_path', type=str, default=None,
                    help='Specify first stage model path')
parser.add_argument('--trainable_ae', action='store_true',
                    help='Train first stage AutoEncoder model')

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
parser.add_argument('--diffusion_steps', type=int, default=500, help='student difusion steps')
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
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
kwargs = {'entity': args.wandb_usr, 'name': args.exp_name, 'project': 'e3_progdistill_diffusion_qm9_new', 'config': args,
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

N = args.diffusion_steps

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

        # Student model starts from the teacher model
        model = copy.deepcopy(teacher)

        # Change noising schedule to have half the steps (student uses N steps)
        model.T = N
        model.gamma = en_diffusion.PredefinedNoiseSchedule(
            args.diffusion_noise_schedule,
            N,
            args.diffusion_noise_precision,
        )
    else:
        # Progressive distillation requires a teacher model
        raise ValueError(
            "Progressive distillation requires a teacher model. "
            "Please specify --teacher_path with a path to a trained model checkpoint. "
            "If you want to train from scratch, use main_qm9.py instead."
        )
else:
    # Training autoencoder only (no diffusion)
    model, nodes_dist, prop_dist = get_autoencoder(args, device, dataset_info, dataloaders['train'])

if prop_dist is not None:
    prop_dist.set_normalizer(property_norms)
model = model.to(device)
optim = get_optim(args, model)
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


def main():
    # ============ LOAD MODEL STATES FROM CHECKPOINT ============
    if args.resume is not None:
        # Try to load student model state (prefer EMA version for best checkpoint)
        try:
            student_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'))
            print(f"Loaded student model from: {join(args.resume, 'generative_model_ema.npy')}")
        except FileNotFoundError:
            print("WARNING: generative_model_ema.npy not found, trying generative_model.npy")
            student_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
            print(f"Loaded student model from: {join(args.resume, 'generative_model.npy')}")

        model.load_state_dict(student_state_dict)

        # Load optimizer state
        try:
            optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
            optim.load_state_dict(optim_state_dict)
            print("Loaded optimizer state from checkpoint")
        except FileNotFoundError:
            print("WARNING: optim.npy not found, starting with fresh optimizer state")
            print("This may cause training instability. Consider using the checkpoint with optimizer state.")

        print(f"Successfully loaded checkpoint from epoch {args.start_epoch}")

    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs')
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        # Load EMA state if resuming
        if args.resume is not None:
            try:
                ema_state_dict = torch.load(join(args.resume, 'generative_model_ema.npy'))
                model_ema.load_state_dict(ema_state_dict)
                print("Loaded EMA model state from checkpoint")
            except FileNotFoundError:
                print("WARNING: Could not load EMA state, will use current model state")

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    # Validate that teacher exists for progressive distillation training
    if args.train_diffusion and teacher is None:
        raise RuntimeError(
            "Cannot start training: teacher model is None but train_diffusion=True. "
            "This should not happen. Please check teacher_path configuration."
        )

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, teacher=teacher, model=model,
                    model_ema=model_ema, model_dp=model_dp, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist)
        
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:
            if isinstance(model, en_diffusion.EnVariationalDiffusion):
                wandb.log(model.log_info(), commit=True)

            # # Skip sampling at epoch 0 to avoid NaN issues with untrained student model
            # if not args.break_train_epoch and args.train_diffusion and epoch > 0:
            if not args.break_train_epoch and args.train_diffusion :
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms)
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms)

            if nll_val < best_nll_val:
                best_nll_val = nll_val
                best_nll_test = nll_test
                if args.save_model:
                    args.current_epoch = epoch + 1
                    # Save best model with standard names (no epoch number)
                    utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                    utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                    if args.ema_decay > 0:
                        utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                    with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                        pickle.dump(args, f)

            # Save periodic checkpoint after every test epoch (regardless of validation performance)
            if args.save_model:
                args.current_epoch = epoch + 1
                # Save epoch-numbered checkpoint for this test epoch
                utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (args.exp_name, epoch))
                utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (args.exp_name, epoch))
                if args.ema_decay > 0:
                    utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (args.exp_name, epoch))
                with open('outputs/%s/args_%d.pickle' % (args.exp_name, epoch), 'wb') as f:
                    pickle.dump(args, f)
                print(f'Saved periodic checkpoint for epoch {epoch}')

            print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
            print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))
            wandb.log({"Val loss ": nll_val}, commit=True)
            wandb.log({"Test loss ": nll_test}, commit=True)
            wandb.log({"Best cross-validated test loss ": best_nll_test}, commit=True)


if __name__ == "__main__":
    main()
