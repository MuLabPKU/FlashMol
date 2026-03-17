import torch
import torch.nn as nn
from egnn.egnn_new import GNN
from qm9.utils import get_adj_matrix
from equivariant_diffusion.en_diffusion import EnVariationalDiffusion


class MolecularDiscriminator(nn.Module):
    """
    The discriminator used in DMD-GAN loss.
    Each instance has an attach_to method that registers a forward hook on
    mu_fake_dynamics.egnn.embedding_out, capturing the pre-projection bottleneck
    features [B*N, hidden_nf] into self.mu_fake_out on every mu_fake forward pass.
    attach_to must be re-called after every load_state_dict (hooks are not in state_dict).
    """

    def __init__(self, in_node_nf, n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=2, attention=False,
                 normalization_factor=100, aggregation_method='sum'):
        super().__init__()
        self.mu_fake_out = None          # instance variable — not shared across instances
        self.detach_hook = True          # True → detach in mu_fake loop; False → keep grad for G loop
        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.n_layers = n_layers
        self.device = device

        self.gnn = GNN(in_node_nf, in_edge_nf=1, hidden_nf=hidden_nf,
                       aggregation_method=aggregation_method, device=device,
                       act_fn=act_fn, n_layers=n_layers, attention=attention,
                       normalization_factor=normalization_factor)

        self.mlp = nn.Sequential(
            nn.Linear(in_node_nf, in_node_nf),
            act_fn,
            nn.Linear(in_node_nf, 1))

        self.to(device)

    def _forward(self, node_mask, edge_mask):
        # node_mask: [B, N, 1],  edge_mask: [B, N*N, 1]
        bs, n_nodes, _ = node_mask.shape
        atom_num = node_mask.sum(dim=1, keepdim=True)           # [B, 1, 1]

        # mu_fake_out from hook is flat [B*N, in_node_nf] — reshape to [B, N, in_node_nf]
        h = self.mu_fake_out.view(bs, n_nodes, -1)

        assert h.shape[-1] == self.in_node_nf, (
            f"mu_fake bottleneck dim {h.shape[-1]} != discriminator in_node_nf {self.in_node_nf}")

        edges = get_adj_matrix(n_nodes, bs, self.device)
        edges = [e.to(self.device) for e in edges]

        node_mask_flat = node_mask.view(bs * n_nodes, 1)
        edge_mask_flat = edge_mask.view(bs * n_nodes * n_nodes, 1)

        h = h.view(bs * n_nodes, -1).clone() * node_mask_flat   # [B*N, in_node_nf]

        h = self.gnn(h, edges, edge_attr=edge_mask_flat,
                     node_mask=node_mask_flat, edge_mask=edge_mask_flat)
                                                                 # [B*N, in_node_nf]
        h = h.view(bs, n_nodes, -1)                             # [B, N, in_node_nf]
        h = h.sum(dim=1, keepdim=True) / atom_num               # [B, 1, in_node_nf]

        logits = self.mlp(h)                                     # [B, 1, 1]
        logits = logits.squeeze(-1).squeeze(-1)                  # [B]
        return logits

    def attach_to(self, mu_fake: EnVariationalDiffusion, hook_layer='embedding_out'):
        """Register a forward hook on a layer of mu_fake's EGNN.

        hook_layer: 'embedding_out' (default, captures input[0] before final projection)
                    or 'e_block_N' (captures output h from the N-th EquivariantBlock).
        """
        egnn = mu_fake.dynamics.egnn

        if hook_layer == 'embedding_out':
            target = egnn.embedding_out
            def _hook(_module, input, _output):
                if self.detach_hook:
                    self.mu_fake_out = input[0].detach()
                else:
                    self.mu_fake_out = input[0]
        elif hook_layer.startswith('e_block_'):
            target = getattr(egnn, hook_layer)
            def _hook(_module, _input, output):
                # EquivariantBlock.forward returns (h, x); capture h [B*N, hidden_nf]
                h = output[0]
                if self.detach_hook:
                    self.mu_fake_out = h.detach()
                else:
                    self.mu_fake_out = h
        else:
            raise ValueError(f"Unknown hook_layer '{hook_layer}'. Use 'embedding_out' or 'e_block_N'.")

        target.register_forward_hook(_hook)
