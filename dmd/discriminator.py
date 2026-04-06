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

    def __init__(self, in_node_nf, n_dims, r1_weight=None, r1_sigma=None, device='cpu'):
        super().__init__()
        self.mu_fake_out = None          # instance variable — not shared across instances
        for i in ['0', '1', '2'] :
            setattr(self, f"mu_fake_out_{i}", None)
        self.r1_weight = r1_weight
        self.r1_sigma = r1_sigma
        self.detach_hook = True          # True → detach in mu_fake loop; False → keep grad for G loop
        self.n_dims = n_dims
        self.in_node_nf = in_node_nf
        self.device = device

        # Cross-attention pooling heads — one per hooked layer
        self.queries = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, in_node_nf) * 0.02)
            for _ in range(3)
        ])
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(in_node_nf, num_heads=4, batch_first=True)
            for _ in range(3)
        ])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_node_nf),
                nn.Linear(in_node_nf, in_node_nf),
                nn.SiLU(),
                nn.Linear(in_node_nf, 1)
            )
            for _ in range(3)
        ])

        self.to(device)

    def _forward(self, h1, h2, h3, node_mask, edge_mask):
        # Assume layer 2, 5, 7 is chosen
        # node_mask: [B, N, 1],  edge_mask: [B, N*N, 1]
        bs, n_nodes, _ = node_mask.shape
        atom_num = node_mask.sum(dim=1, keepdim=True)           # [B, 1, 1]

        # mu_fake_out from hook is flat [B*N, in_node_nf] — reshape to [B, N, in_node_nf]
        h1 = h1.view(bs, n_nodes, -1)
        h2 = h2.view(bs, n_nodes, -1)
        h3 = h3.view(bs, n_nodes, -1)
        
        key_padding_mask = ~node_mask.squeeze(-1).bool()  # [B, N], True = ignore
        
        logits = []

        for h, query, attn, mlp in zip(
            [h1, h2, h3], self.queries, self.attns, self.mlps
        ):
            assert h.shape[-1] == self.in_node_nf, (
                f"mu_fake bottleneck dim {h.shape[-1]} != discriminator in_node_nf {self.in_node_nf}")

            q = query.expand(bs, -1, -1)            # [B, 1, in_node_nf]
            pooled, _ = attn(
                query=q, key=h, value=h,
                key_padding_mask=key_padding_mask
            )                                         # [B, 1, in_node_nf]
            logits.append(mlp(pooled).squeeze(-1).squeeze(-1))  # [B]
            
        logits = torch.stack(logits)

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
            block_num = hook_layer[-1]
            def _hook(_module, _input, output):
                # EquivariantBlock.forward returns (h, x); capture h [B*N, hidden_nf]
                h = output[0]
                if self.detach_hook:
                    setattr(self, f"mu_fake_out_{block_num}", h.detach())
                else:
                    setattr(self, f"mu_fake_out_{block_num}", h)
        else:
            raise ValueError(f"Unknown hook_layer '{hook_layer}'. Use 'embedding_out' or 'e_block_N'.")

        target.register_forward_hook(_hook)

    def r1_loss(self, real_logits, node_mask, edge_mask) :

        bs_data, n_nodes, _ = node_mask.shape

        fwd_args = []

        for i in ['0', '1', '2'] :
            h = getattr(self, f"mu_fake_out_{i}")
            epsilon = self.r1_sigma * torch.randn_like(h) 
            # Note that the noise needs not be masked 
            # since padding positions will be ignored in multi-head attention
            noise_h = h + epsilon
            fwd_args.append(noise_h)
        
        fwd_args.append(node_mask)
        fwd_args.append(edge_mask)

        logits = self._forward(*fwd_args)

        loss = (real_logits - logits) / self.r1_sigma
        loss = self.r1_weight * torch.mean(loss ** 2)

        return loss

        
