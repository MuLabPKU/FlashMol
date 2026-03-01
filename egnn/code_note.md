Claude Session: claude --resume 82d21e76-0dcd-4adb-9902-a6da7f0054b8 in tmux DMDCOde session
---
  Method 1: __init__ (Constructor) - Lines 6-28

  Required Parameters

  ┌──────────────────────┬───────┬─────────────────┬───────────────────────────────────────────────────────────┐
  │      Parameter       │ Type  │     Example     │                          Purpose                          │
  ├──────────────────────┼───────┼─────────────────┼───────────────────────────────────────────────────────────┤
  │ input_nf             │ int   │ 5               │ Input node feature dimension (e.g., 5 atom properties)    │
  ├──────────────────────┼───────┼─────────────────┼───────────────────────────────────────────────────────────┤
  │ output_nf            │ int   │ 64              │ Output node feature dimension after transformation        │
  ├──────────────────────┼───────┼─────────────────┼───────────────────────────────────────────────────────────┤
  │ hidden_nf            │ int   │ 128             │ Hidden layer dimension in internal MLPs                   │
  ├──────────────────────┼───────┼─────────────────┼───────────────────────────────────────────────────────────┤
  │ normalization_factor │ float │ 100             │ Divides aggregated messages to prevent gradient explosion │
  ├──────────────────────┼───────┼─────────────────┼───────────────────────────────────────────────────────────┤
  │ aggregation_method   │ str   │ 'sum' or 'mean' │ How to combine edge messages at each node                 │
  └──────────────────────┴───────┴─────────────────┴───────────────────────────────────────────────────────────┘

  Optional Parameters

  ┌───────────────┬───────────┬───────────┬──────────────────────────────────────────────────────────────┐
  │   Parameter   │  Default  │  Example  │                           Purpose                            │
  ├───────────────┼───────────┼───────────┼──────────────────────────────────────────────────────────────┤
  │ edges_in_d    │ 0         │ 3         │ Edge feature dimension (e.g., bond type, order, aromaticity) │
  ├───────────────┼───────────┼───────────┼──────────────────────────────────────────────────────────────┤
  │ nodes_att_dim │ 0         │ 2         │ Auxiliary node attributes (e.g., pH, temperature)            │
  ├───────────────┼───────────┼───────────┼──────────────────────────────────────────────────────────────┤
  │ act_fn        │ nn.SiLU() │ nn.ReLU() │ Activation function for MLPs                                 │
  ├───────────────┼───────────┼───────────┼──────────────────────────────────────────────────────────────┤
  │ attention     │ False     │ True      │ Whether to weight edge messages by learned attention         │
  └───────────────┴───────────┴───────────┴──────────────────────────────────────────────────────────────┘

  Example Instantiation

  # For molecular graph with rich features
  gcl = GCL(
      input_nf=5,              # 5 atom features per node
      output_nf=64,            # Output: 64-dim learned representations
      hidden_nf=128,           # 128-dim hidden layers
      normalization_factor=100,# Normalize aggregated messages
      aggregation_method='sum',# Sum-then-normalize
      edges_in_d=3,            # 3 bond features per edge
      attention=True           # Use attention weighting
  )

  ---
  Method 2: edge_model - Lines 30-45

  Computes messages for each edge by processing connected node pairs.

  ┌───────────┬──────────┬─────────────────────────┬──────────────────────────────────────────────────────────┐
  │ Parameter │ Required │          Shape          │                       Description                        │
  ├───────────┼──────────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
  │ source    │ ✓        │ [num_edges, input_nf]   │ Features of source nodes (obtained via h[edge_index[0]]) │
  ├───────────┼──────────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
  │ target    │ ✓        │ [num_edges, input_nf]   │ Features of target nodes (obtained via h[edge_index[1]]) │
  ├───────────┼──────────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
  │ edge_attr │ ✗        │ [num_edges, edges_in_d] │ Optional edge features (e.g., bond types)                │
  ├───────────┼──────────┼─────────────────────────┼──────────────────────────────────────────────────────────┤
  │ edge_mask │ ✗        │ [num_edges, 1]          │ Binary mask to zero out edges (for batching)             │
  └───────────┴──────────┴─────────────────────────┴──────────────────────────────────────────────────────────┘

  Returns:
  - out: Processed edge messages [num_edges, hidden_nf]
  - mij: Pre-attention messages (for analysis)

  Example

  # Molecule with 24 atoms, 26 bonds
  source = h[edge_index[0]]  # [26, 5] - "sending" atom features
  target = h[edge_index[1]]  # [26, 5] - "receiving" atom features
  edge_attr = bond_features  # [26, 3] - bond type, order, aromatic

  edge_messages, mij = gcl.edge_model(source, target, edge_attr, edge_mask=None)
  # edge_messages: [26, 128] - one message vector per bond

  ---
  Method 3: node_model - Lines 47-57

  Aggregates edge messages and updates node features.

  ┌────────────┬──────────┬────────────────────────────┬──────────────────────────────────────────────────────┐
  │ Parameter  │ Required │           Shape            │                     Description                      │
  ├────────────┼──────────┼────────────────────────────┼──────────────────────────────────────────────────────┤
  │ x          │ ✓        │ [num_nodes, input_nf]      │ Current node features (before update)                │
  ├────────────┼──────────┼────────────────────────────┼──────────────────────────────────────────────────────┤
  │ edge_index │ ✓        │ [2, num_edges]             │ Graph connectivity: row 0 = sources, row 1 = targets │
  ├────────────┼──────────┼────────────────────────────┼──────────────────────────────────────────────────────┤
  │ edge_attr  │ ✓        │ [num_edges, hidden_nf]     │ Processed edge messages from edge_model              │
  ├────────────┼──────────┼────────────────────────────┼──────────────────────────────────────────────────────┤
  │ node_attr  │ ✗        │ [num_nodes, nodes_att_dim] │ Optional auxiliary node attributes                   │
  └────────────┴──────────┴────────────────────────────┴──────────────────────────────────────────────────────┘

  Important: The edge_attr here is the OUTPUT of edge_model, not raw edge features!

  Returns:
  - out: Updated node features [num_nodes, output_nf]
  - agg: Pre-MLP concatenated features (for debugging)

  Example

  # After edge_model produced edge_messages [26, 128]
  h_updated, agg = gcl.node_model(
      x=h,                      # [24, 5] - current atom features
      edge_index=bonds,         # [2, 26] - bond connectivity
      edge_attr=edge_messages,  # [26, 128] - from edge_model!
      node_attr=None
  )
  # h_updated: [24, 64] - new atom representations

  ---
  Method 4: forward (Main Entry Point) - Lines 59-65

  End-to-end graph convolution combining edge processing and node updates.

  ┌────────────┬──────────┬────────────────────────────┬────────────────────────────────────────────┐
  │ Parameter  │ Required │           Shape            │                Description                 │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ h          │ ✓        │ [num_nodes, input_nf]      │ Node feature matrix                        │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ edge_index │ ✓        │ [2, num_edges]             │ Graph connectivity (COO format)            │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ edge_attr  │ ✗        │ [num_edges, edges_in_d]    │ Raw edge features (NOT processed messages) │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ node_attr  │ ✗        │ [num_nodes, nodes_att_dim] │ Auxiliary node attributes                  │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ node_mask  │ ✗        │ [num_nodes, 1]             │ Binary mask for nodes (for batching)       │
  ├────────────┼──────────┼────────────────────────────┼────────────────────────────────────────────┤
  │ edge_mask  │ ✗        │ [num_edges, 1]             │ Binary mask for edges (for batching)       │
  └────────────┴──────────┴────────────────────────────┴────────────────────────────────────────────┘

  Returns:
  - h: Updated node features [num_nodes, output_nf]
  - mij: Pre-attention edge messages

  Complete Example

  # Caffeine molecule: 24 atoms, 26 bonds
  h = torch.randn(24, 5)           # Initial atom features
  edge_index = torch.tensor([      # Bond connectivity
      [0, 0, 1, 1, 2, ...],        # Source atoms
      [1, 3, 0, 2, 1, ...]         # Target atoms
  ])  # Shape: [2, 26]
  bond_features = torch.randn(26, 3)  # Bond type, order, aromatic

  # Forward pass
  h_new, messages = gcl.forward(
      h=h,                    # [24, 5]
      edge_index=edge_index,  # [2, 26]
      edge_attr=bond_features,# [26, 3] - RAW features
      node_mask=None,         # All atoms active
      edge_mask=None          # All bonds active
  )
  # h_new: [24, 64] - updated atom representations

  ---
  Key Distinctions to Remember

  1. Two Types of edge_attr

  - In forward(): Raw edge features [num_edges, edges_in_d]
  - In node_model(): Processed edge messages [num_edges, hidden_nf] from edge_model

  2. Edge Index Format

  edge_index = [[source_nodes], [target_nodes]]  # Shape: [2, num_edges]

  # Example: Bond between atom 0 and atom 1
  edge_index = [[0, 1],  # For undirected graph, need both directions
                [1, 0]]

  3. Masking for Batching

  When processing multiple molecules simultaneously:
  # Batch of 3 molecules: [10, 15, 8] atoms → total 33 nodes
  # If padding to 20 nodes each → 60 total nodes

  node_mask = torch.cat([
      torch.ones(10, 1),   # Molecule 1: all real
      torch.ones(15, 1),   # Molecule 2: all real
      torch.ones(8, 1),    # Molecule 3: all real
      torch.zeros(27, 1)   # Padding: masked out
  ])  # Shape: [60, 1]

  ---
  Quick Reference: Shape Flow

  Example: 24 atoms, 26 bonds, 5 input features, 128 hidden, 64 output

  forward() input:
  ├─ h: [24, 5]
  ├─ edge_index: [2, 26]
  └─ edge_attr: [26, 3]

    ↓ edge_model()

  ├─ source: [26, 5]
  ├─ target: [26, 5]
  ├─ concat: [26, 13]  (5+5+3)
  ├─ edge_mlp: [26, 128]
  └─ edge_messages: [26, 128]

    ↓ node_model()

  ├─ aggregate: [24, 128]  (sum over edges per node)
  ├─ concat: [24, 133]  (5+128)
  ├─ node_mlp: [24, 64]
  ├─ residual: [24, 64]  (5 → 64 via node_mlp output)
  └─ h_new: [24, 64]

  forward() output:
  ├─ h: [24, 64]  ← Updated node features
  └─ mij: [26, 128]  ← Edge messages

  ---
  Common Issues & Solutions

  ❌ Problem: Dimension mismatch in residual connection

  # Error: Can't add [24, 5] + [24, 64]
  Solution: For residual to work, ensure input_nf == output_nf OR the node_mlp projects to correct size

  ❌ Problem: Wrong edge_attr passed to node_model

  # BAD: Passing raw edge features
  h, _ = gcl.node_model(h, edge_index, raw_bond_features, None)
  Solution: Always pass processed messages from edge_model:
  edge_messages, _ = gcl.edge_model(h[row], h[col], raw_bond_features, None)
  h, _ = gcl.node_model(h, edge_index, edge_messages, None)  # ✓

  ❌ Problem: Forgetting bidirectional edges for undirected graphs

  # BAD: Only one direction per bond
  edge_index = [[0, 1, 2], [1, 2, 3]]  # 3 edges
  Solution: Add reverse edges:
  edge_index = [[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]  # 6 edges (3 bidirectional)

  ---

  ---
# The Use of coord_range:
---                                                                                                                            
  What It Does                                                                                                                 
                                                                                                                                 
  coords_range is a scaling factor that bounds the magnitude of coordinate updates when using tanh activation. It only affects   
  the equation when tanh=True (line 90):                                                                                         

  if self.tanh:
      trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
      #                                                                  ↑ This!
  else:
      trans = coord_diff * self.coord_mlp(input_tensor)  # No coords_range

  ---
  Mathematical Effect

  Coordinate update formula:
  Δx_i = Σ_j (normalized_direction) × tanh(MLP_output) × coords_range
         ↑                            ↑                    ↑
     unit vector                   bounded to [-1,1]    rescale

  Key insight: tanh bounds the MLP output to [-1, 1], then coords_range rescales it:

  ┌────────────┬──────────────┬───────────────────────┬──────────────┐
  │ MLP Output │ tanh(output) │ × coords_range (10.0) │ Max movement │
  ├────────────┼──────────────┼───────────────────────┼──────────────┤
  │ -1000      │ -1.0         │ -10.0                 │ 10 Å         │
  ├────────────┼──────────────┼───────────────────────┼──────────────┤
  │ -5         │ -0.9999      │ -9.999                │ ~10 Å        │
  ├────────────┼──────────────┼───────────────────────┼──────────────┤
  │ 0          │ 0.0          │ 0.0                   │ 0 Å          │
  ├────────────┼──────────────┼───────────────────────┼──────────────┤
  │ 5          │ 0.9999       │ 9.999                 │ ~10 Å        │
  ├────────────┼──────────────┼───────────────────────┼──────────────┤
  │ 1000       │ 1.0          │ 10.0                  │ 10 Å         │
  └────────────┴──────────────┴───────────────────────┴──────────────┘

  No matter how large the MLP output, per-edge updates are bounded by coords_range.

  ---
  Why It's Needed

  Problem Without Bounds

  Without tanh + coords_range:
  - ❌ MLP might predict 100 Å movements in early training
  - ❌ Molecules "explode" to unphysical configurations
  - ❌ Numerical instability and NaNs
  - ❌ Cannot learn reasonable molecular geometries

  Solution With Bounds

  With tanh + coords_range=15.0:
  - ✅ Maximum atomic displacement: 15 Å per layer
  - ✅ Stable training from random initialization
  - ✅ Physically reasonable denoising steps
  - ✅ Network learns when to use full budget vs. small corrections

  ---
  Typical Values

  From the codebase:

  ┌──────────────────────────────────┬───────────────┬────────────────────────────────┐
  │             Context              │ Default Value │           Reasoning            │
  ├──────────────────────────────────┼───────────────┼────────────────────────────────┤
  │ EquivariantUpdate (single layer) │ 10.0 Å        │ Conservative per-layer budget  │
  ├──────────────────────────────────┼───────────────┼────────────────────────────────┤
  │ EquivariantBlock                 │ 15.0 Å        │ Block-level default            │
  ├──────────────────────────────────┼───────────────┼────────────────────────────────┤
  │ EGNN (QM9 generation)            │ 15.0 Å        │ Total budget across all layers │
  ├──────────────────────────────────┼───────────────┼────────────────────────────────┤
  │ Property prediction              │ 3.0 Å         │ Smaller updates for stability  │
  └──────────────────────────────────┴───────────────┴────────────────────────────────┘

  For QM9 molecules:
  - Typical molecular radius: ~6 Å
  - coords_range = 15.0 ≈ 2.5× radius
  - With 6 layers: ~2.5 Å per layer

  ---
  Division by n_layers (CRITICAL!)

  From line 160 in EGNN.__init__:
  self.coords_range_layer = float(coords_range/n_layers)

  This implements a budget allocation strategy:

  Example: coords_range=15.0, n_layers=6

  Per-layer budget: 15.0 / 6 = 2.5 Å

  Layer 1: max update = 2.5 Å  (rough positioning)
  Layer 2: max update = 2.5 Å  (refine backbone)
  Layer 3: max update = 2.5 Å  (place heteroatoms)
  Layer 4: max update = 2.5 Å  (adjust hydrogens)
  Layer 5: max update = 2.5 Å  (bond lengths)
  Layer 6: max update = 2.5 Å  (final angles)
  ────────────────────────────────────────────
  Theoretical max: 6 × 2.5 = 15.0 Å total

  Why Divide?

  Without division:
  - 6 layers × 10 Å/layer = 60 Å max update ❌ (too much!)

  With division:
  - 6 layers × (10/6) Å/layer = 10 Å max update ✅ (reasonable)

  Benefits:
  1. Consistent behavior: 3-layer vs 6-layer models have similar total capacity
  2. Prevents growth with depth: Updates don't scale linearly with layer count
  3. Compositional refinement: Early layers make rough adjustments, later layers fine-tune

  ---
  Physical Interpretation (Molecular Generation)

  In diffusion-based molecule generation:

  Generation Trajectory Example: Aspirin (21 atoms)

  t=1.0 (High noise):
  ├─ Atoms scattered randomly in 20 Å sphere
  ├─ Network: "Move all toward center"
  └─ Update: ~10 Å (uses most of budget)

  t=0.5 (Medium noise):
  ├─ Rough molecular shape formed
  ├─ Network: "Make benzene ring planar"
  └─ Update: ~2 Å (moderate use)

  t=0.1 (Low noise):
  ├─ Almost correct structure
  ├─ Network: "Fine-tune bond lengths"
  └─ Update: ~0.3 Å (small correction)

  t=0.0 (Final):
  ├─ Clean molecular structure
  └─ Update: ~0.05 Å (minimal)

  Role of coords_range:
  - Matches physical scales (bonds ~1.5 Å, conformations ~5 Å)
  - Network learns when to use full budget vs. refinements
  - Prevents unphysical jumps during denoising

  ---
  Does It Affect Equivariance?

  No, it preserves E(3) equivariance. ✅

  Why?

  coords_range is a scalar constant that scales all coordinates uniformly:

  Under rotation R:
  Δx'_i = R × (coord_diff × tanh(...) × coords_range)
        = R × Δx_i

  coords_range commutes with rotation → equivariance preserved

  It's like multiplying by any scalar—doesn't break rotational or translational symmetry.

  ---
  Quick Reference

  Choosing coords_range

  Increase (20-30 Å) if:
  - Large molecules (>50 atoms)
  - Training plateaus ("stuck" structures)
  - Polymers or large conformational changes

  Decrease (5-10 Å) if:
  - Training unstable (NaNs, explosions)
  - Small molecules (<10 atoms)
  - Want fine-grained control

  Rule of thumb:
  coords_range ≈ 2 × max_molecular_radius

  Debugging Checklist

  Seeing NaNs or exploding molecules?
  1. ✅ Verify tanh=True
  2. ✅ Reduce coords_range to 5-10
  3. ✅ Check MLP initialization (line 76: gain=0.001)

  Training too slow or structures stuck?
  1. ✅ Increase coords_range to 20-30
  2. ✅ Check gradient flow to later layers

  ---
  Summary

  ┌──────────────────┬────────────────────────────────────────────────────────┐
  │      Aspect      │                         Value                          │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Purpose          │ Bound per-layer coordinate updates for stability       │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Active when      │ tanh=True (standard for generation)                    │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Default (QM9)    │ 15.0 Å total, ~2.5 Å per layer (6 layers)              │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Effect           │ Max atomic movement = ±coords_range/n_layers per layer │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Equivariance     │ ✅ Preserved (scalar scaling)                          │
  ├──────────────────┼────────────────────────────────────────────────────────┤
  │ Physical meaning │ Denoising step size in molecular generation            │
  └──────────────────┴────────────────────────────────────────────────────────┘

  ---

EGNN vs GNN: Complete Comparison & Input Reference                                                                             
                                                                                                                               
  The code analyst has provided a comprehensive comparison. Here's the organized breakdown:                                      
                                                                                                                                 
  ---                                                                                                                            
  Part 1: Core Differences                                                                                                       
                                                                                                                                 
  Architecture                                                                                                                   
                                                                                                                                 
  ┌─────────────────────┬───────────────────────────────────────────────────────┬──────────────────────────┐
  │       Feature       │                         EGNN                          │           GNN            │
  ├─────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────┤
  │ What it processes   │ Node features h + 3D coordinates x                    │ Node features h only     │
  ├─────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────┤
  │ Geometric awareness │ ✅ E(n)-equivariant (respects rotations/translations) │ ❌ No geometry awareness │
  ├─────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────┤
  │ Internal blocks     │ EquivariantBlock (GCL + coordinate updates)           │ GCL only                 │
  ├─────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────┤
  │ Outputs             │ Updated features h + refined coordinates x            │ Updated features h       │
  └─────────────────────┴───────────────────────────────────────────────────────┴──────────────────────────┘

  When to Use Each

  Use EGNN for:
  - 🧬 Molecular conformation generation
  - 🔬 3D structure prediction
  - ⚛️  Molecular dynamics
  - 🏗️  Protein folding
  - 📍 Any task where 3D geometry matters

  Use GNN for:
  - 📊 Property prediction (solubility, toxicity)
  - 🏷️  Graph classification
  - 📈 QSAR modeling from 2D graphs
  - 🔗 Link prediction
  - 📝 Any task with abstract graphs (no geometry)

  Performance

  ┌────────────┬───────────────────────────────────┬───────────────────────┐
  │   Metric   │               EGNN                │          GNN          │
  ├────────────┼───────────────────────────────────┼───────────────────────┤
  │ Speed      │ Slower (~2-3x)                    │ Faster                │
  ├────────────┼───────────────────────────────────┼───────────────────────┤
  │ Memory     │ Higher (stores coords + features) │ Lower (features only) │
  ├────────────┼───────────────────────────────────┼───────────────────────┤
  │ Parameters │ More (coordinate update layers)   │ Fewer                 │
  └────────────┴───────────────────────────────────┴───────────────────────┘

  ---
  Part 2: EGNN Input Reference

  Constructor (__init__)

  Required Parameters:

  EGNN(
      in_node_nf=5,        # Input node feature dim (e.g., atom types)
      in_edge_nf=0,        # UNUSED (distances computed from x)
      hidden_nf=128,       # Hidden layer dimension

  Important Optional Parameters:

  ┌──────────────────────┬─────────┬───────────────────────────────────┬───────────────────────────────────┐
  │      Parameter       │ Default │            Description            │          Typical Values           │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ n_layers             │ 3       │ Number of equivariant blocks      │ 3-7                               │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ coords_range         │ 15.0    │ Max coordinate update (Ångströms) │ 10.0-30.0                         │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ tanh                 │ False   │ Bound coordinate updates?         │ True for generation               │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ sin_embedding        │ False   │ Sinusoidal distance encoding?     │ True for better distance features │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ attention            │ False   │ Use attention mechanism?          │ False (start simple)              │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ normalization_factor │ 100     │ Message aggregation normalization │ 100 (molecular graphs)            │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ aggregation_method   │ 'sum'   │ How to aggregate messages         │ 'sum' or 'mean'                   │
  ├──────────────────────┼─────────┼───────────────────────────────────┼───────────────────────────────────┤
  │ device               │ 'cpu'   │ Where to run                      │ 'cuda' for GPU                    │
  └──────────────────────┴─────────┴───────────────────────────────────┴───────────────────────────────────┘

  Full Constructor Signature:
  EGNN(in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(),
       n_layers=3, attention=False, norm_diff=True, out_node_nf=None,
       tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
       sin_embedding=False, normalization_factor=100, aggregation_method='sum')

  Forward Pass

  h_out, x_out = egnn.forward(
      h,              # [n_nodes, in_node_nf] - Node features (REQUIRED)
      x,              # [n_nodes, 3] - 3D coordinates (REQUIRED)
      edge_index,     # [2, n_edges] - Graph connectivity (REQUIRED)
      node_mask=None, # [n_nodes, 1] - Binary mask for padding (OPTIONAL)
      edge_mask=None  # [n_edges, 1] - Binary mask for padding (OPTIONAL)
  )

  Input Shapes Example (Water molecule H₂O):

  # 3 atoms: O, H, H
  h = torch.tensor([
      [1, 0, 0, 0, 0],  # Oxygen (one-hot)
      [0, 1, 0, 0, 0],  # Hydrogen
      [0, 1, 0, 0, 0],  # Hydrogen
  ])  # Shape: [3, 5]

  x = torch.tensor([
      [0.0, 0.0, 0.0],    # O position
      [0.96, 0.0, 0.0],   # H1 position
      [-0.24, 0.93, 0.0], # H2 position
  ])  # Shape: [3, 3]

  edge_index = torch.tensor([
      [0, 1, 0, 2],  # Source: O→H1, H1→O, O→H2, H2→O
      [1, 0, 2, 0],  # Target
  ])  # Shape: [2, 4] (bidirectional O-H bonds)

  # Returns:
  # h_out: [3, 5] - updated node features
  # x_out: [3, 3] - refined coordinates

  ---
  Part 3: GNN Input Reference

  Constructor (__init__)

  Required Parameters:

  GNN(
      in_node_nf=10,       # Input node feature dim
      in_edge_nf=3,        # Input edge feature dim (actually used!)
      hidden_nf=64,        # Hidden layer dimension

  Important Optional Parameters:

  ┌──────────────────────┬─────────┬────────────────────────────────────────────────────────┐
  │      Parameter       │ Default │                      Description                       │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ n_layers             │ 4       │ Number of GCL layers                                   │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ aggregation_method   │ 'sum'   │ Message aggregation                                    │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ attention            │ False   │ Use attention?                                         │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ normalization_factor │ 1       │ Aggregation normalization (NOTE: different from EGNN!) │
  ├──────────────────────┼─────────┼────────────────────────────────────────────────────────┤
  │ device               │ 'cpu'   │ Device                                                 │
  └──────────────────────┴─────────┴────────────────────────────────────────────────────────┘

  Full Constructor Signature:
  GNN(in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum',
      device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=False,
      normalization_factor=1, out_node_nf=None)

  Forward Pass

  h_out = gnn.forward(
      h,              # [n_nodes, in_node_nf] - Node features (REQUIRED)
      edges,          # [2, n_edges] - Graph connectivity (REQUIRED)
      edge_attr=None, # [n_edges, in_edge_nf] - Edge features (OPTIONAL)
      node_mask=None, # [n_nodes, 1] - Binary mask (OPTIONAL)
      edge_mask=None  # [n_edges, 1] - Binary mask (OPTIONAL)
  )

  Input Shapes Example (Benzene C₆H₆):

  # 12 atoms: 6 carbons + 6 hydrogens
  h = torch.randn(12, 8)  # Shape: [12, 8] - atom features

  # Connectivity: C-C ring + C-H bonds (bidirectional)
  edges = torch.tensor([
      [0,1, 1,2, 2,3, 3,4, 4,5, 5,0,  # C-C ring
       0,6, 1,7, 2,8, 3,9, 4,10, 5,11],  # C-H bonds
      [1,0, 2,1, 3,2, 4,3, 5,4, 0,5,
       6,0, 7,1, 8,2, 9,3, 10,4, 11,5],
  ])  # Shape: [2, 24]

  edge_attr = torch.tensor([
      [1.0, 1.5],  # C-C aromatic bond
      [1.0, 1.5],  # (repeated for all C-C edges)
      ...
      [0.0, 1.0],  # C-H single bond
      ...
  ])  # Shape: [24, 2] - [is_aromatic, bond_order]

  # Returns:
  # h_out: [12, 8] - updated node features (NO coordinates!)

  ---
  Part 4: Side-by-Side Parameter Comparison

  Constructor Parameters

  ┌──────────────────────┬──────────────┬────────────┬───────────────────────────┐
  │      Parameter       │     EGNN     │    GNN     │      Key Difference       │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ in_node_nf           │ ✅           │ ✅         │ Same                      │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ in_edge_nf           │ ✅ (unused)  │ ✅ (used)  │ EGNN computes from coords │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ hidden_nf            │ ✅           │ ✅         │ Same                      │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ n_layers             │ Default: 3   │ Default: 4 │ Different defaults        │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ normalization_factor │ Default: 100 │ Default: 1 │ ⚠️  Very different!        │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ coords_range         │ ✅           │ ❌         │ EGNN only                 │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ tanh                 │ ✅           │ ❌         │ EGNN only                 │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ sin_embedding        │ ✅           │ ❌         │ EGNN only                 │
  ├──────────────────────┼──────────────┼────────────┼───────────────────────────┤
  │ inv_sublayers        │ ✅           │ ❌         │ EGNN only                 │
  └──────────────────────┴──────────────┴────────────┴───────────────────────────┘

  Forward Pass Inputs

  ┌─────────────────┬──────────────────────────┬──────────────────────────┐
  │      Input      │           EGNN           │           GNN            │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Node features h │ ✅ [n_nodes, in_node_nf] │ ✅ [n_nodes, in_node_nf] │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Coordinates x   │ ✅ [n_nodes, 3]          │ ❌ None                  │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Edge index      │ ✅ edge_index            │ ✅ edges (same format)   │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Edge features   │ Auto-computed            │ ✅ edge_attr (manual)    │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Node mask       │ ✅                       │ ✅                       │
  ├─────────────────┼──────────────────────────┼──────────────────────────┤
  │ Edge mask       │ ✅                       │ ✅                       │
  └─────────────────┴──────────────────────────┴──────────────────────────┘

  Outputs

  ┌───────────────────────┬─────────────────────────────┬─────────────────────────────┐
  │        Output         │            EGNN             │             GNN             │
  ├───────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │ Updated node features │ ✅ h [n_nodes, out_node_nf] │ ✅ h [n_nodes, out_node_nf] │
  ├───────────────────────┼─────────────────────────────┼─────────────────────────────┤
  │ Updated coordinates   │ ✅ x [n_nodes, 3]           │ ❌ None                     │
  └───────────────────────┴─────────────────────────────┴─────────────────────────────┘

  ---
  Part 5: Practical Code Examples

  Example 1: EGNN for Molecule Generation

  import torch
  from egnn_new import EGNN

  # Methane (CH4): 1 carbon + 4 hydrogens
  egnn = EGNN(
      in_node_nf=2,           # [is_carbon, is_hydrogen]
      in_edge_nf=0,           # Unused
      hidden_nf=128,
      n_layers=4,
      coords_range=10.0,
      tanh=True,              # Bound updates for stability
      sin_embedding=True,     # Better distance encoding
      device='cuda'
  )

  # Node features: one-hot atom types
  h = torch.tensor([
      [1.0, 0.0],  # Carbon
      [0.0, 1.0],  # H
      [0.0, 1.0],  # H
      [0.0, 1.0],  # H
      [0.0, 1.0],  # H
  ], device='cuda')

  # Random initial coordinates
  x = torch.randn(5, 3, device='cuda')

  # Connectivity: C connected to all H (bidirectional)
  edge_index = torch.tensor([
      [0,1, 0,2, 0,3, 0,4, 1,0, 2,0, 3,0, 4,0],
      [1,0, 2,0, 3,0, 4,0, 0,1, 0,2, 0,3, 0,4],
  ], device='cuda')

  # Iterative refinement
  for step in range(100):
      h, x = egnn(h, x, edge_index)

  print(f"Final coordinates:\n{x}")
  # x now contains realistic CH4 geometry!

  Example 2: GNN for Property Prediction

  import torch
  from egnn_new import GNN

  # Molecule property prediction (e.g., toxicity)
  gnn = GNN(
      in_node_nf=10,      # Atom features
      in_edge_nf=3,       # Bond features
      hidden_nf=64,
      n_layers=3,
      out_node_nf=1,      # Per-node output
      device='cpu'
  )

  # 9 atoms (e.g., ethanol)
  h = torch.randn(9, 10)

  # Connectivity
  edges = torch.tensor([
      [0,1, 1,0, 0,2, 2,0, ...],  # C-C, C-H, C-O bonds
      [1,0, 0,1, 2,0, 0,2, ...],
  ])

  # Bond features
  edge_attr = torch.randn(edges.shape[1], 3)

  # Forward
  node_embeddings = gnn(h, edges, edge_attr=edge_attr)

  # Global pooling for graph-level prediction
  graph_embedding = node_embeddings.mean(dim=0)
  print(f"Graph prediction: {graph_embedding}")

  ---
  Quick Decision Guide

  ┌────────────────────────────────┬───────┬──────────────────────────────────┐
  │           Your Task            │ Model │              Reason              │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ Generate 3D molecules          │ EGNN  │ Need coordinate outputs          │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ Predict solubility/toxicity    │ GNN   │ No geometry needed               │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ Refine molecular conformations │ EGNN  │ Equivariance preserves structure │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ Graph classification           │ GNN   │ Faster, simpler                  │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ Molecular dynamics             │ EGNN  │ Physical symmetries required     │
  ├────────────────────────────────┼───────┼──────────────────────────────────┤
  │ QSAR from 2D graphs            │ GNN   │ No 3D data                       │
  └────────────────────────────────┴───────┴──────────────────────────────────┘

  ---
  Critical Differences to Remember

  1. Edge Features:
    - EGNN: Auto-computed from x (distances) → ignore in_edge_nf parameter
    - GNN: Manual edge_attr required if in_edge_nf > 0
  2. Normalization Factor:
    - EGNN: Default 100 (for molecular graphs with many edges)
    - GNN: Default 1 (for general graphs)
  3. Coordinate Updates:
    - EGNN: Returns (h, x) - both features and coords
    - GNN: Returns h only - no coordinates
  4. Equivariance:
    - EGNN: E(3)-equivariant → preserves rotations/translations
    - GNN: No geometric guarantees




# Step-by-Step Walkthrough: egnn/models.py             

  Using a concrete batch: bs=2, n_nodes=4 (NH₃: N+3H), n_dims=3, in_node_nf=6 (5 atom types + charge)                            
  
  ---                                                                                                                            
  Shared Setup: The xh Tensor Format                                                                                           

  All 3 classes use xh as a packed tensor:
  xh: [bs, n_nodes, n_dims + in_node_nf]

  xh[:, :, :3]  → x  (3D coordinates)
  xh[:, :, 3:]  → h  (atom features)

  ---
  Class 1: EGNN_dynamics_QM9 — Diffusion Score Network

  Instantiation

  # IMPORTANT: caller pre-adds +1 to in_node_nf for the time dimension
  net_dynamics = EGNN_dynamics_QM9(
      in_node_nf=7,          # 6 atom features + 1 time slot (pre-inflated!)
      context_node_nf=0,
      n_dims=3,
      hidden_nf=64,
      condition_time=True,
  )
  # Internal EGNN: Linear(7→64) embedding

  Key design pattern: the CALLER adds +1 to in_node_nf to reserve a slot for the time scalar. The _forward then fills that slot
  at runtime.

  _forward(t, xh, node_mask, edge_mask, context=None)

  ┌─────────────────────────┬─────────────────────────────────────────┬───────────────┬─────────────────────────────────────┐
  │          Step           │                  Code                   │ Shape Before  │             Shape After             │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Unpack dims             │ bs, n_nodes, dims = xh.shape            │ [2,4,9]       │ bs=2, n_nodes=4, dims=9             │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ h_dims                  │ h_dims = dims - self.n_dims             │ —             │ h_dims = 6                          │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Build fully-connected   │ get_adj_matrix(4, 2, device)            │ —             │ edges: 2×[32] (all node pairs:      │
  │ edges                   │                                         │               │ 4×4×2 batches)                      │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Flatten node_mask       │ .view(bs*n_nodes, 1)                    │ [2,4,1]       │ [8,1]                               │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Flatten edge_mask       │ .view(bs*n_nodes*n_nodes, 1)            │ [2,4,4,1]     │ [32,1]                              │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Flatten & mask xh       │ .view(bs*n_nodes, -1) * node_mask       │ [2,4,9]       │ [8,9]                               │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Extract coords          │ x = xh[:, 0:3]                          │ [8,9]         │ x: [8,3]                            │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Extract features        │ h = xh[:, 3:]                           │ [8,9]         │ h: [8,6]                            │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Append time             │ cat([h, h_time], dim=1)                 │ [8,6]         │ h: [8,7] ← fills reserved slot      │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ (No context here)       │ —                                       │ —             │ —                                   │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ EGNN forward            │ h_final, x_final = self.egnn(h, x,      │ h:[8,7],      │ h_final:[8,7], x_final:[8,3]        │
  │                         │ edges, ...)                             │ x:[8,3]       │                                     │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Velocity = delta        │ vel = (x_final - x) * node_mask         │ —             │ vel: [8,3]                          │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Strip time dim          │ h_final = h_final[:, :-1]               │ [8,7]         │ [8,6]                               │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Reshape                 │ vel.view(bs, n_nodes, -1)               │ [8,3]         │ [2,4,3]                             │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Remove CoM drift        │ remove_mean_with_mask(vel, ...)         │ [2,4,3]       │ [2,4,3] zero-mean per molecule      │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Reshape h_final         │ .view(bs, n_nodes, -1)                  │ [8,6]         │ [2,4,6]                             │
  ├─────────────────────────┼─────────────────────────────────────────┼───────────────┼─────────────────────────────────────┤
  │ Return                  │ cat([vel, h_final], dim=2)              │ —             │ [2,4,9] ← same shape as input!      │
  └─────────────────────────┴─────────────────────────────────────────┴───────────────┴─────────────────────────────────────┘

  Output: [bs, n_nodes, 9]

  - [:, :, :3] → Predicted velocity (how much to move each atom to denoise)
  - [:, :, 3:] → Predicted atom feature update

  ---
  Class 2: EGNN_encoder_QM9 — VAE Posterior Encoder

  Instantiation

  encoder = EGNN_encoder_QM9(
      in_node_nf=6,    # raw atom features
      context_node_nf=0,
      out_node_nf=4,   # latent_nf: latent space dim
      n_dims=3,
      hidden_nf=64,
      n_layers=1,      # lightweight — only 1 layer!
      include_charges=True,
  )
  # Internal EGNN: Linear(6→64), out_node_nf=64 (feeds final_mlp)
  # final_mlp: Linear(64→64) → SiLU → Linear(64→9)
  #                                               ↑ = 1 + out_node_nf + out_node_nf = 1+4+4

  _forward(xh, node_mask, edge_mask, context=None) — no t!

  ┌──────────────────────────────────────────────────┬───────────────────────────────┬────────────────────────────────────┐
  │                       Step                       │             Shape             │                Note                │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ Setup (same as dynamics)                         │ x:[8,3], h:[8,6]              │ No time append                     │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ EGNN forward                                     │ h_final:[8,64], x_final:[8,3] │ out_node_nf=64 (→ final_mlp)       │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ vel = absolute positions                         │ vel = x_final * node_mask     │ NOT a delta! Latent coordinate     │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ Remove CoM, reshape                              │ vel: [2,4,3]                  │ x_mean (latent coord mean)         │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ final_mlp                                        │ h_final: [8,64] → [8,9]       │ 1 + 4 + 4 channels                 │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ Mask & reshape                                   │ h_final: [2,4,9]              │ —                                  │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ Unpack 9 channels:                               │                               │                                    │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ vel_std = h_final[:,:,:1] → sum over nodes → exp │ [2,4,1]                       │ Shared per-molecule coord variance │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ h_mean = h_final[:,:,1:5]                        │ [2,4,4]                       │ Latent feature means               │
  ├──────────────────────────────────────────────────┼───────────────────────────────┼────────────────────────────────────┤
  │ h_std = exp(0.5 * h_final[:,:,5:])               │ [2,4,4]                       │ Latent feature stds                │
  └──────────────────────────────────────────────────┴───────────────────────────────┴────────────────────────────────────┘

  Output: 4 tensors (VAE posterior q(z|molecule))

  ┌──────────┬─────────┬────────────────────────────────────────────────┐
  │  Tensor  │  Shape  │                    Meaning                     │
  ├──────────┼─────────┼────────────────────────────────────────────────┤
  │ vel_mean │ [2,4,3] │ μ for latent coordinates                       │
  ├──────────┼─────────┼────────────────────────────────────────────────┤
  │ vel_std  │ [2,4,1] │ σ for latent coordinates (shared per molecule) │
  ├──────────┼─────────┼────────────────────────────────────────────────┤
  │ h_mean   │ [2,4,4] │ μ for latent features                          │
  ├──────────┼─────────┼────────────────────────────────────────────────┤
  │ h_std    │ [2,4,4] │ σ for latent features                          │
  └──────────┴─────────┴────────────────────────────────────────────────┘

  Reparameterization: z = mean + std * ε,  ε ~ N(0,I)

  ---
  Class 3: EGNN_decoder_QM9 — VAE Decoder

  Instantiation

  decoder = EGNN_decoder_QM9(
      in_node_nf=4,    # latent_nf (decoder INPUT is the latent)
      context_node_nf=0,
      out_node_nf=6,   # reconstruct original atom features
      n_dims=3,
      hidden_nf=64,
      n_layers=4,      # full depth (encoder was 1, decoder is 4)
      include_charges=True,
  )
  # Internal EGNN: Linear(4→64), out_node_nf=6 (directly outputs atom features)
  # No final_mlp — direct output

  _forward(xh, node_mask, edge_mask, context=None) — no t

  ┌──────────────────────────┬──────────────────────────────────┬─────────────────────────────────────┐
  │           Step           │              Shape               │                Note                 │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ Input xh                 │ [2,4,7]                          │ 3 latent coords + 4 latent features │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ Setup                    │ x:[8,3], h:[8,4]                 │ Latent sample (not raw molecule)    │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ EGNN forward             │ h_final:[8,6], x_final:[8,3]     │ out_node_nf=6                       │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ vel = absolute positions │ vel = x_final * node_mask: [8,3] │ Reconstructed coords                │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ Remove CoM, reshape      │ vel: [2,4,3]                     │ x̂ (reconstructed coordinates)       │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ Mask h_final, reshape    │ h_final: [2,4,6]                 │ ĥ (reconstructed atom features)     │
  ├──────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
  │ Return                   │ vel, h_final                     │ 2 tensors                           │
  └──────────────────────────┴──────────────────────────────────┴─────────────────────────────────────┘

  Output: 2 tensors (point-estimate reconstruction)

  ┌─────────┬─────────┬─────────────────────────────────────────────┐
  │ Tensor  │  Shape  │                   Meaning                   │
  ├─────────┼─────────┼─────────────────────────────────────────────┤
  │ vel     │ [2,4,3] │ x̂ — reconstructed 3D coordinates            │
  ├─────────┼─────────┼─────────────────────────────────────────────┤
  │ h_final │ [2,4,6] │ ĥ — reconstructed atom type logits + charge │
  └─────────┴─────────┴─────────────────────────────────────────────┘

  ---
  Full Data Flow (VAE + Latent Diffusion)

  Real molecule xh [2,4,9]
           │
           ▼  EGNN_encoder_QM9
  vel_mean[2,4,3]  vel_std[2,4,1]  h_mean[2,4,4]  h_std[2,4,4]
           │
           ▼  Reparameterize: z = mean + std * ε
  z_x[2,4,3]  z_h[2,4,4]
           │
           ▼  Concatenate → z_xh [2,4,7]
           │
           ├──► EGNN_decoder_QM9 ──► vel[2,4,3]  h_final[2,4,6]
           │    (reconstruction)       (x̂)              (ĥ)
           │
           └──► EGNN_dynamics_QM9 ──► output [2,4,9]
                (latent diffusion      (score function:
                 denoising at t)        how to denoise z)

  ---
  Key Differences at a Glance

  ┌────────────────┬────────────────────────┬────────────────────┬────────────────────┐
  │                │     EGNN_dynamics      │    EGNN_encoder    │    EGNN_decoder    │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ Role           │ Diffusion denoiser     │ VAE encoder        │ VAE decoder        │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ Input dims     │ 3 + 6 = 9 (real/noisy) │ 3 + 6 = 9 (real)   │ 3 + 4 = 7 (latent) │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ Takes t?       │ ✅ appended to h       │ ❌                 │ ❌                 │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ n_layers       │ 4                      │ 1 (lightweight)    │ 4                  │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ Coord output   │ x_final - x (delta)    │ x_final (absolute) │ x_final (absolute) │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ Has final_mlp? │ ❌                     │ ✅ (→ mu/sigma)    │ ❌                 │
  ├────────────────┼────────────────────────┼────────────────────┼────────────────────┤
  │ # outputs      │ 1 tensor [bs,n,9]      │ 4 tensors          │ 2 tensors          │
  └────────────────┴────────────────────────┴────────────────────┴────────────────────┘

  ---











DMD2 + GeoLDM Integration Guide                     
                                                                                                                                  
  Both specialists have completed their analysis. Here is the full synthesis.                                                     
                                                                                                                                           
  ---                                                                                                                                      
  Part 1: Paper Core Ideas                                                                                                                 
                                                                                                                                           
  DMD2 in One Paragraph                                                                                                                    

  DMD2 distills a slow T-step diffusion model into a fast (1–4 step) generator G by minimizing the KL divergence between G's output
  distribution and the teacher's real distribution — not pointwise, but in aggregate. The gradient of this KL is tractable via two score
  networks:

  ∇_θ L_DMD = -E_t [(s_real(z_t, t) - s_fake(z_t, t)) · (∂G/∂θ)]

  where:
    z_t = F(G(z), t) = α_t·G(z) + σ_t·ε     ← noise-inject G's output
    s_real(z_t, t) = -(z_t - α_t·μ_real(z_t, t)) / σ_t²   ← frozen teacher
    s_fake(z_t, t) = -(z_t - α_t·μ_fake(z_t, t)) / σ_t²   ← trained online on G's outputs

  A GAN loss L_GAN is added on top, with D as a classification head on μ_fake's bottleneck. No regression loss. μ_fake is updated 5× per 1
  G update.

  GeoLDM in One Paragraph

  GeoLDM is a two-stage system. Stage 1 trains a geometric VAE (1-layer EGNN encoder, 4-layer EGNN decoder) that maps molecules (x, h) →
  latent z = (z_x ∈ ℝ^{N×3}, z_h ∈ ℝ^{N×k}). Stage 2 trains a diffusion model (9-layer EGNN) directly in the latent space. At inference:
  sample z_T ~ N(0,I), run T reverse denoising steps to get z_0, decode z_0 → (x, h) via the frozen decoder. All coordinate operations
  enforce zero-center-of-mass (equivariance constraint).

  ---
  Part 2: Component Mapping

  ┌────────────────────────────┬──────────────────────────────────────────────────────┬───────────────────────────────────────────────┐
  │         DMD2 Role          │                   GeoLDM Component                   │             Codebase Class/Method             │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ G (generator)              │ LDM denoiser = N-step sampler in latent space        │ EnLatentDiffusion.sample() + phi()            │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ μ_real (frozen teacher)    │ Fully pretrained EnLatentDiffusion loaded from       │ teacher.phi(z_t, t, ...) inside               │
  │                            │ checkpoint                                           │ torch.no_grad()                               │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ μ_fake (fake score,        │ Second copy of LDM denoiser, trained on G's outputs  │ model.phi(z_t, t, ...) with gradient          │
  │ trained)                   │                                                      │                                               │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ F (noise injection)        │ GeoLDM forward diffusion                             │ z_t = α_t·z_0 + σ_t·ε with CoM correction on  │
  │                            │                                                      │ z_x                                           │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ D (discriminator)          │ New: classification head on μ_fake's EGNN bottleneck │ To be written                                 │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ VAE encoder E_φ            │ Frozen after Stage 1                                 │ EnHierarchicalVAE.encode()                    │
  ├────────────────────────────┼──────────────────────────────────────────────────────┼───────────────────────────────────────────────┤
  │ VAE decoder D_ξ            │ Frozen; used only at inference to decode G's output  │ EnHierarchicalVAE.decode()                    │
  └────────────────────────────┴──────────────────────────────────────────────────────┴───────────────────────────────────────────────┘

  Key insight: DMD2 operates entirely in latent space (z_x, z_h). The VAE decoder is outside the training loop — it's only used at
  inference to get the final molecule.

  ---
  Part 3: Input/Output Shapes Through the Training Loop

  For latent_nf=2, N=29 (QM9), B=batch_size:

  Latent tensor z:  (B, N, 5)    # 3 position + 2 feature dims
    z[:, :, :3]  → z_x           equivariant, zero-CoM
    z[:, :, 3:]  → z_h           invariant features

  phi() call:
    IN:   z_t        (B, N, 5)
          t           (B, 1)       normalized to [0, 1]
          node_mask  (B, N, 1)
          edge_mask  (B·N², 1)
    OUT:  ε_pred     (B, N, 5)   same shape as input z_t
            [:,:,:3] → equivariant (zero-CoM) position noise
            [:,:,3:] → invariant feature noise

  denoise_step() → x0_pred:
    x0_pred = (z_t - σ_t · ε_pred) / α_t     shape: (B, N, 5)

  score s from mu:
    s(z_t, t) = -(z_t - α_t · mu(z_t, t)) / σ_t²    shape: (B, N, 5)
    where mu(z_t, t) = (z_t - σ_t · phi(z_t, t)) / α_t

  ---
  Part 4: Code Writing Guide

  File Structure to Create

  DMDMolGen/
  ├── dmd/
  │   ├── __init__.py
  │   ├── loss.py          ← DMD2 loss functions
  │   ├── discriminator.py ← GAN head on top of mu_fake's EGNN
  │   └── sampling.py      ← G's forward pass (with gradients)
  ├── train_dmd.py         ← Training loop (modeled on train_progdistill.py)
  └── main_dmd.py          ← Entry point (modeled on main_progdistill.py)

  ---
  Step 1 — Model Instantiation (main_dmd.py)

  Modeled on main_progdistill.py. You need three model objects:

  import copy

  # 1. mu_real: fully pretrained GeoLDM, loaded from checkpoint, FROZEN
  mu_real, nodes_dist, prop_dist = get_latent_diffusion(args, ...)
  mu_real.load_state_dict(torch.load(args.pretrained_model_path))
  mu_real.eval()
  for p in mu_real.parameters():
      p.requires_grad_(False)

  # 2. G (generator): initialized from mu_real weights, TRAINED
  G = copy.deepcopy(mu_real)
  G.train()
  # G.vae stays frozen (same as progdistill)
  for p in G.vae.parameters():
      p.requires_grad_(False)
  # Only G.dynamics is trained
  for p in G.dynamics.parameters():
      p.requires_grad_(True)

  # 3. mu_fake: second copy of mu_real, TRAINED (5x per 1 G update)
  mu_fake = copy.deepcopy(mu_real)
  mu_fake.train()
  for p in mu_fake.dynamics.parameters():
      p.requires_grad_(True)

  # 4. Discriminator: new classification head (see Step 2)
  discriminator = MolecularDiscriminator(hidden_nf=args.nf)

  # Optimizers
  optim_G    = torch.optim.AdamW(G.dynamics.parameters(), lr=args.lr)
  optim_fake = torch.optim.AdamW(
      list(mu_fake.dynamics.parameters()) + list(discriminator.parameters()),
      lr=args.lr)

  ---
  Step 2 — Discriminator (dmd/discriminator.py)

  The GAN head must be permutation-invariant (pool over N nodes) and rotation-invariant (operates on invariant node features h, not
  equivariant x):

  class MolecularDiscriminator(nn.Module):
      """
      Attaches to the bottleneck of mu_fake's EGNN.
      Input: node features h_final from EGNN, shape (B, N, hidden_nf)
      Output: scalar logit per molecule, shape (B, 1)
      """
      def __init__(self, hidden_nf):
          super().__init__()
          self.mlp = nn.Sequential(
              nn.Linear(hidden_nf, hidden_nf),
              nn.SiLU(),
              nn.Linear(hidden_nf, 1)
          )

      def forward(self, h_node_features, node_mask):
          # h_node_features: (B, N, hidden_nf) — invariant node features
          # node_mask: (B, N, 1)
          logits = self.mlp(h_node_features)              # (B, N, 1)
          logits = logits * node_mask                      # zero out padding
          N_real = node_mask.sum(dim=1, keepdim=True)     # (B, 1, 1)
          pooled = logits.sum(dim=1) / N_real.squeeze(-1) # (B, 1) mean pooling
          return pooled !!!Wrong! pool then mlp!

  Note: To use this, you'll need to modify EGNN_dynamics_QM9._forward (or add a wrapper) to also return the pre-final EGNN node features
  h_final before the embedding-out projection. Currently _forward only returns the concatenated [vel, h_final] output. Add a flag
  return_bottleneck=False to optionally expose this.

  ---
  Step 3 — DMD2 Loss (dmd/loss.py)

  def compute_score(mu_output, z_t, alpha_t, sigma_t):
      """
      Convert denoised estimate mu(z_t, t) to score s(z_t, t).
      mu_output: predicted noise ε (same as phi output), shape (B, N, dims)
      Returns score s = -(z_t - α_t·mu) / σ_t²
      """
      # phi predicts ε, so mu (denoised estimate) = (z_t - σ_t·ε) / α_t
      mu = (z_t - sigma_t * mu_output) / alpha_t       # (B, N, dims)
      score = -(z_t - alpha_t * mu) / (sigma_t ** 2)  # (B, N, dims)
      return score


  def dmd_generator_loss(G, mu_real, mu_fake, z_0, t, alpha_t, sigma_t,
                          node_mask, edge_mask, context):
      """
      Compute DMD2 gradient signal for generator G.
      z_0: G's output latent, shape (B, N, dims)   ← must have grad
      """
      # Noise-inject G's output to level t
      eps = sample_combined_position_feature_noise(...)  # CoM-corrected
      z_t = alpha_t * z_0 + sigma_t * eps               # (B, N, dims)

      # Real score (frozen teacher)
      with torch.no_grad():
          eps_real = mu_real.phi(z_t, t, node_mask, edge_mask, context)
      s_real = compute_score(eps_real, z_t, alpha_t, sigma_t)

      # Fake score (mu_fake, no grad through mu_fake weights)
      with torch.no_grad():
          eps_fake = mu_fake.phi(z_t, t, node_mask, edge_mask, context)
      s_fake = compute_score(eps_fake, z_t, alpha_t, sigma_t)

      # Score difference — this IS the DMD gradient signal
      # Backprop through z_t → z_0 → G_theta
      score_diff = (s_real - s_fake).detach()         # stop grad through scores
      loss = -(score_diff * z_0).sum()                # dot product for gradient
      return loss


  def mu_fake_denoising_loss(mu_fake, z_fake_t, eps_true, t, node_mask, edge_mask, context):
      """
      Train mu_fake to denoise G's outputs (fake denoising score matching).
      """
      eps_pred = mu_fake.phi(z_fake_t, t, node_mask, edge_mask, context)
      loss = ((eps_pred - eps_true) ** 2 * node_mask.unsqueeze(-1)).mean()
      return loss


  def gan_discriminator_loss(D, mu_fake, z_real_t, z_fake_t, t,
                              node_mask, edge_mask, context):
      """
      Non-saturating GAN loss for discriminator D.
      """
      # Get bottleneck features from mu_fake for real and fake
      h_real = mu_fake_bottleneck(mu_fake, z_real_t, t, node_mask, edge_mask, context)
      h_fake = mu_fake_bottleneck(mu_fake, z_fake_t, t, node_mask, edge_mask, context)

      logit_real = D(h_real, node_mask)
      logit_fake = D(h_fake, node_mask)

      loss_D = -F.logsigmoid(logit_real).mean() - F.logsigmoid(-logit_fake).mean()
      return loss_D


  def gan_generator_loss(D, mu_fake, z_fake_t, t, node_mask, edge_mask, context):
      h_fake = mu_fake_bottleneck(mu_fake, z_fake_t, t, node_mask, edge_mask, context)
      logit_fake = D(h_fake.detach(), node_mask)  # stop grad through D features
      loss_G_gan = -F.logsigmoid(logit_fake).mean()
      return loss_G_gan

  ---
  Step 4 — G's Forward Pass with Gradients (dmd/sampling.py)

  The key change from EnLatentDiffusion.sample() is enabling gradients and using a small fixed number of steps:

  def generate_fake_sample(G, n_steps, node_mask, edge_mask, context, device):
      """
      Run G for n_steps (e.g. 1 or 4) with gradients enabled.
      Returns z_0: (B, N, 3+latent_nf) in latent space.
      """
      B, N, _ = node_mask.shape[:3]

      # Start from pure noise (CoM-corrected for position channel)
      z = G.sample_combined_position_feature_noise(B, N, node_mask)

      # Predefined step schedule (e.g. for 4 steps: [999, 749, 499, 249]/1000)
      timesteps = torch.linspace(1.0, 1.0/G.T, n_steps, device=device)

      for t in timesteps:
          t_batch = t.expand(B, 1)
          alpha_t = G.gamma.alpha(t_batch)
          sigma_t = G.gamma.sigma(t_batch)

          # Denoising step → x0_pred
          eps_pred = G.phi(z, t_batch, node_mask, edge_mask, context)
          x0_pred = (z - sigma_t * eps_pred) / alpha_t   # Tweedie formula

          # Re-inject noise to next step level (if not final step)
          if t > timesteps[-1]:
              t_next = ... # next timestep
              alpha_next = G.gamma.alpha(t_next)
              sigma_next = G.gamma.sigma(t_next)
              eps_new = G.sample_combined_position_feature_noise(B, N, node_mask)
              z = alpha_next * x0_pred + sigma_next * eps_new
          else:
              z = x0_pred

      return z   # z_0: G's output latent (B, N, 3+latent_nf), with gradients

  ---
  Step 5 — Training Loop (train_dmd.py)

  Modeled on train_progdistill.py but with the DMD2 two-timescale update:

  def train_epoch(G, mu_real, mu_fake, D, optim_G, optim_fake,
                  loader, args, node_mask, edge_mask):

      for batch in loader:
          x, h = batch

          # ── INNER LOOP: Update mu_fake and D (5× per 1 G update) ──
          for _ in range(5):
              optim_fake.zero_grad()

              # Generate fake z_0 (no grad through G here)
              with torch.no_grad():
                  z_fake_0 = generate_fake_sample(G, args.n_gen_steps, ...)

              # Sample noise level t
              t = sample_t(args)
              alpha_t, sigma_t = G.gamma.alpha(t), G.gamma.sigma(t)
              eps = G.sample_combined_position_feature_noise(...)

              # Noise-inject fake sample
              z_fake_t = alpha_t * z_fake_0 + sigma_t * eps

              # Encode real data to latent
              with torch.no_grad():
                  z_real_0 = encode_to_latent_space(mu_real, x, h, ...)
              z_real_t = alpha_t * z_real_0 + sigma_t * eps

              # mu_fake denoising loss on fake data
              loss_score = mu_fake_denoising_loss(mu_fake, z_fake_t, eps, t, ...)

              # GAN discriminator loss
              loss_D = gan_discriminator_loss(D, mu_fake, z_real_t, z_fake_t, t, ...)

              (loss_score + loss_D).backward()
              optim_fake.step()

          # ── OUTER LOOP: Update G (1×) ──
          optim_G.zero_grad()

          # Generate fake z_0 WITH gradients
          z_fake_0 = generate_fake_sample(G, args.n_gen_steps, ...)

          t = sample_t(args)
          alpha_t, sigma_t = G.gamma.alpha(t), G.gamma.sigma(t)

          # DMD score matching loss
          loss_dmd = dmd_generator_loss(G, mu_real, mu_fake, z_fake_0, t, ...)

          # GAN generator loss
          z_fake_t = alpha_t * z_fake_0 + sigma_t * eps
          loss_gan = gan_generator_loss(D, mu_fake, z_fake_t, t, ...)

          total_loss = loss_dmd + args.lambda_gan * loss_gan
          total_loss.backward()
          optim_G.step()

  ---
  Part 5: Critical Implementation Constraints

  1. Always preserve zero-CoM on position channel

  Every time you compute z_t = α_t·z_0 + σ_t·ε, the noise ε_x must have zero center of mass. Use the existing
  sample_combined_position_feature_noise() — do not replace it with plain torch.randn.

  2. mu_fake must be an EGNN (equivariant)

  Do NOT use a standard MLP or non-equivariant network as μ_fake. The latent z_x is an equivariant tensor. The existing EGNN_dynamics_QM9
  already satisfies this.

  3. Discriminator pools over nodes, not raw positions

  D should operate on the invariant node features h from the EGNN bottleneck (not z_x), then pool with node_mask to handle variable
  molecule sizes.

  4. Stop gradients correctly

  ┌────────────────────────┬───────────────────────┬───────────────────────────┐
  │     Gradient flow      │      Should flow      │      Should NOT flow      │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ DMD loss → G           │ ✅ Through z_0 = G(z) │ ❌ Through score networks │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ mu_fake denoising loss │ ✅ Through mu_fake    │ ❌ Through G              │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ GAN D loss             │ ✅ Through D          │ ❌ Through G              │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ GAN G loss             │ ✅ Through G          │ ❌ Through D weights      │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ mu_real                │ ❌ Always frozen      │ —                         │
  ├────────────────────────┼───────────────────────┼───────────────────────────┤
  │ VAE encoder/decoder    │ ❌ Always frozen      │ —                         │
  └────────────────────────┴───────────────────────┴───────────────────────────┘

  5. Build incrementally

  Recommended build order:
  1. Start here: Get generate_fake_sample() producing valid latents with gradients
  2. Verify dmd_generator_loss() backpropagates through G
  3. Add mu_fake_denoising_loss() and inner loop
  4. Add MolecularDiscriminator + GAN losses last

  ---
  Summary Checklist

  Phase 1 — Model setup
    □ Load pretrained GeoLDM as mu_real (freeze all)
    □ Deep-copy as G (freeze VAE, train dynamics only)
    □ Deep-copy as mu_fake (train dynamics + new D head)
    □ Implement MolecularDiscriminator class

  Phase 2 — Core mechanics
    □ Implement generate_fake_sample() with grad support
    □ Implement compute_score() utility
    □ Implement dmd_generator_loss()
    □ Implement mu_fake_denoising_loss()

  Phase 3 — GAN components
    □ Expose EGNN bottleneck features from _forward()
    □ Implement gan_discriminator_loss()
    □ Implement gan_generator_loss()

  Phase 4 — Training loop
    □ Write train_dmd.py with 5:1 inner/outer update ratio
    □ Write main_dmd.py with checkpoint loading
    □ Add EMA for G (reuse existing EMA infrastructure)


 Classes

  ┌──────────────────┬───────────────────┬────────────────────────────────────────────┐
  │     Variable     │       Class       │                    File                    │
  ├──────────────────┼───────────────────┼────────────────────────────────────────────┤
  │ G                │ EnLatentDiffusion │ equivariant_diffusion/en_diffusion.py:1057 │
  ├──────────────────┼───────────────────┼────────────────────────────────────────────┤
  │ mu_real          │ EnLatentDiffusion │ same — it is teacher                       │
  ├──────────────────┼───────────────────┼────────────────────────────────────────────┤
  │ mu_fake_dynamics │ EGNN_dynamics_QM9 │ egnn/models.py                             │
  └──────────────────┴───────────────────┴────────────────────────────────────────────┘

  EnLatentDiffusion inherits from EnVariationalDiffusion which inherits from nn.Module.

  ---
  How EnLatentDiffusion carries VAE + DDPM

  EnLatentDiffusion
  ├── .vae              EnHierarchicalVAE      — the frozen first stage
  │     ├── .encoder   EGNN_encoder_QM9
  │     └── .decoder   EGNN_decoder_QM9
  ├── .dynamics         EGNN_dynamics_QM9      — the score network (DDPM denoiser)
  │     └── .egnn      EGNN
  │           └── .embedding_out  nn.Linear(hidden_nf → out_node_nf)  ← hook target
  └── .gamma            GammaNetwork           — noise schedule γ(t)

  ---
  Callable methods on G and mu_real (EnLatentDiffusion)

  ┌──────────────────────────────────────────────────────────────────┬─────────────────┬───────────────────────────────────────────────┐
  │                              Method                              │   Grad needed   │                 What it does                  │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.sample(n_samples, n_nodes, node_mask, edge_mask, context)      │ YES             │ Full reverse diffusion in latent space → VAE  │
  │                                                                  │                 │ decode → (x, h)                               │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.phi(z_t, t, node_mask, edge_mask, context)                     │ YES             │ Score network forward → ε_pred                │
  │                                                                  │                 │ [B,N,3+latent_nf]                             │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.sample_combined_position_feature_noise(n_samples, n_nodes,     │ No              │ Sample z_T ~ N(0,I) with zero-CoM constraint  │
  │ node_mask)                                                       │                 │ on z_x                                        │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.gamma(t)                                                       │ No              │ Noise schedule → scalar γ                     │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.alpha(gamma, x) / G.sigma(gamma, x)                            │ No              │ α_t and σ_t from γ                            │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.inflate_batch_array(arr, target)                               │ No              │ Broadcast [B] → [B,1,1] to match shape of x   │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.vae.encode(x, h, node_mask, edge_mask, context)                │ No (encoder     │ x,h → z_x_mu, z_x_sigma, z_h_mu, z_h_sigma    │
  │                                                                  │ frozen)         │                                               │
  ├──────────────────────────────────────────────────────────────────┼─────────────────┼───────────────────────────────────────────────┤
  │ G.vae.decode(z_xh, node_mask, edge_mask, context)                │ YES (for G      │ z_xh [B,N,3+latent_nf] → x, h                 │
  │                                                                  │ loss)           │                                               │
  └──────────────────────────────────────────────────────────────────┴─────────────────┴───────────────────────────────────────────────┘

  mu_real has the same interface as G — it is the same EnLatentDiffusion class, just frozen (requires_grad=False) and in .eval() mode.

  ---
  Callable methods on mu_fake_dynamics (EGNN_dynamics_QM9)

  mu_fake_dynamics does not have .vae, .gamma, .alpha, .sigma — those live on EnLatentDiffusion. It only has:

  ┌──────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────┐
  │                              Method                              │                  What it does                   │
  ├──────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ mu_fake_dynamics._forward(t, z_t, node_mask, edge_mask, context) │ Same as phi() — raw score network call → ε_pred │
  └──────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────┘

  To use it like phi(), you call mu_fake_dynamics._forward(t, z_t, ...) directly, or wrap it:
  # Equivalent to G.phi() but using mu_fake's weights:
  eps_fake = mu_fake_dynamics._forward(t, z_t, node_mask, edge_mask, context)
  Note: t and z_t argument order is swapped vs phi():
  - phi(z_t, t, ...) — EnLatentDiffusion
  - _forward(t, z_t, ...) — EGNN_dynamics_QM9

  For the noise schedule scalars (α_t, σ_t) when scoring with mu_fake, borrow them from G or mu_real since they share the same schedule.