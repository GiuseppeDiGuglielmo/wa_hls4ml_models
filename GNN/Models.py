import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch # Batch is implicitly used by DataLoader




class FPGA_GNN(nn.Module):
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=2, mlp_hidden_dim=64, dropout_rate=0.2):
        """
        GraphSAGE based GNN for FPGA resource and latency prediction.

        Args:
            node_feature_dim (int): Dimensionality of node features.
            num_targets (int): Number of target values to predict (6 in your case).
            hidden_dim (int): Hidden dimension for SAGEConv layers.
            num_gnn_layers (int): Number of SAGEConv layers.
            mlp_hidden_dim (int): Hidden dimension for the MLP head.
            dropout_rate (float): Dropout rate.
        """
        super(FPGA_GNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        # Initial SAGEConv layer
        self.convs.append(SAGEConv(node_feature_dim, hidden_dim, aggr='sum'))
        
        # Additional SAGEConv layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr='sum'))

        # MLP head for graph-level regression
        # The input to the MLP will be the pooled graph embedding + the strategy feature
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim), # +4 for the strategy and io_type features
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )
        
        # Batch normalization for GNN layers
        self.bns = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        """
        Forward pass of the GNN.

        Args:
            data (torch_geometric.data.Batch or torch_geometric.data.Data):
                A batch of graph data containing:
                - data.x: Node features [num_nodes_in_batch, node_feature_dim]
                - data.edge_index: Edge connectivity [2, num_edges_in_batch]
                - data.batch: Batch assignment vector [num_nodes_in_batch]
                - data.strategy: Global strategy feature [batch_size, 2]
                - data.io_type: Global io_type feature [batch_size, 2]
        
        Returns:
            torch.Tensor: Predicted target values [batch_size, num_targets]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Node embeddings through SAGEConv layers
        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)  # Apply batch normalization
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Global pooling (sum aggregation as discussed)
        # Alternatives: global_mean_pool, global_max_pool
        graph_embedding = global_add_pool(x, batch) # [batch_size, hidden_dim]
        
        # Concatenate global strategy and io_type features (both one-hot encoded)
        strategy_feature = data.strategy  # [batch_size, 2]
        io_type_feature = data.io_type    # [batch_size, 2]
        
        # Handle potential dimension issues
        if strategy_feature.dim() == 3 and strategy_feature.shape[2] == 1:
            strategy_feature = strategy_feature.squeeze(-1)
        if io_type_feature.dim() == 3 and io_type_feature.shape[2] == 1:
            io_type_feature = io_type_feature.squeeze(-1)
        
        # Verify batch sizes
        if strategy_feature.shape[0] != graph_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) and strategy feature ({strategy_feature.shape[0]})")
        if io_type_feature.shape[0] != graph_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) and io_type feature ({io_type_feature.shape[0]})")

        # Concatenate all: graph_embedding + strategy + io_type
        combined_embedding = torch.cat([graph_embedding, strategy_feature, io_type_feature], dim=1) # [batch_size, hidden_dim + 4]
        
        # MLP for prediction
        output = self.mlp(combined_embedding)
        
        return output
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class FPGA_GNN_GATv2(nn.Module):
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=3, 
                 num_attention_heads=4, mlp_hidden_dim=64, dropout_rate=0.2, 
                 edge_dim=None, concat_heads=True, residual_connections=True):
        """
        GATv2-based GNN for FPGA resource and latency prediction.

        Args:
            node_feature_dim (int): Dimensionality of node features.
            num_targets (int): Number of target values to predict (6 in your case).
            hidden_dim (int): Hidden dimension for GATv2Conv layers.
            num_gnn_layers (int): Number of GATv2Conv layers.
            num_attention_heads (int): Number of attention heads for GATv2Conv.
            mlp_hidden_dim (int): Hidden dimension for the MLP head.
            dropout_rate (float): Dropout rate.
            edge_dim (int, optional): Edge feature dimension if using edge features.
            concat_heads (bool): Whether to concatenate attention heads or average them.
            residual_connections (bool): Whether to use residual connections between layers.
        """
        super(FPGA_GNN_GATv2, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.concat_heads = concat_heads
        self.residual_connections = residual_connections
        
        # Calculate dimensions based on concatenation strategy
        if concat_heads:
            gat_out_dim = hidden_dim * num_attention_heads
        else:
            gat_out_dim = hidden_dim
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()  # Layer normalization
        
        # Initial GATv2Conv layer
        self.convs.append(
            GATv2Conv(node_feature_dim, hidden_dim, 
                      heads=num_attention_heads, 
                      dropout=dropout_rate,
                      edge_dim=edge_dim,
                      concat=concat_heads)
        )
        self.norms.append(nn.LayerNorm(gat_out_dim))
        
        # Additional GATv2Conv layers
        for i in range(num_gnn_layers - 1):
            # For intermediate layers, input dim depends on concat strategy
            in_dim = gat_out_dim
            
            # Keep all layers with same configuration for simplicity
            self.convs.append(
                GATv2Conv(in_dim, hidden_dim, 
                          heads=num_attention_heads,
                          dropout=dropout_rate,
                          edge_dim=edge_dim,
                          concat=concat_heads)
            )
            self.norms.append(nn.LayerNorm(gat_out_dim))
        
        # Projection layers for residual connections if needed
        self.residual_projs = nn.ModuleList()
        if residual_connections:
            # First residual projection
            if node_feature_dim != gat_out_dim:
                self.residual_projs.append(nn.Linear(node_feature_dim, gat_out_dim))
            else:
                self.residual_projs.append(nn.Identity())
            
            # Remaining residual projections - all stay at gat_out_dim
            for i in range(num_gnn_layers - 1):
                self.residual_projs.append(nn.Identity())
        
        # Final projection to reduce dimension before pooling
        self.final_projection = nn.Linear(gat_out_dim, hidden_dim)
        
        # Global pooling - using multiple pooling strategies
        self.pool_weight = nn.Parameter(torch.ones(3) / 3)  # Learnable pooling weights
        
        # MLP head for graph-level regression
        # Input: pooled features (hidden_dim) + strategy feature (1)
        mlp_input_dim = hidden_dim + 4  # +2 for strategy OHE + 2 for io_type OHE
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )

    def forward(self, data):
        """
        Forward pass of the GATv2-based GNN.

        Args:
            data (torch_geometric.data.Batch or torch_geometric.data.Data):
                A batch of graph data containing:
                - data.x: Node features [num_nodes_in_batch, node_feature_dim]
                - data.edge_index: Edge connectivity [2, num_edges_in_batch]
                - data.batch: Batch assignment vector [num_nodes_in_batch]
                - data.strategy: Global strategy feature [batch_size, 1]
                - data.edge_attr (optional): Edge features [num_edges_in_batch, edge_dim]
        
        Returns:
            torch.Tensor: Predicted target values [batch_size, num_targets]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Store attention weights for potential visualization
        attention_weights = []
        
        # Node embeddings through GATv2Conv layers
        for i in range(self.num_gnn_layers):
            identity = x
            
            # Apply GATv2Conv
            if edge_attr is not None:
                x, (edge_index_out, alpha) = self.convs[i](x, edge_index, edge_attr, 
                                                           return_attention_weights=True)
            else:
                x, (edge_index_out, alpha) = self.convs[i](x, edge_index, 
                                                           return_attention_weights=True)
            
            # Store attention weights (optional, for analysis)
            attention_weights.append(alpha)
            
            # Apply normalization
            x = self.norms[i](x)
            
            # Apply activation and dropout
            x = F.elu(x)  # ELU often works better with attention mechanisms
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection
            if self.residual_connections:
                proj_identity = self.residual_projs[i](identity)
                x = x + proj_identity
        
        # Project to hidden_dim before pooling
        x = self.final_projection(x)
        
        # Multi-strategy global pooling
        # Combine different pooling strategies with learnable weights
        pool_weights = F.softmax(self.pool_weight, dim=0)
        
        graph_embedding_add = global_add_pool(x, batch)
        graph_embedding_mean = global_mean_pool(x, batch)
        graph_embedding_max = global_max_pool(x, batch)
        
        # Weighted combination of pooling strategies
        graph_embedding = (pool_weights[0] * graph_embedding_add + 
                          pool_weights[1] * graph_embedding_mean + 
                          pool_weights[2] * graph_embedding_max)
        
        # Concatenate global features: strategy (one-hot) + io_type (one-hot)
        strategy_feature = data.strategy  # Shape: [batch_size, 2]
        io_type_feature = data.io_type    # Shape: [batch_size, 2]
        
        # Handle dimension issues if they exist
        if strategy_feature.dim() == 3:
            strategy_feature = strategy_feature.squeeze(-1)
        if io_type_feature.dim() == 3:
            io_type_feature = io_type_feature.squeeze(-1)
        
        # Verify batch sizes match
        if strategy_feature.shape[0] != graph_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) "
                           f"and strategy feature ({strategy_feature.shape[0]})")
        if io_type_feature.shape[0] != graph_embedding.shape[0]:
            raise ValueError(f"Batch size mismatch between graph embedding ({graph_embedding.shape[0]}) "
                           f"and io_type feature ({io_type_feature.shape[0]})")
        
        # Concatenate all features: graph_embedding + strategy + io_type
        combined_embedding = torch.cat([graph_embedding, strategy_feature, io_type_feature], dim=1)
        
        # MLP for prediction
        output = self.mlp(combined_embedding)
        
        return output


class FPGA_GNN_GATv2_Enhanced(nn.Module):
    """
    Enhanced version with additional features like edge features and skip connections across all layers.
    """
    def __init__(self, node_feature_dim, num_targets, hidden_dim=128, num_gnn_layers=3, 
                 num_attention_heads=4, mlp_hidden_dim=64, dropout_rate=0.2,
                 use_edge_features=True, edge_feature_dim=1):
        super(FPGA_GNN_GATv2_Enhanced, self).__init__()
        
        self.use_edge_features = use_edge_features
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        
        # Edge feature embedding (if using edge features)
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feature_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            )
            edge_dim = hidden_dim // 4
        else:
            edge_dim = None
        
        # GATv2 layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        # Track dimensions for skip connections
        current_dim = node_feature_dim
        skip_dims = [current_dim]
        
        for i in range(num_gnn_layers):
            # GATv2Conv layer
            self.convs.append(
                GATv2Conv(current_dim, hidden_dim, 
                          heads=num_attention_heads,
                          dropout=dropout_rate,
                          edge_dim=edge_dim,
                          concat=True)
            )
            
            # Update current dimension
            current_dim = hidden_dim * num_attention_heads
            skip_dims.append(current_dim)
            
            # Layer normalization
            self.norms.append(nn.LayerNorm(current_dim))
            
            # Skip connection projection
            if i > 0:  # Skip connections from all previous layers
                skip_dim_total = sum(skip_dims[:i+1])
                self.skip_connections.append(
                    nn.Linear(skip_dim_total + current_dim, current_dim)
                )
        
        # Final projection to standard hidden dimension
        self.final_projection = nn.Linear(current_dim, hidden_dim)
        
        # Global attention pooling layer
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim),  # +1 for strategy
            nn.LayerNorm(mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.LayerNorm(mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, num_targets)
        )
    
    def create_edge_features(self, edge_index, x):
        """
        Create edge features based on node features.
        For now, using the absolute difference in node feature magnitudes.
        """
        row, col = edge_index
        # Compute L2 norm of node features
        node_norms = torch.norm(x, p=2, dim=1)
        # Edge feature is the absolute difference in norms
        edge_features = torch.abs(node_norms[row] - node_norms[col]).unsqueeze(1)
        return edge_features
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Create or process edge features
        if self.use_edge_features:
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = self.edge_encoder(data.edge_attr)
            else:
                # Create edge features from node features
                edge_features = self.create_edge_features(edge_index, x)
                edge_attr = self.edge_encoder(edge_features)
        else:
            edge_attr = None
        
        # Store all layer outputs for skip connections
        layer_outputs = [x]
        
        # Forward through GATv2 layers
        for i in range(self.num_gnn_layers):
            # Apply GATv2Conv
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Skip connections from all previous layers
            if i > 0:
                skip_features = torch.cat(layer_outputs + [x], dim=1)
                x = self.skip_connections[i-1](skip_features)
                x = F.elu(x)
            
            layer_outputs.append(x)
        
        # Project to standard hidden dimension
        x = self.final_projection(x)
        
        # Global attention pooling
        attention_scores = self.global_attention(x)
        attention_scores = F.softmax(attention_scores, dim=0)
        
        # Weighted sum based on attention
        graph_embedding = global_add_pool(x * attention_scores, batch)
        
        # Concatenate global features: strategy + io_type (both one-hot encoded)
        strategy_feature = data.strategy  # Shape: [batch_size, 2]
        io_type_feature = data.io_type    # Shape: [batch_size, 2]
        
        # Handle dimension issues
        if strategy_feature.dim() == 3:
            strategy_feature = strategy_feature.squeeze(-1)
        if io_type_feature.dim() == 3:
            io_type_feature = io_type_feature.squeeze(-1)
        
        # Concatenate: graph_embedding + strategy + io_type
        combined_embedding = torch.cat([graph_embedding, strategy_feature, io_type_feature], dim=1)

        # MLP prediction
        output = self.mlp(combined_embedding)
        return output


class FPGA_GNN_SumDecomp(nn.Module):
    """GATv2 backbone with a SUM-DECOMPOSITION (deep-sets) readout for depth extrapolation.

    Instead of pooling node embeddings then predicting a graph-level total, this predicts a
    NON-NEGATIVE per-node contribution (a per-layer cost), then SUMS over the nodes of each graph
    to get the design total. Because the per-node head is weight-shared and the aggregation is a
    plain sum, a model trained on shallow (1-2 layer) designs naturally extends to deeper (3-layer)
    designs — the additive inductive bias matches the physics (total area ~= sum of per-layer areas).

    forward(data) returns the LINEAR (un-normalized, non-negative) per-target total, shape
    [num_graphs, num_targets]. Train it with a log-space loss (see GNN/run_sumdecomp.py).
    """

    def __init__(self, node_feature_dim, num_targets, hidden_dim=256, num_gnn_layers=4,
                 num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = node_feature_dim
        for _ in range(num_gnn_layers):
            # concat=False averages heads -> output stays hidden_dim (keeps dims simple)
            self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=num_attention_heads,
                                        concat=False, dropout=dropout_rate))
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        # per-node head sees the node embedding + the graph globals (strategy[2] + io_type[2])
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, num_targets),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        strat = data.strategy
        io = data.io_type
        if strat.dim() == 3:
            strat = strat.squeeze(1)
        if io.dim() == 3:
            io = io.squeeze(1)
        # broadcast per-graph globals down to each node via the batch index
        strat_n = strat[batch]
        io_n = io[batch]
        h = torch.cat([x, strat_n, io_n], dim=1)

        logit = self.node_head(h).clamp(max=30.0)         # clamp to avoid exp overflow (float32-safe)
        per_node = torch.exp(logit)                       # [N, num_targets], positive LINEAR per-layer contribution
        total = global_add_pool(per_node, batch)          # [num_graphs, num_targets], LINEAR design total
        return total

    @torch.no_grad()
    def init_head_bias(self, mean_total, mean_layers):
        """Init the final head bias so the summed total starts near the data magnitude.
        mean_total: array-like [num_targets] of mean LINEAR target; mean_layers: avg #layers/design."""
        import numpy as _np
        b = _np.log(_np.maximum(_np.asarray(mean_total, dtype=_np.float64), 1e-6) / max(float(mean_layers), 1.0))
        self.node_head[-1].bias.copy_(torch.tensor(b, dtype=self.node_head[-1].bias.dtype))


class FPGA_GNN_SumDecompCorr(FPGA_GNN_SumDecomp):
    """Sum-decomposition PLUS a bounded, size-agnostic multiplicative correction:

        total = ( Σ_i c_i ) * ( 1 + eps * tanh( mean_i s_i ) )

    The additive part (Σ c_i) carries the magnitude and extrapolates with depth; the correction is a
    small per-target RELATIVE factor near 1, computed from a MEAN-pooled per-node interaction signal
    (bounded as #layers grows) and squashed by tanh, so it never re-introduces magnitude extrapolation.
    eps = exp(log_eps) is a small, learnable per-target scale; at init eps~=0.1 and g~=0 -> factor~=1,
    so the model STARTS as the plain sum-decomposition and only learns a correction if the data needs it.
    """

    def __init__(self, node_feature_dim, num_targets, hidden_dim=256, num_gnn_layers=4,
                 num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2, eps_init=0.1):
        super().__init__(node_feature_dim, num_targets, hidden_dim, num_gnn_layers,
                         num_attention_heads, mlp_hidden_dim, dropout_rate)
        corr_in = hidden_dim + 4  # node embedding + strategy(2) + io_type(2)
        self.corr_head = nn.Sequential(
            nn.Linear(corr_in, 64), nn.ReLU(), nn.Linear(64, num_targets),
        )
        self.log_eps = nn.Parameter(torch.full((num_targets,), float(np.log(eps_init))))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = F.elu(norm(conv(x, edge_index)))
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        strat = data.strategy
        io = data.io_type
        if strat.dim() == 3:
            strat = strat.squeeze(1)
        if io.dim() == 3:
            io = io.squeeze(1)
        h = torch.cat([x, strat[batch], io[batch]], dim=1)

        cost = torch.exp(self.node_head(h).clamp(max=30.0))      # [N, T] per-layer cost (≥0)
        base = global_add_pool(cost, batch)                      # [G, T] additive magnitude
        g = global_mean_pool(self.corr_head(h), batch)           # [G, T] MEAN-pooled interaction signal (size-agnostic)
        factor = 1.0 + torch.exp(self.log_eps) * torch.tanh(g)   # [G, T] bounded correction, ~1
        return base * factor


class FPGA_GNN_GATv2_SumDecomp(nn.Module):
    """The published lui-gnn GATv2 backbone (concat heads + residual connections + LayerNorm) paired
    with the SUM-DECOMPOSITION readout (per-layer exp cost, summed). Combines the strong GATv2 feature
    extractor with the additive readout that enables depth extrapolation. forward(data) returns the
    LINEAR per-target total [num_graphs, num_targets]; train with the same log-space loss as
    FPGA_GNN_SumDecomp (run_sumdecomp.py)."""

    def __init__(self, node_feature_dim, num_targets, hidden_dim=512, num_gnn_layers=5,
                 num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        in_dim = node_feature_dim
        gat_out = hidden_dim * num_attention_heads  # concat=True -> heads are concatenated
        for _ in range(num_gnn_layers):
            self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=num_attention_heads,
                                        concat=True, dropout=dropout_rate))
            self.norms.append(nn.LayerNorm(gat_out))
            self.res_proj.append(nn.Linear(in_dim, gat_out) if in_dim != gat_out else nn.Identity())
            in_dim = gat_out
        self.final_proj = nn.Linear(gat_out, hidden_dim)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, mlp_hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2), nn.ReLU(),
            nn.Linear(mlp_hidden_dim // 2, num_targets),
        )

    def _node_repr(self, data):
        """GATv2 backbone -> ([N, hidden_dim+4] node embedding+globals, batch index)."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm, res in zip(self.convs, self.norms, self.res_proj):
            h = F.elu(norm(conv(x, edge_index)) + res(x))            # GATv2 + residual + LayerNorm
            x = F.dropout(h, p=self.dropout_rate, training=self.training)
        x = F.elu(self.final_proj(x))                               # [N, hidden_dim]
        strat = data.strategy
        io = data.io_type
        if strat.dim() == 3:
            strat = strat.squeeze(1)
        if io.dim() == 3:
            io = io.squeeze(1)
        return torch.cat([x, strat[batch], io[batch]], dim=1), batch

    def forward(self, data):
        h, batch = self._node_repr(data)
        cost = torch.exp(self.node_head(h).clamp(max=30.0))         # [N, T] per-layer cost (≥0)
        return global_add_pool(cost, batch)                         # [G, T] additive total

    @torch.no_grad()
    def init_head_bias(self, mean_total, mean_layers):
        import numpy as _np
        b = _np.log(_np.maximum(_np.asarray(mean_total, dtype=_np.float64), 1e-6) / max(float(mean_layers), 1.0))
        self.node_head[-1].bias.copy_(torch.tensor(b, dtype=self.node_head[-1].bias.dtype))


class FPGA_GNN_GATv2_SumDecompCorr(FPGA_GNN_GATv2_SumDecomp):
    """The lui-gnn GATv2 backbone (residual, 512/5/5) + sum-decomposition + the bounded correction:
        total = (Σ cᵢ) × (1 + ε · tanh(meanᵢ sᵢ))
    Same additive-magnitude / bounded-relative-correction split as FPGA_GNN_SumDecompCorr, but on the
    strong GATv2 backbone. Starts ≈ the plain GATv2 sum-decomp (ε small)."""

    def __init__(self, node_feature_dim, num_targets, hidden_dim=512, num_gnn_layers=5,
                 num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2, eps_init=0.1):
        super().__init__(node_feature_dim, num_targets, hidden_dim, num_gnn_layers,
                         num_attention_heads, mlp_hidden_dim, dropout_rate)
        self.corr_head = nn.Sequential(
            nn.Linear(hidden_dim + 4, 64), nn.ReLU(), nn.Linear(64, num_targets),
        )
        self.log_eps = nn.Parameter(torch.full((num_targets,), float(np.log(eps_init))))

    def forward(self, data):
        h, batch = self._node_repr(data)
        cost = torch.exp(self.node_head(h).clamp(max=30.0))         # [N, T] per-layer cost
        base = global_add_pool(cost, batch)                         # [G, T] additive magnitude
        g = global_mean_pool(self.corr_head(h), batch)             # [G, T] MEAN-pooled interaction signal
        factor = 1.0 + torch.exp(self.log_eps) * torch.tanh(g)     # [G, T] bounded correction ~1
        return base * factor