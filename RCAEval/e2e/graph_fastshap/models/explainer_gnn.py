"""
Explainer GNN (HeteroGAT) -- outputs per-metric Shapley values phi_hat.

Architecture
------------
HeteroConv(GATConv) x n_layers  -->  Linear (no activation)  -->  phi_hat

The last layer is intentionally a *linear* projection without ReLU because
Shapley values can be negative.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv


class ExplainerGNN(nn.Module):
    """Heterogeneous GAT explainer that outputs per-metric Shapley values.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden dimension for GAT layers.
    n_layers : int
        Number of HeteroConv(GAT) layers.
    n_heads : int
        Number of attention heads (used in intermediate layers).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for layer_idx in range(n_layers):
            d_in = in_dim if layer_idx == 0 else hidden_dim
            conv_dict = {
                ("service", "calls", "service"): GATConv(
                    d_in, hidden_dim // n_heads, heads=n_heads, concat=True,
                    add_self_loops=False,
                ),
                ("service", "owns", "metric"): GATConv(
                    (d_in, d_in), hidden_dim // n_heads, heads=n_heads,
                    concat=True, add_self_loops=False,
                ),
                ("metric", "belongs_to", "service"): GATConv(
                    (d_in, d_in), hidden_dim // n_heads, heads=n_heads,
                    concat=True, add_self_loops=False,
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # Residual projections
        self.res_projs = nn.ModuleList()
        for layer_idx in range(n_layers):
            d_in = in_dim if layer_idx == 0 else hidden_dim
            if d_in != hidden_dim:
                self.res_projs.append(nn.Linear(d_in, hidden_dim))
            else:
                self.res_projs.append(nn.Identity())

        # Final linear head: metric node embeddings -> scalar phi per node
        # NO activation -- Shapley values can be positive or negative
        self.phi_head = nn.Linear(hidden_dim, 1)

    def forward(self, hetero_data):
        """Compute per-metric Shapley values.

        Parameters
        ----------
        hetero_data : HeteroData

        Returns
        -------
        phi_hat : Tensor, shape ``(N_met,)``
        """
        x_dict = {
            "metric": hetero_data["metric"].x.clone(),
            "service": hetero_data["service"].x.clone(),
        }

        edge_index_dict = {}
        for edge_type in [
            ("service", "calls", "service"),
            ("service", "owns", "metric"),
            ("metric", "belongs_to", "service"),
        ]:
            if edge_type in hetero_data.edge_types:
                edge_index_dict[edge_type] = hetero_data[edge_type].edge_index

        for layer_idx, conv in enumerate(self.convs):
            x_prev = x_dict.copy()
            x_dict = conv(x_dict, edge_index_dict)

            for ntype in x_dict:
                residual = self.res_projs[layer_idx](x_prev[ntype])
                x_dict[ntype] = F.elu(x_dict[ntype]) + residual
                x_dict[ntype] = F.dropout(
                    x_dict[ntype], p=self.dropout, training=self.training,
                )

        # Only output for metric nodes
        phi_hat = self.phi_head(x_dict["metric"]).squeeze(-1)  # (N_met,)
        return phi_hat
