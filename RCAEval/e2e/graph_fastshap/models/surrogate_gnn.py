"""
Surrogate GNN (HeteroSAGE) -- v(S) predicts SLI anomaly probability
given a masked subset S of metric nodes.

Architecture
------------
HeteroConv(SAGEConv) x n_layers  -->  global readout  -->  MLP  -->  sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GraphConv, global_mean_pool


class SurrogateGNN(nn.Module):
    """Heterogeneous GNN surrogate that maps masked metric features to a
    scalar SLI-anomaly probability.

    Parameters
    ----------
    in_dim : int
        Input feature dimension (per metric / service node).
    hidden_dim : int
        Hidden dimension for message-passing layers.
    n_layers : int
        Number of HeteroConv layers.
    dropout : float
        Dropout probability applied between layers.
    """

    def __init__(
        self,
        in_dim: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # -- heterogeneous message-passing layers --------------------------
        self.convs = nn.ModuleList()
        for layer_idx in range(n_layers):
            d_in = in_dim if layer_idx == 0 else hidden_dim
            conv_dict = {
                ("service", "calls", "service"): GraphConv(d_in, hidden_dim),
                ("service", "owns", "metric"): SAGEConv((d_in, d_in), hidden_dim),
                ("metric", "belongs_to", "service"): SAGEConv((d_in, d_in), hidden_dim),
            }
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # -- residual projections (for skip connections when n_layers > 2) --
        self.res_projs = nn.ModuleList()
        for layer_idx in range(n_layers):
            d_in = in_dim if layer_idx == 0 else hidden_dim
            if d_in != hidden_dim:
                self.res_projs.append(nn.Linear(d_in, hidden_dim))
            else:
                self.res_projs.append(nn.Identity())

        # -- readout MLP ---------------------------------------------------
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(self, hetero_data, s=None):
        """Run the surrogate on *hetero_data* with optional mask *s*.

        Parameters
        ----------
        hetero_data : HeteroData
            Graph produced by ``build_hetero_graph``.
        s : Tensor of shape ``(N_met,)`` or ``None``
            Binary mask per metric node.  ``None`` means keep all.

        Returns
        -------
        v : Tensor, scalar in [0, 1]
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

        # --- feature masking ----------------------------------------------
        if s is not None:
            # s: (N_met,) binary mask
            mask = s.unsqueeze(-1)  # (N_met, 1)
            x_dict["metric"] = x_dict["metric"] * mask

        # --- target masking (SLI node zeroed) -----------------------------
        sli_idx = getattr(hetero_data, "sli_idx", -1)
        if isinstance(sli_idx, int) and sli_idx >= 0:
            x_dict["metric"][sli_idx] = 0.0

        # --- edge masking (soft) for service-calls-service ----------------
        ss_key = ("service", "calls", "service")
        edge_weight_dict = {}
        if ss_key in hetero_data.edge_types and hasattr(hetero_data[ss_key], "edge_weight"):
            edge_weight_dict[ss_key] = hetero_data[ss_key].edge_weight

        # --- message passing ----------------------------------------------
        for layer_idx, conv in enumerate(self.convs):
            x_prev = x_dict.copy()
            x_dict = conv(x_dict, edge_index_dict, edge_weight_dict=edge_weight_dict)

            # ReLU + dropout + residual
            for ntype in x_dict:
                residual = self.res_projs[layer_idx](x_prev[ntype])
                x_dict[ntype] = F.relu(x_dict[ntype]) + residual
                x_dict[ntype] = F.dropout(
                    x_dict[ntype], p=self.dropout, training=self.training,
                )

        # --- global readout -----------------------------------------------
        # Mean-pool metric nodes and service nodes, then concatenate
        metric_pool = x_dict["metric"].mean(dim=0)   # (hidden,)
        service_pool = x_dict["service"].mean(dim=0)  # (hidden,)
        h = torch.cat([metric_pool, service_pool], dim=-1)  # (2*hidden,)

        v = torch.sigmoid(self.readout(h).squeeze(-1))
        return v
