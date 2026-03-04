"""
graph_fastshap -- end-to-end entry point for RCAEval.

This function is called by ``main.py`` via ``globals()[args.method]``.
It performs:
  1. Preprocessing + normal/anomal split
  2. HeteroData graph construction (M1)
  3. Surrogate training (M2)
  4. Explainer training (M3)
  5. Single-pass inference -> ranked metric list (M4)
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from RCAEval.io.time_series import preprocess

from .graph_fastshap.data_pipeline.build_hetero_graph import build_hetero_graph
from .graph_fastshap.models.surrogate_gnn import SurrogateGNN
from .graph_fastshap.models.explainer_gnn import ExplainerGNN
from .graph_fastshap.trainers.train_surrogate import train_surrogate
from .graph_fastshap.trainers.train_explainer import train_explainer
from .graph_fastshap.inference import phi_to_ranks


def graph_fastshap(
    data,
    inject_time=None,
    dataset=None,
    sli=None,
    **kwargs,
):
    """Graph-guided FastSHAP root-cause analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Combined normal + anomalous data with a ``time`` column.
    inject_time : int
        Unix timestamp separating normal from anomalous data.
    dataset : str
        Dataset identifier (e.g. ``"online-boutique"``).
    sli : str
        SLI column name (e.g. ``"frontend_latency"``).

    Returns
    -------
    dict
        ``{"ranks": List[str]}`` -- metric column names ranked by
        descending Shapley value.
    """

    # ==================================================================
    # 1. Preprocessing + split
    # ==================================================================
    if inject_time is not None and "time" in data.columns:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        # Fallback: first half normal, second half anomal
        mid = len(data) // 2
        normal_df = data.iloc[:mid]
        anomal_df = data.iloc[mid:]

    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=False,
    )
    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=False,
    )

    # Intersect columns (some may be dropped by preprocess)
    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    metric_cols = [c for c in intersects if c != "time"]

    if len(metric_cols) == 0:
        return {"ranks": []}

    # ==================================================================
    # 2. Build HeteroData graph (M1)
    # ==================================================================
    hetero_data = build_hetero_graph(
        normal_df, anomal_df, metric_cols,
        dataset=dataset, sli=sli,
    )

    # ==================================================================
    # 3. Hyper-parameters (from env or defaults)
    # ==================================================================
    device = "cpu"
    feat_dim = hetero_data["metric"].x.shape[1]
    hidden_dim = int(os.environ.get("GFS_HIDDEN_DIM", 64))
    n_layers = int(os.environ.get("GFS_N_LAYERS", 3))
    surr_epochs = int(os.environ.get("GFS_SURR_EPOCHS", 200))
    expl_epochs = int(os.environ.get("GFS_EXPL_EPOCHS", 300))
    n_samples = int(os.environ.get("GFS_N_SAMPLES", 16))
    gamma = float(os.environ.get("GFS_GAMMA", 0.01))
    lambda_ = float(os.environ.get("GFS_LAMBDA", 1.0))
    mu = float(os.environ.get("GFS_MU", 0.1))

    # ==================================================================
    # 4. Train Surrogate (M2)
    # ==================================================================
    surrogate = SurrogateGNN(
        in_dim=feat_dim, hidden_dim=hidden_dim, n_layers=n_layers,
    )
    surrogate = train_surrogate(
        surrogate, hetero_data,
        n_epochs=surr_epochs, n_samples=n_samples,
        lr=1e-3, mu=mu, device=device,
    )

    # ==================================================================
    # 5. Train Explainer (M3)
    # ==================================================================
    explainer = ExplainerGNN(
        in_dim=feat_dim, hidden_dim=hidden_dim, n_layers=n_layers,
    )
    explainer = train_explainer(
        explainer, surrogate, hetero_data,
        n_epochs=expl_epochs, n_samples=n_samples,
        lr=1e-3, gamma=gamma, lambda_=lambda_, device=device,
    )

    # ==================================================================
    # 6. Inference (M4)
    # ==================================================================
    explainer.eval()
    with torch.no_grad():
        phi_hat = explainer(hetero_data.to(device))

    ranks = phi_to_ranks(phi_hat, metric_cols)

    return {"ranks": ranks}
