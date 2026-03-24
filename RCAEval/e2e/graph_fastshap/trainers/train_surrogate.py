"""
Surrogate training loop.

For each training step we sample a Bernoulli mask *s* over metric nodes,
pass the masked graph through the surrogate, and optimise:

    L = L_pred(BCE) + mu * L_mono(hinge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import pandas as pd


def _compute_soft_label(anomal_df, sli, s, metric_cols, normal_mu, normal_sigma):
    """
    soft label = Weighted sum of deviations for retained metrics / Full deviations sum
    Intuition: Masking root cause -> soft label drops -> Surrogate learns causation
    """
    # Dynamic smoothing to prevent noise amplification on stable metrics
    dynamic_eps = float(np.median(normal_sigma))
    if dynamic_eps < 1e-5:
        dynamic_eps = 1e-5
        
    deviations = np.abs((anomal_df[metric_cols].mean().values - normal_mu) / (normal_sigma + dynamic_eps))
    masked_devs = deviations * s.cpu().numpy()
    full_dev_sum = deviations.sum() + dynamic_eps
    return float(masked_devs.sum() / full_dev_sum)


def train_surrogate(
    surrogate,
    hetero_data,
    normal_df: pd.DataFrame,
    anomal_df: pd.DataFrame,
    metric_cols: list,
    sli: str,
    n_epochs: int = 200,
    n_samples: int = 16,
    lr: float = 1e-3,
    mu: float = 0.1,
    device: str = "cpu",
):
    """Train the surrogate model on a single HeteroData graph.

    Parameters
    ----------
    surrogate : SurrogateGNN
    hetero_data : HeteroData
    normal_df : pd.DataFrame
    anomal_df : pd.DataFrame
    metric_cols : list
    sli : str
    n_epochs : int
    n_samples : int
        Number of random masks per epoch.
    lr : float
    mu : float
        Weight for the monotonicity constraint.
    device : str

    Returns
    -------
    surrogate : SurrogateGNN (trained, on *device*)
    """
    surrogate = surrogate.to(device)
    hetero_data = hetero_data.to(device)
    optimiser = torch.optim.Adam(surrogate.parameters(), lr=lr, weight_decay=1e-5)

    n_met = hetero_data["metric"].x.shape[0]

    # Calculate baseline stats for soft label computation
    normal_vals = normal_df[metric_cols].to_numpy(dtype=np.float64)
    normal_mu = np.mean(normal_vals, axis=0)
    normal_sigma = np.std(normal_vals, axis=0)

    # The anomal window is label=1 by definition for v_full
    label_full = torch.tensor([1.0], device=device)
    # Also create a label=0 for the "fully masked" case (all metrics hidden)
    label_empty = torch.tensor([0.0], device=device)

    for epoch in range(n_epochs):
        surrogate.train()
        epoch_loss = 0.0

        for _ in range(n_samples):
            # -- Sample mask s ~ Bernoulli(0.5) ----------------------------
            s = torch.bernoulli(0.5 * torch.ones(n_met, device=device))

            # -- Surrogate predictions -------------------------------------
            v_s = surrogate(hetero_data, s=s)
            v_full = surrogate(hetero_data, s=torch.ones(n_met, device=device))
            v_empty = surrogate(hetero_data, s=torch.zeros(n_met, device=device))

            # -- L_pred: BCE -----------------------------------------------
            # v_full should be close to 1 (anomalous), v_empty close to 0
            loss_pred = (
                F.binary_cross_entropy(v_full.unsqueeze(0), label_full)
                + F.binary_cross_entropy(v_empty.unsqueeze(0), label_empty)
            )
            
            # soft label computation
            sl_val = _compute_soft_label(anomal_df, sli, s, metric_cols, normal_mu, normal_sigma)
            label_s = torch.tensor([sl_val], device=device)
            loss_pred += F.binary_cross_entropy(v_s.unsqueeze(0), label_s)

            # -- L_mono: monotonicity hinge --------------------------------
            # s' = s with one extra metric masked (removing an anomalous signal
            # should not *increase* v).
            loss_mono = torch.tensor(0.0, device=device)
            nonzero_indices = s.nonzero(as_tuple=True)[0]
            if len(nonzero_indices) > 0:
                idx = nonzero_indices[torch.randint(len(nonzero_indices), (1,))]
                s_prime = s.clone()
                s_prime[idx] = 0.0
                v_s_prime = surrogate(hetero_data, s=s_prime)
                # v(s') should be <= v(s) + eps
                loss_mono = F.relu(v_s_prime - v_s + 1e-3)

            loss = loss_pred + mu * loss_mono

            optimiser.zero_grad()
            loss.backward()
            clip_grad_norm_(surrogate.parameters(), max_norm=5.0)
            optimiser.step()
            epoch_loss += loss.item()

    # Freeze
    for param in surrogate.parameters():
        param.requires_grad = False
    surrogate.eval()

    return surrogate
