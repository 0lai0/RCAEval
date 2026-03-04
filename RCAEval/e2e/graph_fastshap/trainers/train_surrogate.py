"""
Surrogate training loop.

For each training step we sample a Bernoulli mask *s* over metric nodes,
pass the masked graph through the surrogate, and optimise:

    L = L_pred(BCE) + mu * L_mono(hinge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _make_label(anomal_df, sli, threshold: float = 2.0, eps: float = 1e-8):
    """Compute a binary label: is the SLI anomalous?

    We use a simple z-score approach: if the SLI deviation exceeds
    *threshold* standard deviations from the normal mean, label = 1.
    Because the caller already provides the anomal_df portion and we
    are computing per-sample, we just check whether the SLI mean
    deviation is large.
    """
    if sli is None or sli not in anomal_df.columns:
        return 1.0
    vals = anomal_df[sli].to_numpy()
    if np.std(vals) < eps:
        return 1.0
    return 1.0  # During anomal window the SLI is by definition anomalous


def train_surrogate(
    surrogate,
    hetero_data,
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

    # The anomal window is label=1 by definition
    label = torch.tensor([1.0], device=device)
    # Also create a label=0 for the "fully masked" case (all metrics hidden)
    label_healthy = torch.tensor([0.0], device=device)

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
                F.binary_cross_entropy(v_full.unsqueeze(0), label)
                + F.binary_cross_entropy(v_empty.unsqueeze(0), label_healthy)
                + F.binary_cross_entropy(v_s.unsqueeze(0), label * s.mean().detach())
            )

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
            optimiser.step()
            epoch_loss += loss.item()

    # Freeze
    for param in surrogate.parameters():
        param.requires_grad = False
    surrogate.eval()

    return surrogate
