"""
Three-part FastSHAP loss for the Explainer.

    Loss = L_wls + gamma * L_eff + lambda_ * L_asym

- **WLS** (Weighted Least Squares): paired-sampling consistency with the
  frozen surrogate.
- **Efficiency Penalty**: ``sum(phi) == v(1) - v(0)``.
- **Asymmetric RCA Loss**: suppress phi for low-prior (downstream victim)
  nodes.
"""

import torch
import torch.nn.functional as F


def fastshap_loss(
    phi_hat,
    s,
    s_bar,
    v_0,
    v_1,
    v_s,
    v_s_bar,
    prior=None,
    gamma: float = 0.01,
    lambda_: float = 1.0,
    alpha: float = 0.5,
):
    """Compute the three-part FastSHAP loss.

    Parameters
    ----------
    phi_hat : Tensor, shape ``(N_met,)``
        Predicted Shapley values from the explainer.
    s, s_bar : Tensor, shape ``(N_met,)``
        Paired binary masks (``s_bar = 1 - s``).
    v_0, v_1, v_s, v_s_bar : Tensor (scalar each)
        Surrogate values under empty, full, s, and s_bar masks.
    prior : Tensor, shape ``(N_met,)`` or None
        Graph prior (e.g. PageRank scores).  Higher = more likely root cause.
    gamma : float
        Efficiency penalty weight.
    lambda_ : float
        Asymmetric RCA loss weight.

    Returns
    -------
    loss : Tensor (scalar)
    """

    # --- Scale for Relative Error ----------------------------------------
    # Detach to prevent Explainer from gaming the denominator.
    # Use + 1e-3 (not 1e-8) to avoid secondary explosions when v(1) ≈ v(0).
    scale = ((v_1 - v_0).abs() + 1e-3).detach()

    # --- WLS (Weighted Least Squares) ------------------------------------
    # v(s)     approx = v(0) + s . phi
    # v(s_bar) approx = v(0) + s_bar . phi
    pred_s = v_0 + torch.dot(s, phi_hat)
    pred_s_bar = v_0 + torch.dot(s_bar, phi_hat)
    l_wls = ((v_s - pred_s) / scale) ** 2 + ((v_s_bar - pred_s_bar) / scale) ** 2

    # --- Efficiency Penalty ----------------------------------------------
    l_eff = ((v_1 - v_0 - phi_hat.sum()) / scale) ** 2

    # --- Asymmetric RCA Loss & Contrastive Ranking Loss ------------------
    n_met = len(phi_hat)
    BASELINE_N = 50.0  # Baseline node count to preserve penalty strength on small graphs

    if prior is not None:
        # Low-prior nodes get penalised more: weight = (1 - prior[i])
        weight = 1.0 - prior
        l_asym = (weight * phi_hat ** 2).sum() / n_met * BASELINE_N
        
        # Contrastive Ranking Loss: Push high-prior nodes' phi above low-prior nodes
        _, indices = torch.sort(prior, descending=True)
        # Define Top-K as potential root causes
        n_top = max(1, min(5, len(prior) // 2))
        top_idx = indices[:n_top]
        bottom_idx = indices[n_top:]
        if len(bottom_idx) > 0:
            phi_top = phi_hat[top_idx].unsqueeze(1) # (n_top, 1)
            phi_bottom = phi_hat[bottom_idx].unsqueeze(0) # (1, n_bottom)
            # Margin ranking loss: ReLU(margin - (phi_top - phi_bottom))
            margin = 0.1
            l_rank = F.relu(margin - (phi_top - phi_bottom)).mean()
        else:
            l_rank = torch.tensor(0.0, device=phi_hat.device)
    else:
        l_asym = torch.tensor(0.0, device=phi_hat.device)
        l_rank = torch.tensor(0.0, device=phi_hat.device)

    loss = l_wls + gamma * l_eff + lambda_ * l_asym + alpha * l_rank
    return loss
