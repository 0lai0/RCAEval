"""
Inference: phi_hat -> ranked metric list.
"""

import torch


def phi_to_ranks(phi_hat, metric_cols):
    """Convert Shapley values to a ranked list of metric column names.

    Parameters
    ----------
    phi_hat : Tensor, shape ``(N_met,)``
    metric_cols : list[str]

    Returns
    -------
    ranks : list[str]
        Metric column names sorted by descending Shapley value.
        Positive-phi metrics come first; the rest are appended at the end.
    """
    scores = {}
    for i, col in enumerate(metric_cols):
        val = phi_hat[i]
        scores[col] = val.item() if hasattr(val, 'item') else float(val)

    # Positive phi first, sorted descending
    positive = [(c, v) for c, v in scores.items() if v > 0]
    positive.sort(key=lambda x: x[1], reverse=True)
    ranks = [c for c, _ in positive]

    # Append remaining (phi <= 0) sorted by descending value (least negative first)
    remaining = [(c, v) for c, v in scores.items() if v <= 0]
    remaining.sort(key=lambda x: x[1], reverse=True)
    ranks.extend([c for c, _ in remaining])

    return ranks
