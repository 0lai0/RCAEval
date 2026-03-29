"""
Explainer training loop (Paired Sampling with a frozen Surrogate).

For each iteration we:
1. Forward the **full** graph through the Explainer -> phi_hat.
2. Sample mask s ~ Bernoulli(0.5), compute s_bar = 1 - s.
3. Evaluate the **frozen** Surrogate at v(0), v(1), v(s), v(s_bar).
4. Compute the three-part FastSHAP loss and back-propagate through the
   Explainer only.
"""

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import networkx as nx

from .custom_loss import fastshap_loss


def _compute_prior(hetero_data):
    """Compute a graph-based prior for each metric node.

    We build the **reverse** call graph G_call^rev from the ``calls`` edges
    and compute PageRank on it.  In the reverse graph, upstream (root-cause)
    services receive more in-edges from their downstream dependents, yielding
    higher PageRank scores — consistent with fault-origin prioritisation.

    Returns
    -------
    prior : Tensor, shape ``(N_met,)``
    """
    n_met = hetero_data["metric"].x.shape[0]
    service_names = hetero_data["service"].names
    n_srv = len(service_names)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_srv))

    ss_key = ("service", "calls", "service")
    if ss_key in hetero_data.edge_types:
        ei = hetero_data[ss_key].edge_index
        for k in range(ei.shape[1]):
            src, dst = int(ei[0, k]), int(ei[1, k])
            # Reverse edge: effect→cause, so upstream gets higher PageRank
            G.add_edge(dst, src)

    pr = nx.pagerank(G, alpha=0.85)

    # Map service-level PageRank to metric nodes
    col_names = hetero_data["metric"].col_names
    col_to_service = hetero_data["metric"].col_to_service
    svc_to_idx = {s: i for i, s in enumerate(service_names)}

    prior = torch.zeros(n_met)
    for mi, col in enumerate(col_names):
        svc = col_to_service[col]
        si = svc_to_idx.get(svc, 0)
        prior[mi] = pr.get(si, 1.0 / n_srv)

    # Normalise to [0, 1]
    pmin, pmax = prior.min(), prior.max()
    if pmax - pmin > 1e-12:
        prior = (prior - pmin) / (pmax - pmin)
    else:
        prior = torch.ones(n_met) / n_met

    return prior


def train_explainer(
    explainer,
    surrogate,
    hetero_data,
    n_epochs: int = 300,
    n_samples: int = 16,
    lr: float = 1e-3,
    gamma: float = 0.01,
    lambda_: float = 1.0,
    device: str = "cpu",
):
    """Train the Explainer with a frozen Surrogate.

    Parameters
    ----------
    explainer : ExplainerGNN
    surrogate : SurrogateGNN (frozen)
    hetero_data : HeteroData
    n_epochs, n_samples, lr : training hyper-parameters
    gamma : float
        Efficiency penalty weight.
    lambda_ : float
        Asymmetric RCA loss weight.
    device : str

    Returns
    -------
    explainer : ExplainerGNN (trained)
    """
    explainer = explainer.to(device)
    surrogate = surrogate.to(device)
    hetero_data = hetero_data.to(device)

    optimiser = torch.optim.Adam(explainer.parameters(), lr=lr, weight_decay=1e-5)
    n_met = hetero_data["metric"].x.shape[0]

    prior = _compute_prior(hetero_data).to(device)

    for epoch in range(n_epochs):
        explainer.train()
        
        # Dynamic annealing for Contrastive Ranking Loss (warmup over first 50% of epochs)
        # Avoids overriding WLS early in training
        warmup_fraction = min(1.0, epoch / max(1, n_epochs * 0.5))
        dynamic_alpha = 0.5 * warmup_fraction

        for _ in range(n_samples):
            # 1. Explainer forward (full graph, no masking)
            phi_hat = explainer(hetero_data)  # (N_met,)

            # 2. Sample paired masks
            s = torch.bernoulli(0.5 * torch.ones(n_met, device=device))
            s_bar = 1.0 - s

            # 3. Frozen surrogate evaluations
            with torch.no_grad():
                v_0 = surrogate(hetero_data, s=torch.zeros(n_met, device=device))
                v_1 = surrogate(hetero_data, s=torch.ones(n_met, device=device))
                v_s = surrogate(hetero_data, s=s)
                v_s_bar = surrogate(hetero_data, s=s_bar)

            # 4. Loss
            loss = fastshap_loss(
                phi_hat, s, s_bar, v_0, v_1, v_s, v_s_bar,
                prior=prior, gamma=gamma, lambda_=lambda_, alpha=dynamic_alpha,
            )

            optimiser.zero_grad()
            loss.backward()
            clip_grad_norm_(explainer.parameters(), max_norm=5.0)
            optimiser.step()

    explainer.eval()
    return explainer
