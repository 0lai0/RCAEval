"""
CausalSHAP: Shapley Value-based Root Cause Analysis with Causal Propagation

This module implements a non-additive value function for Shapley value computation
in microservice RCA scenarios. The key innovation is using anomaly propagation
along causal graphs instead of linear aggregation.

Key Components:
- PropagationValueFunction: Non-additive value function with max-based propagation
- sampling_shapley_with_propagation: Monte Carlo Shapley value estimation
- build_causal_graph_for_shapley: Multi-layer causal graph construction

"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.decomposition import PCA


class PropagationValueFunction:
    """
    Non-additive value function for Shapley computation based on anomaly propagation.
    
    The key non-linearity comes from:
    1. Max operation: node_anomaly = max(self_anomaly, upstream_propagated)
    2. Path blocking: masked nodes block propagation paths
    3. Decay propagation: anomalies decay along edges
    
    This captures redundancy and masking effects in microservice cascading failures.
    
    Parameters
    ----------
    causal_graph : nx.DiGraph
        Causal graph (must be DAG)
    anomaly_scores : Dict[str, float]
        Anomaly score for each service
    focus_node : str
        The focus node (usually the service with highest anomaly)
    decay : float, default=0.7
        Decay factor for propagation along edges
    propagation_mode : str, default="max"
        "max": take max of self and upstream (recommended)
        "attenuated_sum": upstream contributions have diminishing returns
    """

    def __init__(
        self,
        causal_graph: nx.DiGraph,
        anomaly_scores: Dict[str, float],
        focus_node: str,
        decay: float = 0.7,
        propagation_mode: str = "max",
    ):
        self.G = causal_graph
        self.anomaly_scores = anomaly_scores
        self.focus = focus_node
        self.decay = decay
        self.mode = propagation_mode

        self.services = sorted(list(causal_graph.nodes()))
        self.service_to_idx = {s: i for i, s in enumerate(self.services)}
        self.n_services = len(self.services)

        # Ensure DAG
        if not nx.is_directed_acyclic_graph(self.G):
            self.G = break_cycles_safe(self.G.copy())
            if not nx.is_directed_acyclic_graph(self.G):
                # If still has cycles, at least make it runnable
                pass

        try:
            self.topo_order = list(nx.topological_sort(self.G))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            # Fallback: use arbitrary order
            self.topo_order = self.services.copy()

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> float:
        """
        Compute value function v(S) where S is indicated by mask.
        
        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Anomaly score vector for all services
        mask : np.ndarray, shape (N,)
            Binary mask, 1 = service in subset S, 0 = masked out
        
        Returns
        -------
        float
            v(S) = final anomaly value at focus_node
        """
        if len(x) != self.n_services or len(mask) != self.n_services:
            raise ValueError(
                f"Input dimension mismatch: x={len(x)}, mask={len(mask)}, "
                f"expected {self.n_services}"
            )

        # Initialize node anomalies
        node_anomaly = {}
        for i, svc in enumerate(self.services):
            node_anomaly[svc] = float(x[i]) if mask[i] == 1.0 else 0.0

        # Propagate along topological order
        for node in self.topo_order:
            if node not in node_anomaly:
                continue
                
            incoming = []
            for parent in self.G.predecessors(node):
                if parent not in node_anomaly:
                    continue
                edge_data = self.G.get_edge_data(parent, node, {})
                edge_weight = edge_data.get("weight", 1.0)
                propagated = node_anomaly[parent] * self.decay * edge_weight
                incoming.append(propagated)

            self_val = node_anomaly[node]
            if incoming:
                if self.mode == "max":
                    # Non-linearity: max captures redundancy
                    node_anomaly[node] = max(self_val, max(incoming))
                elif self.mode == "attenuated_sum":
                    # Diminishing returns for multiple upstream sources
                    s = sorted(incoming, reverse=True)
                    upstream = sum(s[i] * (0.5 ** i) for i in range(len(s)))
                    node_anomaly[node] = max(self_val, upstream)
                else:
                    upstream = sum(incoming)
                    node_anomaly[node] = max(self_val, upstream)

        return float(node_anomaly.get(self.focus, 0.0))


def sampling_shapley_with_propagation(
    causal_graph: nx.DiGraph,
    anomaly_scores: Dict[str, float],
    focus_node: str,
    n_permutations: int = 1000,
    decay: float = 0.7,
    propagation_mode: str = "max",
    seed: Optional[int] = None,
    isolated_bonus_weight: float = 0.05,
) -> List[Tuple[str, float]]:
    """
    Compute Shapley values using Monte Carlo sampling with propagation value function.
    
    Algorithm:
    1. For each permutation, add services one by one
    2. Compute marginal contribution: v(S ∪ {i}) - v(S)
    3. Average over all permutations
    4. Add small bonus for isolated anomalous nodes
    
    Parameters
    ----------
    causal_graph : nx.DiGraph
        Causal graph
    anomaly_scores : Dict[str, float]
        Anomaly score for each service
    focus_node : str
        Focus node
    n_permutations : int, default=1000
        Number of permutations to sample
    decay : float, default=0.7
        Decay factor for propagation
    propagation_mode : str, default="max"
        Propagation mode
    seed : int, optional
        Random seed for reproducibility
    isolated_bonus_weight : float, default=0.05
        Bonus weight for isolated anomalous nodes
    
    Returns
    -------
    List[Tuple[str, float]]
        Ranked list of (service, shapley_value)
    """
    if seed is not None:
        np.random.seed(seed)

    value_fn = PropagationValueFunction(
        causal_graph=causal_graph,
        anomaly_scores=anomaly_scores,
        focus_node=focus_node,
        decay=decay,
        propagation_mode=propagation_mode,
    )

    services = value_fn.services
    n = value_fn.n_services
    x = np.array([anomaly_scores.get(s, 0.0) for s in services])
    shapley = np.zeros(n)
    empty_mask = np.zeros(n)
    v_empty = value_fn(x, empty_mask)

    # Monte Carlo sampling
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        mask = np.zeros(n, dtype=float)
        prev = v_empty

        for pos in perm:
            mask[pos] = 1.0
            curr = value_fn(x, mask)
            marginal = curr - prev
            shapley[pos] += marginal
            prev = curr

    shapley /= n_permutations

    # Bonus for isolated anomalous nodes (no causal path to focus)
    # This avoids penalizing truly anomalous but isolated services
    for i, svc in enumerate(services):
        if svc == focus_node:
            continue
        try:
            if not nx.has_path(causal_graph, svc, focus_node):
                shapley[i] += anomaly_scores.get(svc, 0) * isolated_bonus_weight
        except:
            pass

    ranked = [(services[i], float(shapley[i])) for i in range(n)]
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked


def extract_service_name(col: str, dataset: Optional[str] = None) -> str:
    """
    Extract service name from column name based on dataset format.
    
    Parameters
    ----------
    col : str
        Column name (e.g., "checkoutservice_cpu" or "ts-checkoutservice_cpu_usage")
    dataset : str, optional
        Dataset name for format-specific parsing
    
    Returns
    -------
    str
        Service name
    """
    if col == "time":
        return col
    
    if dataset in ["sock-shop", "sock-shop-1", "sock-shop-2", "my-sock-shop", "fse-ss", "re1-ss", "re2-ss", "re3-ss"]:
        if col.startswith("ts-"):
            col = col[3:]
        return col.split("_")[0]
    elif dataset in ["train-ticket", "mm-tt", "fse-tt", "re1-tt", "re2-tt", "re3-tt"]:
        if col.startswith("ts-"):
            col = col[3:]
        return col.split("_")[0]
    elif dataset in ["online-boutique", "mm-ob", "fse-ob", "RE2-OB", "RE2-SS", "re1-ob", "re2-ob", "re3-ob"]:
        if col.startswith("ts-"):
            col = col[3:]
        return col.split("_")[0]
    else:
        # Default: first part before underscore
        return col.split("_")[0]


def aggregate_to_service_level(data: pd.DataFrame, dataset: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate metrics to service level using PCA first component.
    
    This reduces dimensionality before causal discovery, significantly improving
    speed and statistical power when sample size is limited.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data with metric-level columns
    dataset : str, optional
        Dataset name for format-specific parsing
    
    Returns
    -------
    pd.DataFrame
        Service-level aggregated data
    """
    service_cols = defaultdict(list)
    for col in data.columns:
        if col == "time":
            continue
        svc = extract_service_name(col, dataset)
        service_cols[svc].append(col)

    result = {}
    if "time" in data.columns:
        result["time"] = data["time"].values

    for svc, cols in sorted(service_cols.items()):
        # Get columns that have at least some non-NaN values
        valid_cols = [c for c in cols if c in data.columns]
        if not valid_cols:
            result[svc] = np.zeros(len(data))
            continue
            
        vals = data[valid_cols].values
        
        # Handle single metric case
        if vals.shape[1] == 0:
            result[svc] = np.zeros(len(data))
        elif vals.shape[1] == 1:
            result[svc] = np.nan_to_num(vals[:, 0], nan=0.0)
        else:
            # Multiple metrics: use PCA first component
            # Fill NaN with 0 for PCA
            vals = np.nan_to_num(vals, nan=0.0)
            
            # Check if all columns are constant
            if np.all(np.std(vals, axis=0) == 0):
                result[svc] = vals[:, 0] if vals.shape[1] > 0 else np.zeros(len(data))
            else:
                try:
                    pca = PCA(n_components=1)
                    result[svc] = pca.fit_transform(vals)[:, 0]
                except Exception:
                    # Fallback to mean if PCA fails
                    result[svc] = np.nanmean(vals, axis=1)
                    result[svc] = np.nan_to_num(result[svc], nan=0.0)

    return pd.DataFrame(result)


def adaptive_max_conds_dim(n_samples: int, n_variables: int) -> int:
    """
    Adaptive max conditioning set size for PCMCI.
    
    Heuristic: need ~10 samples per parameter in the conditional 
    independence test. For partial correlation with d conditions,
    we estimate d+2 parameters, so need ~10*(d+2) samples.
    
    Also capped by n_variables - 2 (theoretical max).
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    n_variables : int
        Number of variables
    
    Returns
    -------
    int
        Adaptive max_conds_dim value
    
    Examples
    --------
    >>> adaptive_max_conds_dim(50, 14)   # 50 samples, 14 services
    3
    >>> adaptive_max_conds_dim(100, 14)  # 100 samples, 14 services
    8
    >>> adaptive_max_conds_dim(200, 14)  # 200 samples, 14 services
    12
    """
    # Solve: n_samples >= 10 * (d + 2)  →  d <= n_samples/10 - 2
    from_samples = max(1, int(n_samples / 10) - 2)
    from_variables = max(1, n_variables - 2)
    return min(from_samples, from_variables)


def _adj_to_graph(
    adj: np.ndarray,
    node_names: List[str],
    dataset: Optional[str] = None,
    agg: str = "max",
) -> nx.DiGraph:
    """
    Convert adjacency matrix to NetworkX graph at service level.
    
    Aggregates metric-level edges to service-level using max (not sum)
    to avoid bias from services with more metrics.
    
    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix at metric level
    node_names : List[str]
        Metric names
    dataset : str, optional
        Dataset name
    agg : str, default="max"
        Aggregation method: "max", "mean", or "sum"
    
    Returns
    -------
    nx.DiGraph
        Service-level causal graph
    """
    G = nx.DiGraph()
    service_to_metrics = defaultdict(list)
    
    for i, col in enumerate(node_names):
        if col == "time":
            continue
        svc = extract_service_name(col, dataset)
        service_to_metrics[svc].append(i)

    services = sorted(list(service_to_metrics.keys()))
    G.add_nodes_from(services)

    edge_weights = defaultdict(list)
    for i, col_i in enumerate(node_names):
        if col_i == "time":
            continue
        svc_i = extract_service_name(col_i, dataset)
        for j, col_j in enumerate(node_names):
            if col_j == "time" or i == j:
                continue
            svc_j = extract_service_name(col_j, dataset)
            if svc_i == svc_j:
                continue
            if adj[i, j] > 0:
                edge_weights[(svc_i, svc_j)].append(float(adj[i, j]))

    for (si, sj), weights in edge_weights.items():
        if agg == "max":
            w = max(weights)
        elif agg == "mean":
            w = sum(weights) / len(weights)
        else:  # sum
            w = sum(weights)
        G.add_edge(si, sj, weight=w, source="metric_adj")

    return G


def _build_directed_correlation_graph(
    data: pd.DataFrame,
    threshold: float = 0.3,
    dataset: Optional[str] = None,
    max_lag: int = 5,
) -> nx.DiGraph:
    """
    Build directed correlation graph using lag correlation.
    
    Uses cross-correlation at different lags to determine edge direction,
    avoiding bidirectional edges that create cycles.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    threshold : float, default=0.3
        Correlation threshold
    dataset : str, optional
        Dataset name
    max_lag : int, default=5
        Maximum lag to consider
    
    Returns
    -------
    nx.DiGraph
        Directed correlation graph
    """
    G = nx.DiGraph()
    services = []
    svc_series = {}

    # Data is now service-level (after aggregation), so columns are service names
    for col in data.columns:
        if col == "time":
            continue
        svc = col  # Column name is already service name after aggregation
        services.append(svc)
        svc_series[svc] = np.nan_to_num(data[col].values, nan=0.0)
    
    services = sorted(services)
    G.add_nodes_from(services)

    # Lag correlation to determine direction
    for i, si in enumerate(services):
        if si not in svc_series:
            continue
        for j, sj in enumerate(services):
            if i >= j or sj not in svc_series:
                continue

            a, b = svc_series[si], svc_series[sj]
            if len(a) < max_lag * 2 or len(b) < max_lag * 2:
                continue

            best_corr = 0.0
            best_dir = None

            for lag in range(1, min(max_lag + 1, len(a) // 3)):
                # si leads sj (si -> sj)
                if len(a) > lag:
                    try:
                        c1 = np.corrcoef(a[:-lag], b[lag:])[0, 1]
                        if not np.isnan(c1) and abs(c1) > abs(best_corr):
                            best_corr = c1
                            best_dir = (si, sj)
                    except:
                        pass

                # sj leads si (sj -> si)
                if len(b) > lag:
                    try:
                        c2 = np.corrcoef(b[:-lag], a[lag:])[0, 1]
                        if not np.isnan(c2) and abs(c2) > abs(best_corr):
                            best_corr = c2
                            best_dir = (sj, si)
                    except:
                        pass

            if abs(best_corr) > threshold and best_dir is not None:
                G.add_edge(
                    best_dir[0], best_dir[1],
                    weight=abs(best_corr),
                    source="directed_correlation",
                )

    return G


def _fuse_graphs(
    graphs: List[Tuple[nx.DiGraph, float]],
    target_avg_degree: float = 3.0,
) -> nx.DiGraph:
    """
    Fuse multiple graphs with weighted scoring.
    
    Prioritizes edges supported by multiple layers, then by weight.
    
    Parameters
    ----------
    graphs : List[Tuple[nx.DiGraph, float]]
        List of (graph, weight) tuples
    target_avg_degree : float, default=3.0
        Target average degree for the fused graph
    
    Returns
    -------
    nx.DiGraph
        Fused graph
    """
    if not graphs:
        return nx.DiGraph()

    all_nodes = set()
    for G, _ in graphs:
        all_nodes.update(G.nodes())
    if not all_nodes:
        return nx.DiGraph()

    fused = nx.DiGraph()
    fused.add_nodes_from(all_nodes)

    edge_scores = defaultdict(float)
    edge_support = defaultdict(int)  # Number of layers supporting this edge
    
    for G, w in graphs:
        for u, v, d in G.edges(data=True):
            ew = d.get("weight", 1.0)
            edge_scores[(u, v)] += w * ew
            edge_support[(u, v)] += 1

    # Sort by (support count, weight) - prioritize multi-layer support
    sorted_edges = sorted(
        edge_scores.items(),
        key=lambda x: (edge_support[x[0]], x[1]),
        reverse=True,
    )

    n_nodes = len(all_nodes)
    target_edges = int(n_nodes * target_avg_degree)
    selected = sorted_edges[:target_edges]

    for (u, v), score in selected:
        fused.add_edge(
            u, v,
            weight=score,
            n_sources=edge_support[(u, v)],
            source="fused",
        )

    return fused


def _normalize_edge_weights(G: nx.DiGraph) -> nx.DiGraph:
    """
    Normalize edge weights to [0.1, 1.0].
    
    This prevents scale inconsistency from different causal discovery methods
    (e.g., Granger returns 0/1, PCMCI returns p-values, correlation returns [0,1]).
    
    Parameters
    ----------
    G : nx.DiGraph
        Graph with potentially inconsistent edge weights
    
    Returns
    -------
    nx.DiGraph
        Graph with normalized edge weights
    """
    G = G.copy()
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    if not weights:
        return G
    
    max_w, min_w = max(weights), min(weights)
    rng = max_w - min_w if max_w != min_w else 1.0
    
    for _, _, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        # Normalize to [0.1, 1.0] to avoid zero weights
        d["weight"] = 0.1 + 0.9 * (w - min_w) / rng
    
    return G


def break_cycles_safe(G: nx.DiGraph) -> nx.DiGraph:
    """
    Safely break cycles in graph by removing weakest edges.
    
    Uses nx.find_cycle (O(V+E)) instead of nx.simple_cycles (exponential).
    
    Parameters
    ----------
    G : nx.DiGraph
        Possibly cyclic graph
    
    Returns
    -------
    nx.DiGraph
        Acyclic graph (DAG)
    """
    G = G.copy()
    for _ in range(100):  # Max iterations to prevent infinite loop
        try:
            cycle = nx.find_cycle(G, orientation="original")
        except nx.NetworkXNoCycle:
            break
        # Remove weakest edge in cycle
        weakest = min(
            cycle,
            key=lambda e: G[e[0]][e[1]].get("weight", 1.0)
        )
        G.remove_edge(weakest[0], weakest[1])
    return G


def _normalize_graph_nodes(
    G: nx.DiGraph, target_nodes: List[str]
) -> nx.DiGraph:
    """
    Normalize graph node names to match target node list.
    
    Parameters
    ----------
    G : nx.DiGraph
        Graph with possibly inconsistent node names
    target_nodes : List[str]
        Target node names
    
    Returns
    -------
    nx.DiGraph
        Graph with normalized node names
    """
    normalized = nx.DiGraph()
    normalized.add_nodes_from(target_nodes)
    mapping = {}
    
    for node in G.nodes():
        for t in target_nodes:
            if node == t or node.replace("-", "") == t.replace("-", ""):
                mapping[node] = t
                break
        if node not in mapping:
            mapping[node] = node
    
    for u, v, d in G.edges(data=True):
        un = mapping.get(u, u)
        vn = mapping.get(v, v)
        if un in target_nodes and vn in target_nodes:
            normalized.add_edge(un, vn, **d)
    
    return normalized


def build_causal_graph_for_shapley(
    data: pd.DataFrame,
    anomaly_scores: Dict[str, float],
    focus_node: str,
    dataset: Optional[str] = None,
    trace_graph: Optional[nx.DiGraph] = None,
    target_avg_degree: float = 3.0,
    causal_method: str = "pcmci",
    pcmci_alpha: float = 0.2,
    granger_alpha: float = 0.05,
    corr_threshold: float = 0.3,
) -> nx.DiGraph:
    """
    Build causal graph optimized for Shapley value computation.
    
    Multi-layer fusion strategy:
    - Layer 1: PCMCI or Granger causality (time-series causal discovery)
    - Layer 2: Trace graph (service call graph, if available)
    - Layer 3: Directed correlation (fallback)
    
    Key optimizations:
    1. Ensures connectivity (all anomalous nodes should have paths)
    2. Controls average degree (avoids too sparse/dense graphs)
    3. Ensures DAG property (required for propagation)
    4. Normalizes edge weights (consistent scale across layers)
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data (preferably anomalous period)
    anomaly_scores : Dict[str, float]
        Anomaly scores for services
    focus_node : str
        Focus node (highest anomaly)
    dataset : str, optional
        Dataset name for format-specific parsing
    trace_graph : nx.DiGraph, optional
        Service call graph from traces
    target_avg_degree : float, default=3.0
        Target average degree
    causal_method : str, default="pcmci"
        "pcmci", "granger", or "both"
    pcmci_alpha : float, default=0.2
        PCMCI significance level
    granger_alpha : float, default=0.05
        Granger p-value threshold
    corr_threshold : float, default=0.3
        Correlation threshold
    
    Returns
    -------
    nx.DiGraph
        Causal graph (DAG with normalized edge weights)
    """
    from RCAEval.graph_construction.granger import granger
    from RCAEval.graph_construction.pcmci import pcmci

    # ========== OPTIMIZATION 1: Aggregate to Service Level First ==========
    # This reduces dimensionality from ~70 metrics to ~14 services before PCMCI,
    # significantly improving speed (5-10x) and statistical power
    n_metrics_original = len([c for c in data.columns if c != "time"])
    svc_data = aggregate_to_service_level(data, dataset)
    services = [c for c in svc_data.columns if c != "time"]
    n_services = len(services)
    
    if not services:
        return nx.DiGraph()
    
    # Log optimization effect
    print(f"[CausalSHAP] Service-level aggregation: {n_metrics_original} metrics → {n_services} services")
    
    # ========== OPTIMIZATION 2: Adaptive max_conds_dim ==========
    # Automatically adjust based on sample size and number of variables
    n_samples = len(svc_data)
    n_vars = len(services)
    max_conds = adaptive_max_conds_dim(n_samples, n_vars)
    print(f"[CausalSHAP] Adaptive max_conds_dim: {max_conds} (n_samples={n_samples}, n_vars={n_vars})")

    layers = []

    # Layer 1: Time-series causal discovery (on service-level data)
    if causal_method in ("pcmci", "both"):
        try:
            # Run PCMCI on service-level data (much faster than metric-level)
            adj = pcmci(svc_data, alpha=pcmci_alpha, max_conds_dim=max_conds)
            
            # Convert adjacency matrix directly to graph (already service-level)
            g = nx.DiGraph()
            g.add_nodes_from(services)
            
            # adj is already service-level, so direct mapping
            for i, si in enumerate(services):
                for j, sj in enumerate(services):
                    if i != j and adj[i, j] > 0:
                        g.add_edge(si, sj, weight=float(adj[i, j]), source="pcmci")
            
            layers.append((g, 1.0))
        except Exception as e:
            print(f"Warning: PCMCI failed: {e}")

    if causal_method in ("granger", "both"):
        try:
            # Granger also runs on service-level data for consistency
            adj = granger(svc_data, p_val_threshold=granger_alpha)
            
            # Convert to graph (already service-level)
            g = nx.DiGraph()
            g.add_nodes_from(services)
            
            for i, si in enumerate(services):
                for j, sj in enumerate(services):
                    if i != j and adj[i, j] > 0:
                        g.add_edge(si, sj, weight=float(adj[i, j]), source="granger")
            
            # Lower weight if both methods used to avoid double-counting
            w = 0.3 if causal_method == "both" else 1.0
            layers.append((g, w))
        except Exception as e:
            print(f"Warning: Granger failed: {e}")

    # Layer 2: Trace graph (if available)
    if trace_graph is not None:
        tg = _normalize_graph_nodes(trace_graph, services)
        layers.append((tg, 0.6))

    # Layer 3: Directed correlation (fallback, also on service-level)
    try:
        corr_g = _build_directed_correlation_graph(
            svc_data, threshold=corr_threshold, dataset=dataset
        )
        layers.append((corr_g, 0.3))
    except Exception as e:
        print(f"Warning: Correlation graph failed: {e}")

    # Fuse layers
    G = _fuse_graphs(layers, target_avg_degree=target_avg_degree)

    # Ensure DAG
    G = break_cycles_safe(G)

    # Normalize edge weights to [0.1, 1.0]
    G = _normalize_edge_weights(G)

    return G
