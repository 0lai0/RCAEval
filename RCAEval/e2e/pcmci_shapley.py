import numpy as np
import pandas as pd
import networkx as nx
from typing import Any, Dict
import logging
import os
import time

from RCAEval.e2e import rca
from RCAEval.io.time_series import preprocess

from .pcmci_shapley_modules import (
    PCMCIShapleyConfig,
    preprocessing as prep_mod,
    node_isolation as iso_mod,
    pcmci_local as pcmci_mod,
    causal_discovery as causal_mod,
    edge_fusion as fuse_mod,
    propagation as prop_mod,
    shapley as shap_mod,
    scoring as score_mod,
    utils as utils_mod,
)


@rca
def pcmci_shapley(
    data: pd.DataFrame,
    inject_time: int | None = None,
    dataset: str | None = None,
    dk_select_useful: bool = False,
    focus_node: str | None = None,
    trace_graph: nx.DiGraph | None = None,
    config: PCMCIShapleyConfig | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """PCMCI-Shapley end-to-end pipeline.

    Implements the CPG-Shap method by:
    (1) fusing deterministic trace-based topological constraints with
        probabilistic causal discovery (PC / PCMCI) to build a weighted
        causal propagation graph over services,
    (2) simulating discrete-time anomaly propagation on this graph, and
    (3) applying cooperative-game-theoretic Shapley attribution to obtain
        fine-grained, causally-informed root cause rankings.

    Returns dict with keys: adj, node_names, ranks.
    """
    logger = logging.getLogger("pcmci_shapley")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # 防止重複輸出
    logger.info("Starting pcmci_shapley pipeline")
    cfg = config or PCMCIShapleyConfig()

    # ---- Env overrides for quick experimentation (RQ2/RQ3) ----
    # 1) Causal method override (already partially supported below)
    # 2) Propagation decay factor alpha_prop
    try:
        alpha_env = os.environ.get("ALPHA_PROP")
        if alpha_env is not None:
            cfg.alpha_prop = float(alpha_env)
    except Exception:
        pass

    # 3) Shapley sampling rounds
    try:
        rounds_env = os.environ.get("SHAPLEY_ROUNDS")
        if rounds_env is not None:
            cfg.sampling_rounds = int(rounds_env)
    except Exception:
        pass

    # 4) Ablation toggles
    try:
        use_trace_env = os.environ.get("USE_TRACE")
        if use_trace_env is not None:
            cfg.use_trace = use_trace_env.lower() in {"1", "true", "yes"}
        attrib_env = os.environ.get("ATTRIBUTION_METHOD")
        if attrib_env is not None:
            cfg.attribution_method = attrib_env
    except Exception:
        pass

    cfg.validate()
    logger.debug({
        "tau_max": cfg.tau_max,
        "pcmci_alpha": cfg.pcmci_alpha,
        "theta": (cfg.theta1, cfg.theta2, cfg.theta3),
        "K": cfg.K,
        "alpha_prop": cfg.alpha_prop,
    })

    # Optional overrides via environment for quick tuning
    try:
        env_method = os.environ.get("CAUSAL_METHOD")
        if env_method in {"pcmci", "pc"}:
            cfg.causal_method = env_method
        if os.environ.get("FAST_MODE", "0") in {"1", "true", "True"}:
            cfg.causal_method = "pc"
            cfg.pcmci_max_conds_dim = 2
            cfg.tau_max = min(cfg.tau_max, 3)
            cfg.pruning_max_nodes = min(cfg.pruning_max_nodes, 15)
            cfg.enable_pruning = True
        logger.info(f"Effective causal_method={cfg.causal_method}, tau_max={cfg.tau_max}, max_conds_dim={cfg.pcmci_max_conds_dim}, pruning_max_nodes={cfg.pruning_max_nodes}")
    except Exception:
        pass

    t0 = time.time()
    # 1) Base preprocessing from framework
    base_df = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    logger.info(f"Base preprocess done: shape={base_df.shape}")
    t1 = time.time(); logger.info(f"TIMER base_preprocess: {(t1 - t0):.3f}s")

    # 2) Our method-specific preprocessing
    pp = prep_mod.preprocess_data(base_df, cfg)
    logger.info("Method-specific preprocessing complete")
    t2 = time.time(); logger.info(f"TIMER method_preprocess: {(t2 - t1):.3f}s")

    # 2.5) Pruning: Reduce candidate nodes before expensive operations
    if cfg.enable_pruning:
        from .pcmci_shapley_modules import pruning as prune_mod
        
        # 獲取所有候選節點 (從 metric_mapping 提取服務名)
        all_candidates = list(pp.get("metric_mapping", {}).keys())
        
        # 應用組合剪枝
        pruned_nodes = prune_mod.combined_pruning(
            all_nodes=all_candidates,
            node_anomaly=pp.get("node_anomaly", {}),
            trace_graph=trace_graph,
            focus_node=focus_node or all_candidates[0] if all_candidates else "unknown",
            max_hops=cfg.pruning_max_hops,
            anomaly_percentile=cfg.pruning_anomaly_percentile,
            min_nodes=cfg.pruning_min_nodes,
            max_nodes=cfg.pruning_max_nodes
        )
        
        logger.info(f"Pruning: {len(all_candidates)} -> {len(pruned_nodes)} nodes")
        
        # 更新 node_anomaly_ts 只保留剪枝後的節點
        node_anomaly_ts_original = pp.get("node_anomaly_ts", {})
        node_anomaly_ts = {k: v for k, v in node_anomaly_ts_original.items() 
                           if k in pruned_nodes}
    else:
        node_anomaly_ts = pp.get("node_anomaly_ts", {})
        logger.info("Pruning disabled")

    # 3) Determine focus node: prefer provided focus or SLI's service, else max anomaly
    if focus_node is None:
        sli = kwargs.get("sli")
        if isinstance(sli, str) and len(sli) > 0:
            focus_node = sli.split("_")[0]
    if focus_node is None and isinstance(pp.get("node_anomaly"), pd.Series) and not pp["node_anomaly"].empty:
        focus_node = pp["node_anomaly"].idxmax()
    logger.info(f"Focus node: {focus_node}")

    focus_node = focus_node or (list(pp.get("metric_mapping", {}).keys())[:1] or ["F"])[0]

    # 4) Ensure node_anomaly_ts is pd.Series types
    node_anomaly_ts = {k: (v if isinstance(v, pd.Series) else pd.Series(v)) 
                       for k, v in node_anomaly_ts.items()}

    # 5) Local node construction
    n_jobs = cfg.node_isolation_n_jobs if cfg.enable_parallel else 1
    U0 = iso_mod.statistical_neighborhood(focus_node, node_anomaly_ts, cfg.tau_max, cfg.top_m1, n_jobs=n_jobs)
    feats = iso_mod.build_isolation_features(U0, focus_node, node_anomaly_ts, cfg.tau_max)
    U1, I_scores = iso_mod.isolation_forest_selection(feats, cfg.top_m2)

    # trace-based augmentation can be disabled for ablation (w/o trace)
    if cfg.use_trace:
        U = iso_mod.trace_augmentation(U1, trace_graph, focus_node, cfg.u_max)
        logger.info(f"Trace augmentation enabled, |U1|={len(U1)} -> |U|={len(U)}")
    else:
        U = list(U1)
        logger.info(f"Trace augmentation DISABLED (use_trace=False), |U|={len(U)}")
    if focus_node not in U:
        U = [focus_node] + [n for n in U if n != focus_node]
    logger.info(f"Local set size: |U0|={len(U0)} |U1|={len(U1)} |U|={len(U)}")
    t3 = time.time(); logger.info(f"TIMER node_isolation: {(t3 - t2):.3f}s")

    # 6) Causal discovery using unified interface
    # 確保所有序列長度一致，缺失者以 0 補齊
    try:
        # 推斷時間長度：優先使用已有節點的長度，否則用 base_df 的行數
        existing_series = [v for v in node_anomaly_ts.values() if isinstance(v, pd.Series) and len(v) > 0]
        series_len = len(existing_series[0]) if existing_series else int(base_df.shape[0])
    except Exception:
        series_len = int(base_df.shape[0])

    series_map: Dict[str, pd.Series] = {}
    for s in U:
        v = node_anomaly_ts.get(s)
        if isinstance(v, pd.Series) and len(v) == series_len:
            series_map[s] = v
        elif isinstance(v, pd.Series) and len(v) > 0 and len(v) != series_len:
            # 對齊長度：截斷或以 0 補齊
            if len(v) > series_len:
                series_map[s] = v.iloc[-series_len:]
            else:
                pad = pd.Series([0.0] * (series_len - len(v)))
                series_map[s] = pd.concat([v, pad], ignore_index=True)
        else:
            # 缺失時以 0 序列填充
            series_map[s] = pd.Series([0.0] * series_len, dtype=float)

    service_df = pd.DataFrame({s: series_map[s].values for s in U})
    # 清理 NaN 值
    service_df = service_df.fillna(0)
    # 移除常數列（標準差為0）
    service_df = service_df.loc[:, service_df.std() > 0]
    
    logger.info(f"Causal discovery input shape: {service_df.shape}, method: {cfg.causal_method}")
    
    try:
        # Use unified causal discovery interface
        pcmci_res = causal_mod.discover_causal_graph(
            service_df, 
            list(service_df.columns), 
            cfg,
            method=cfg.causal_method
        )
        logger.info(f"Causal edges (var-level): {len(pcmci_res['edges'])}")
    except Exception as e:
        logger.warning(f"Causal discovery failed: {e}. Using empty causal graph.")
        pcmci_res = {
            'edges': [],
            'edge_strengths': {},
            'columns': list(service_df.columns)
        }
    t4 = time.time(); logger.info(f"TIMER causal_discovery: {(t4 - t3):.3f}s")

    # Map variable-level indices back to service names (combine by prefix)
    # Create a service graph and strengths aggregated by (service_i, service_j)
    cols = pcmci_res["columns"]
    var_to_service = {idx: col for idx, col in enumerate(cols)}

    agg_strengths: Dict[tuple[str, str], float] = {}
    for (i, j), s in pcmci_res["edge_strengths"].items():
        si, sj = var_to_service.get(i), var_to_service.get(j)
        if si is None or sj is None or si == sj:
            continue
        key = (si, sj)
        agg_strengths[key] = max(agg_strengths.get(key, 0.0), float(s))

    pcmci_edges_svc = list(agg_strengths.keys())

    # 7) Edge fusion
    # For ablation (w/o trace), we can drop all trace-based edges by providing an empty dict.
    trace_w = fuse_mod.extract_trace_weights(trace_graph, U) if cfg.use_trace else {}
    fused = fuse_mod.fuse_edge_weights(trace_w, agg_strengths, I_scores, cfg)
    fused = fuse_mod.apply_conflict_penalty(fused, pcmci_edges_svc, cfg.gamma)
    norm_w = fuse_mod.normalize_incoming_weights(fused, U)
    logger.info(f"Fused edges: raw={len(fused)} normalized={len(norm_w)} trace_edges={len(trace_w)}")
    t5 = time.time(); logger.info(f"TIMER edge_fusion: {(t5 - t4):.3f}s")

    # 8) Propagation
    # initial anomaly at final timestamp
    node_anomaly_last = {s: float(pp["node_anomaly"].get(s, 0.0)) for s in U}
    delta = prop_mod.compute_initial_anomaly(node_anomaly_last)
    prop = prop_mod.propagate_k_steps(delta, norm_w, cfg.K, cfg.alpha_prop)
    logger.info("Propagation completed")
    t6 = time.time(); logger.info(f"TIMER propagation: {(t6 - t5):.3f}s")

    # 9) Attribution: Shapley (default) vs graph-centrality baselines
    attribution_method = (cfg.attribution_method or "shapley").lower()
    if attribution_method == "shapley":
        shapley = shap_mod.compute_shapley_values(U, norm_w, delta, cfg)
        shapley_norm = shap_mod.normalize_shapley(shapley)
        logger.info("Shapley values computed")
    else:
        # Build a directed graph for PageRank / RandomWalk style attribution
        G_attr = nx.DiGraph()
        for (i, j), w in norm_w.items():
            G_attr.add_edge(i, j, weight=w)

        # PageRank-style score as attribution baseline
        try:
            pr_alpha = float(os.environ.get("PAGERANK_ALPHA", "0.85"))
        except Exception:
            pr_alpha = 0.85

        pr_scores = nx.pagerank(G_attr, alpha=pr_alpha, weight="weight") if G_attr.number_of_nodes() > 0 else {s: 0.0 for s in U}
        shapley = pr_scores
        shapley_norm = utils_mod.min_max_normalize(pr_scores)

        if attribution_method == "pagerank":
            logger.info("Using PageRank attribution instead of Shapley (cfg.attribution_method=pagerank)")
        else:
            logger.info(f"Using RandomWalk-style attribution baseline (cfg.attribution_method={cfg.attribution_method})")
    t7 = time.time(); logger.info(f"TIMER shapley: {(t7 - t6):.3f}s")

    # 10) Scoring & ranking
    # Reachability using normalized weights graph
    reach = score_mod.compute_reachability(focus_node, norm_w, U)
    reach_norm = utils_mod.min_max_normalize(reach)

    # Temporal penalty requires a per-node anomaly time (use first non-zero index)
    anomaly_time = {}
    for s in U:
        ser = node_anomaly_ts.get(s, pd.Series())
        idx = int(next((i for i, v in enumerate(ser.values) if v > 0), len(ser))) if len(ser) else 0
        anomaly_time[s] = idx
    # Build a directed graph from norm_w for temporal penalty
    G = nx.DiGraph()
    for (i, j), w in norm_w.items():
        G.add_edge(i, j)
    penalties = score_mod.compute_temporal_penalty(focus_node, anomaly_time, agg_strengths, G, cfg.lambda_penalty, U)

    # Normalize anomaly_last for scoring
    anomaly_norm = utils_mod.min_max_normalize(node_anomaly_last)
    scores = score_mod.compute_comprehensive_score(shapley_norm, reach_norm, anomaly_norm, cfg)
    ranks = score_mod.compute_final_ranking(scores, penalties)
    logger.info(f"Service-level ranking ready. Top-5: {ranks[:5] if ranks else ranks}")
    t8 = time.time(); logger.info(f"TIMER scoring: {(t8 - t7):.3f}s, TOTAL: {(t8 - t0):.3f}s")
    
    # 加入詳細的評分日誌
    logger.info(f"Scores sample: {dict(list(scores.items())[:5])}")
    logger.info(f"Shapley sample: {dict(list(shapley_norm.items())[:5])}")
    logger.info(f"Reachability sample: {dict(list(reach_norm.items())[:5])}")
    logger.info(f"Anomaly sample: {dict(list(anomaly_norm.items())[:5])}")
    logger.info(f"Penalties sample: {dict(list(penalties.items())[:5])}")

    # 11) Build adjacency matrix for return (using normalized fused weights)
    node_names = U
    idx = {n: i for i, n in enumerate(node_names)}
    adj = np.zeros((len(node_names), len(node_names)))
    for (i, j), w in norm_w.items():
        if i in idx and j in idx:
            adj[idx[i], idx[j]] = w

    # 12) Produce metric-level ranking strings like baro (service_metric)
    # Prefer latency metric if available, otherwise first metric under the service prefix
    base_cols = [c for c in base_df.columns if c != "time"]
    svc_to_cols = {}
    for c in base_cols:
        svc = c.split("_")[0] if "_" in c else c
        svc_to_cols.setdefault(svc, []).append(c)
    
    logger.info(f"Service-to-columns mapping sample: {dict(list(svc_to_cols.items())[:3])}")
    logger.info(f"Service ranks sample: {ranks[:5] if ranks else 'Empty'}")
    
    preferred_order = ["latency", "cpu", "mem", "disk", "loss", "io", "socket"]
    metric_ranks: list[str] = []
    for s in ranks:
        cols = svc_to_cols.get(s, [])
        pick = None
        # pick by preferred keyword
        for key in preferred_order:
            for c in cols:
                if key in c:
                    pick = c
                    break
            if pick:
                break
        if pick is None and cols:
            pick = cols[0]
        metric_ranks.append(pick or s)
    
    logger.info(f"Initial metric_ranks sample: {metric_ranks[:5] if metric_ranks else 'Empty'}")
    
    # Fallback if metric choices are empty OR graph is too sparse: sort by last anomaly scores
    total_weight = float(sum(norm_w.values())) if norm_w else 0.0
    logger.info(f"Total weight: {total_weight:.6f}, norm_w edges: {len(norm_w)}")
    
    fallback_triggered = False
    if (not metric_ranks or all((mr is None for mr in metric_ranks))) or total_weight == 0.0:
        fallback_triggered = True
        logger.warning(f"FALLBACK TRIGGERED! Reason: metric_ranks_empty={not metric_ranks}, all_none={all((mr is None for mr in metric_ranks)) if metric_ranks else False}, total_weight_zero={total_weight == 0.0}")
        try:
            anom = pp.get("anomaly_scores")
            if isinstance(anom, pd.DataFrame):
                scores_row = anom[[c for c in anom.columns if c != "time"]].iloc[-1]
                metric_ranks = scores_row.sort_values(ascending=False).index.tolist()
                logger.info(f"Fallback metric_ranks sample: {metric_ranks[:5]}")
            else:
                logger.warning("anomaly_scores is not DataFrame, fallback failed")
        except Exception as e:
            logger.error(f"Fallback exception: {e}")
    
    logger.info(f"Final metric-level Top-5: {metric_ranks[:5] if metric_ranks else metric_ranks}")
    logger.info(f"Fallback triggered: {fallback_triggered}")

    result = {
        "adj": adj,
        # Return the actual local set order used to build adj
        "node_names": node_names,
        "ranks": metric_ranks,
        "scores": scores,
        "shapley_values": shapley,
        "local_graph": G,
    }
    logger.info("pcmci_shapley pipeline completed")
    return result
