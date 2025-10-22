import numpy as np
import pandas as pd
import networkx as nx
from typing import Any, Dict
import logging
import time

from RCAEval.e2e import rca
from RCAEval.io.time_series import preprocess

from .pcmci_shapley_modules import (
    PCMCIShapleyConfig,
    preprocessing as prep_mod,
    node_isolation as iso_mod,
    pcmci_local as pcmci_mod,
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
    start_time = time.time()
    timings = {}
    
    cfg = config or PCMCIShapleyConfig()
    cfg.validate()
    logger.debug({
        "tau_max": cfg.tau_max,
        "pcmci_alpha": cfg.pcmci_alpha,
        "theta": (cfg.theta1, cfg.theta2, cfg.theta3),
        "K": cfg.K,
        "alpha_prop": cfg.alpha_prop,
    })

    # 1) Base preprocessing from framework
    t1 = time.time()
    base_df = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    logger.info(f"Base preprocess done: shape={base_df.shape}")
    timings['base_preprocessing'] = time.time() - t1

    # 2) Our method-specific preprocessing
    t2 = time.time()
    try:
        pp = prep_mod.preprocess_data(base_df, cfg)
        logger.info("Method-specific preprocessing complete")
        logger.info(f"Preprocessing output keys: {list(pp.keys())}")
        logger.info(f"Node anomaly shape: {pp.get('node_anomaly', pd.Series()).shape}")
        logger.info(f"Node spot shape: {pp.get('node_spot', pd.Series()).shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
    timings['method_preprocessing'] = time.time() - t2

    # 2.5) Joint Screener: Advanced node selection using Fallback + SPOT
    t3 = time.time()
    if cfg.joint_screener_enabled:
        try:
            from .pcmci_shapley_modules import joint_screener as js_mod
            
            # 獲取所有候選節點
            all_candidates = list(pp.get("metric_mapping", {}).keys())
            logger.info(f"Joint screener input: {len(all_candidates)} candidate nodes")
            
            # 獲取異常分數和 SPOT 分數
            node_anomaly_series = pp.get("node_anomaly", pd.Series())
            node_spot_series = pp.get("node_spot", pd.Series())
            
            logger.info(f"Anomaly series length: {len(node_anomaly_series)}, Spot series length: {len(node_spot_series)}")
            
            # 驗證輸入
            is_valid, error_msg = js_mod.validate_screening_inputs(node_anomaly_series, node_spot_series)
            if not is_valid:
                logger.warning(f"Joint screener validation failed: {error_msg}, falling back to pruning")
                cfg.joint_screener_enabled = False
            else:
                # 執行聯合篩選
                screened_nodes, fusion_scores = js_mod.joint_screening(
                    node_anomaly_series, 
                    node_spot_series, 
                    cfg
                )
                
                logger.info(f"Joint screening: {len(all_candidates)} -> {len(screened_nodes)} nodes")
                logger.info(f"Top-5 fusion scores: {dict(list(fusion_scores.items())[:5])}")
                
                # 更新 node_anomaly_ts 只保留篩選後的節點
                node_anomaly_ts_original = pp.get("node_anomaly_ts", {})
                node_anomaly_ts = {k: v for k, v in node_anomaly_ts_original.items() 
                                   if k in screened_nodes}
                
                # 記錄融合分數供後續使用
                kwargs['fusion_scores'] = fusion_scores
        except Exception as e:
            logger.error(f"Joint screener failed: {e}, falling back to pruning")
            cfg.joint_screener_enabled = False
    
    # 2.6) Fallback to pruning if joint screener disabled or failed
    if not cfg.joint_screener_enabled:
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
                min_nodes=cfg.pruning_min_nodes
            )
            
            logger.info(f"Pruning: {len(all_candidates)} -> {len(pruned_nodes)} nodes")
            
            # 更新 node_anomaly_ts 只保留剪枝後的節點
            node_anomaly_ts_original = pp.get("node_anomaly_ts", {})
            node_anomaly_ts = {k: v for k, v in node_anomaly_ts_original.items() 
                               if k in pruned_nodes}
        else:
            node_anomaly_ts = pp.get("node_anomaly_ts", {})
            logger.info("Both joint screener and pruning disabled")
    timings['joint_screening'] = time.time() - t3

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
    t4 = time.time()
    n_jobs = cfg.node_isolation_n_jobs if cfg.enable_parallel else 1
    U0 = iso_mod.statistical_neighborhood(focus_node, node_anomaly_ts, cfg.tau_max, cfg.top_m1, n_jobs=n_jobs)
    feats = iso_mod.build_isolation_features(U0, focus_node, node_anomaly_ts, cfg.tau_max)
    U1, I_scores = iso_mod.isolation_forest_selection(feats, cfg.top_m2)
    U = iso_mod.trace_augmentation(U1, trace_graph, focus_node, cfg.u_max)
    if focus_node not in U:
        U = [focus_node] + [n for n in U if n != focus_node]
    logger.info(f"Local set size: |U0|={len(U0)} |U1|={len(U1)} |U|={len(U)}")
    timings['node_isolation'] = time.time() - t4

    # 6) PCMCI local causal test with adaptive strategy
    t5 = time.time()
    service_df = pd.DataFrame({s: node_anomaly_ts.get(s, pd.Series(dtype=float)).values for s in U})

    # 时间窗口截断（如果启用）
    if cfg.enable_window_truncation and cfg.window_length:
        if len(service_df) > cfg.window_length:
            service_df = service_df.tail(cfg.window_length)
            logger.info(f"Truncated time window to {cfg.window_length} timesteps")

    logger.info(f"PCMCI input shape: {service_df.shape}")
    logger.info(f"PCMCI input columns: {list(service_df.columns)}")
    logger.info(f"PCMCI input data range: min={service_df.min().min():.4f}, max={service_df.max().max():.4f}")

    # 使用自适应策略执行器
    try:
        pcmci_res = pcmci_mod.run_adaptive_pcmci(
            service_df,
            list(service_df.columns),
            cfg,
            focus_node=focus_node,
            trace_graph=trace_graph,
            anomaly_scores=pp.get("node_anomaly", None)
        )
        
        logger.info(f"PCMCI completed with strategy: {pcmci_res.get('strategy_used', 'unknown')}")
        logger.info(f"PCMCI edges (var-level): {len(pcmci_res['edges'])}")
        
        # 记录优化统计
        if 'n_chunks' in pcmci_res:
            logger.info(f"Chunking stats: {pcmci_res['n_chunks']} chunks, {pcmci_res['successful_chunks']} successful")
        if 'phase1_iterations' in pcmci_res:
            logger.info(f"Multi-phase stats: {pcmci_res['initial_nodes']} -> {pcmci_res['final_nodes']} nodes in {pcmci_res['phase1_iterations']} iterations")
        
        # 记录诊断信息
        if 'diagnostics' in pcmci_res:
            diagnostics = pcmci_res['diagnostics']
            logger.info(f"Data validation: {diagnostics}")
            
    except Exception as e:
        logger.error(f"PCMCI execution failed completely: {e}")
        # 创建空的 PCMCI 结果
        pcmci_res = {
            'edges': [],
            'edge_strengths': {},
            'columns': list(service_df.columns),
            'strategy_used': 'failed',
            'diagnostics': {'error': str(e)}
        }
    timings['pcmci'] = time.time() - t5

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
    trace_w = fuse_mod.extract_trace_weights(trace_graph, U)
    fused = fuse_mod.fuse_edge_weights(trace_w, agg_strengths, I_scores, cfg)
    fused = fuse_mod.apply_conflict_penalty(fused, pcmci_edges_svc, cfg.gamma)
    norm_w = fuse_mod.normalize_incoming_weights(fused, U)
    logger.info(f"Fused edges: raw={len(fused)} normalized={len(norm_w)} trace_edges={len(trace_w)}")

    # 8) Propagation
    # initial anomaly at final timestamp
    node_anomaly_last = {s: float(pp["node_anomaly"].get(s, 0.0)) for s in U}
    delta = prop_mod.compute_initial_anomaly(node_anomaly_last)
    prop = prop_mod.propagate_k_steps(delta, norm_w, cfg.K, cfg.alpha_prop)
    logger.info("Propagation completed")

    # 9) Shapley values
    t6 = time.time()
    shapley = shap_mod.compute_shapley_values(U, norm_w, delta, cfg)
    shapley_norm = shap_mod.normalize_shapley(shapley)
    logger.info("Shapley values computed")
    timings['shapley'] = time.time() - t6

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
    
    # Enhanced Fallback mechanism with intelligent triggering
    from .pcmci_shapley_modules import fallback_optimizer as fb_opt
    
    # Determine if Fallback should be triggered
    should_fallback, fallback_reason = fb_opt.should_trigger_fallback(
        pcmci_res, norm_w, metric_ranks, cfg
    )
    
    # Compute PCMCI confidence
    num_nodes = len(set([i for i, j in norm_w.keys()] + [j for i, j in norm_w.keys()])) if norm_w else 0
    pcmci_confidence = fb_opt.compute_pcmci_confidence(pcmci_res, norm_w, num_nodes)
    
    logger.info(f"PCMCI confidence: {pcmci_confidence:.3f}")
    logger.info(f"Should trigger fallback: {should_fallback}, reason: {fallback_reason}")
    
    if should_fallback:
        # Compute Fallback confidence
        node_anomaly_dict = {k: float(v) for k, v in pp.get("node_anomaly", {}).items()}
        node_spot_dict = {k: float(v) for k, v in pp.get("node_spot", {}).items()}
        fallback_confidence = fb_opt.compute_fallback_confidence(
            node_anomaly_dict, node_spot_dict
        )
        
        logger.info(f"Fallback confidence: {fallback_confidence:.3f}")
        
        # Generate Fallback ranking
        try:
            fallback_metric_ranks = fb_opt.intelligent_fallback_ranking(
                pp.get("anomaly_scores", pd.DataFrame()),
                pp.get("node_anomaly_ts", {}),
                cfg,
                focus_node
            )
            logger.info(f"Fallback metric_ranks sample: {fallback_metric_ranks[:5]}")
        except Exception as e:
            logger.error(f"Enhanced fallback failed: {e}, using simple fallback")
            # Simple fallback
            anom = pp.get("anomaly_scores")
            if isinstance(anom, pd.DataFrame):
                scores_row = anom[[c for c in anom.columns if c != "time"]].iloc[-1]
                fallback_metric_ranks = scores_row.sort_values(ascending=False).index.tolist()
            else:
                fallback_metric_ranks = []
        
        # Apply hybrid mode if enabled
        if cfg.enable_hybrid_mode and metric_ranks:
            logger.info("Applying hybrid ranking mode")
            metric_ranks = fb_opt.hybrid_ranking(
                metric_ranks, fallback_metric_ranks, pcmci_confidence, cfg
            )
        else:
            metric_ranks = fallback_metric_ranks
    
    logger.info(f"Final metric-level Top-5: {metric_ranks[:5] if metric_ranks else metric_ranks}")
    logger.info(f"Fallback triggered: {should_fallback}")

    # 计算总时间并记录性能统计
    total_time = time.time() - start_time
    timings['total'] = total_time
    
    logger.info(f"Pipeline timings: {timings}")
    logger.info(f"Total time: {total_time:.2f}s")

    result = {
        "adj": adj,
        "node_names": base_cols,
        "ranks": metric_ranks,
        "scores": scores,
        "shapley_values": shapley,
        "local_graph": G,
        "timings": timings,  # 添加性能统计
    }
    logger.info("pcmci_shapley pipeline completed")
    return result
