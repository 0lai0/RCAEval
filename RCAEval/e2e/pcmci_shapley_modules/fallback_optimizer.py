"""
Fallback Mechanism Optimizer: Fallback 機制優化模組

實現智能 Fallback 觸發、混合模式排序、增強異常分數計算和置信度評估
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def should_trigger_fallback(
    pcmci_res: Dict[str, Any], 
    norm_w: Dict[Tuple[str, str], float], 
    metric_ranks: List[str], 
    cfg: Any
) -> Tuple[bool, str]:
    """
    智能 Fallback 觸發邏輯
    
    Args:
        pcmci_res: PCMCI 結果
        norm_w: 歸一化權重
        metric_ranks: 指標排名
        cfg: 配置對象
        
    Returns:
        (should_fallback: bool, reason: str)
    """
    reasons = []
    
    # 1. PCMCI 完全失敗（空圖）
    if len(pcmci_res.get('edges', [])) == 0:
        reasons.append("pcmci_empty")
    
    # 2. 融合後權重總和為 0
    total_weight = sum(norm_w.values()) if norm_w else 0.0
    if total_weight == 0.0:
        reasons.append("total_weight_zero")
    
    # 3. 圖過於稀疏（邊數 < 節點數 / 2）
    num_nodes = len(set([i for i, j in norm_w.keys()] + [j for i, j in norm_w.keys()])) if norm_w else 0
    if num_nodes > 0 and len(norm_w) < num_nodes * cfg.fallback_sparse_threshold:
        reasons.append("graph_too_sparse")
    
    # 4. metric_ranks 為空或全 None
    if not metric_ranks or all(mr is None for mr in metric_ranks):
        reasons.append("metric_ranks_invalid")
    
    # 5. 使用了非標準 PCMCI 策略
    strategy_used = pcmci_res.get('strategy_used', 'unknown')
    if strategy_used in ['very_relaxed', 'lagged_correlation', 'empty']:
        reasons.append(f"pcmci_degraded_{strategy_used}")
    
    # 決策：如果有任何一個嚴重原因，觸發 Fallback
    severe_reasons = ['pcmci_empty', 'total_weight_zero', 'metric_ranks_invalid']
    should_fallback = any(r in reasons for r in severe_reasons)
    
    # 如果圖稀疏但不為空，可以考慮部分 Fallback（混合模式）
    if 'graph_too_sparse' in reasons and not should_fallback:
        should_fallback = cfg.fallback_on_sparse_graph
        reasons.append("sparse_graph_policy")
    
    reason_str = ", ".join(reasons) if reasons else "none"
    logger.info(f"Fallback trigger analysis: PCMCI edges={len(pcmci_res.get('edges', []))}, total_weight={total_weight:.6f}, metric_ranks_valid={bool(metric_ranks)}")
    logger.info(f"Should trigger fallback: {should_fallback}, reason: {reason_str}")
    
    return should_fallback, reason_str


def hybrid_ranking(
    ranks_from_pcmci: List[str],
    ranks_from_fallback: List[str],
    pcmci_confidence: float,
    cfg: Any
) -> List[str]:
    """
    混合排序：根據 PCMCI 信心度混合兩種排序
    
    Args:
        ranks_from_pcmci: PCMCI 產生的排名
        ranks_from_fallback: Fallback 產生的排名
        pcmci_confidence: PCMCI 信心度 (0-1)
        cfg: 配置對象
        
    Returns:
        混合後的排名
    """
    if pcmci_confidence >= cfg.hybrid_confidence_threshold:
        # 高信心，使用 PCMCI 排序
        return ranks_from_pcmci
    elif pcmci_confidence <= 1 - cfg.hybrid_confidence_threshold:
        # 低信心，使用 Fallback 排序
        return ranks_from_fallback
    else:
        # 中等信心，混合排序
        # 策略：交錯選擇，PCMCI 優先
        hybrid = []
        i, j = 0, 0
        seen = set()
        
        while i < len(ranks_from_pcmci) or j < len(ranks_from_fallback):
            # 優先添加 PCMCI 的結果
            if i < len(ranks_from_pcmci) and ranks_from_pcmci[i] not in seen:
                hybrid.append(ranks_from_pcmci[i])
                seen.add(ranks_from_pcmci[i])
            i += 1
            
            # 然後添加 Fallback 的結果
            if j < len(ranks_from_fallback) and ranks_from_fallback[j] not in seen:
                hybrid.append(ranks_from_fallback[j])
                seen.add(ranks_from_fallback[j])
            j += 1
        
        return hybrid


def enhanced_anomaly_score(anomaly_ts: pd.Series, timestamp: int, cfg: Any) -> float:
    """
    增強的異常分數計算
    
    考慮：
    1. 最後時間點分數（即時性）
    2. 近期平均分數（持續性）
    3. 近期最大分數（嚴重性）
    4. 異常突增程度（變化率）
    
    Args:
        anomaly_ts: 異常時間序列
        timestamp: 當前時間戳
        cfg: 配置對象
        
    Returns:
        增強的異常分數
    """
    if len(anomaly_ts) == 0:
        return 0.0
    
    # 1. 即時性：最後時間點
    instant_score = anomaly_ts.iloc[-1] if len(anomaly_ts) > 0 else 0.0
    
    # 2. 持續性：最近 10 個時間點的平均
    window = min(10, len(anomaly_ts))
    persistent_score = anomaly_ts.iloc[-window:].mean()
    
    # 3. 嚴重性：最近 10 個時間點的最大值
    severity_score = anomaly_ts.iloc[-window:].max()
    
    # 4. 變化率：最後時間點相對於之前的變化
    if len(anomaly_ts) >= 2:
        prev_avg = anomaly_ts.iloc[-window:-1].mean() if window > 1 else anomaly_ts.iloc[-2]
        change_rate = (instant_score - prev_avg) / (prev_avg + 1e-10)
        change_score = min(1.0, max(0.0, change_rate))
    else:
        change_score = 0.0
    
    # 增強版加權融合（提升精度）
    # 更重視即時性和嚴重性
    final_score = (
        0.4 * instant_score +      # 即時性40%（提升）
        0.3 * severity_score +    # 嚴重性30%（提升）
        0.2 * persistent_score +  # 持續性20%（降低）
        0.1 * change_score        # 變化率10%（降低）
    )
    
    # 應用非線性變換增強差異
    final_score = np.tanh(final_score * 2.0) * 1.5
    
    return final_score


def compute_fallback_confidence(
    anomaly_scores: Dict[str, float],
    spot_scores: Dict[str, float],
    data_quality: Optional[Dict[str, Any]] = None
) -> float:
    """
    計算 Fallback 結果的置信度
    
    Args:
        anomaly_scores: 異常分數字典
        spot_scores: SPOT 分數字典
        data_quality: 數據質量指標
        
    Returns:
        confidence: 0-1，越高表示越可靠
    """
    confidence_factors = []
    
    # 1. 異常分數分佈的區分度
    scores = list(anomaly_scores.values())
    if len(scores) > 1:
        # 使用變異係數衡量區分度
        cv = np.std(scores) / (np.mean(scores) + 1e-10)
        distinction = min(1.0, cv / 2.0)  # 變異係數越大，區分度越高
        confidence_factors.append(distinction)
    
    # 2. SPOT 分數的一致性
    if spot_scores:
        # 如果 Fallback 和 SPOT 的 Top-5 有重疊，置信度更高
        top5_anomaly = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_spot = sorted(spot_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        overlap = len(set([n for n, _ in top5_anomaly]) & set([n for n, _ in top5_spot]))
        consistency = overlap / 5.0
        confidence_factors.append(consistency)
    
    # 3. 數據質量
    if data_quality:
        # 數據完整性、常數列比例等
        removed_constant_rate = data_quality.get('removed_constant_rate', 0.0)
        data_conf = 1.0 - removed_constant_rate
        confidence_factors.append(data_conf)
    
    # 綜合置信度
    if confidence_factors:
        return np.mean(confidence_factors)
    else:
        return 0.5  # 預設中等置信度


def compute_pcmci_confidence(
    pcmci_res: Dict[str, Any],
    norm_w: Dict[Tuple[str, str], float],
    num_nodes: int
) -> float:
    """
    計算 PCMCI 結果的置信度
    
    Args:
        pcmci_res: PCMCI 結果
        norm_w: 歸一化權重
        num_nodes: 節點數量
        
    Returns:
        confidence: 0-1，越高表示越可靠
    """
    confidence_factors = []
    
    # 1. PCMCI 策略質量
    strategy_used = pcmci_res.get('strategy_used', 'unknown')
    if strategy_used == 'standard':
        strategy_conf = 1.0
    elif strategy_used == 'relaxed':
        strategy_conf = 0.8
    elif strategy_used == 'very_relaxed':
        strategy_conf = 0.6
    elif strategy_used == 'lagged_correlation':
        strategy_conf = 0.4
    else:
        strategy_conf = 0.2
    confidence_factors.append(strategy_conf)
    
    # 2. 圖密度
    if num_nodes > 0:
        edge_density = len(norm_w) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        density_conf = min(1.0, edge_density * 10)  # 密度越高，置信度越高
        confidence_factors.append(density_conf)
    
    # 3. 權重分佈
    if norm_w:
        weights = list(norm_w.values())
        weight_cv = np.std(weights) / (np.mean(weights) + 1e-10)
        weight_conf = min(1.0, weight_cv)  # 權重變異性適中時置信度較高
        confidence_factors.append(weight_conf)
    
    # 綜合置信度
    if confidence_factors:
        return np.mean(confidence_factors)
    else:
        return 0.3  # 預設較低置信度


def intelligent_fallback_ranking(
    anomaly_df: pd.DataFrame,
    node_anomaly_ts: Dict[str, pd.Series],
    cfg: Any,
    focus_node: Optional[str] = None
) -> List[str]:
    """
    智能 Fallback 排名：使用增強的異常分數計算
    
    Args:
        anomaly_df: 異常分數 DataFrame
        node_anomaly_ts: 節點異常時間序列
        cfg: 配置對象
        focus_node: 焦點節點
        
    Returns:
        排名列表
    """
    if anomaly_df.empty:
        logger.warning("Anomaly DataFrame is empty, returning empty ranking")
        return []
    
    # 計算增強的異常分數
    enhanced_scores = {}
    
    for service, ts in node_anomaly_ts.items():
        if len(ts) > 0:
            enhanced_scores[service] = enhanced_anomaly_score(ts, len(ts)-1, cfg)
        else:
            enhanced_scores[service] = 0.0
    
    # 如果沒有時間序列數據，使用最後一行異常分數
    if not enhanced_scores:
        try:
            scores_row = anomaly_df[[c for c in anomaly_df.columns if c != "time"]].iloc[-1]
            # 聚合到服務級別
            service_scores = {}
            for col, score in scores_row.items():
                if "_" in col:
                    service = col.split("_")[0]
                else:
                    service = col
                service_scores[service] = max(service_scores.get(service, 0.0), score)
            enhanced_scores = service_scores
        except Exception as e:
            logger.error(f"Failed to compute fallback scores: {e}")
            return []
    
    # 排序
    sorted_services = sorted(enhanced_scores.items(), key=lambda x: x[1], reverse=True)
    ranking = [service for service, _ in sorted_services]
    
    # 確保焦點節點在排名中（如果提供）
    if focus_node and focus_node not in ranking:
        ranking.insert(0, focus_node)
    
    logger.info(f"Enhanced fallback ranking: {ranking[:5]}")
    return ranking


# 測試函數
def test_fallback_optimizer():
    """測試 Fallback 優化器"""
    from .config import PCMCIShapleyConfig
    
    config = PCMCIShapleyConfig()
    
    # 測試 Fallback 觸發邏輯
    pcmci_res = {'edges': [], 'strategy_used': 'empty'}
    norm_w = {}
    metric_ranks = []
    
    should_fallback, reason = should_trigger_fallback(pcmci_res, norm_w, metric_ranks, config)
    print(f"Should trigger fallback: {should_fallback}, reason: {reason}")
    
    # 測試混合排序
    pcmci_ranks = ['A', 'B', 'C']
    fallback_ranks = ['B', 'D', 'A']
    confidence = 0.8
    
    hybrid = hybrid_ranking(pcmci_ranks, fallback_ranks, confidence, config)
    print(f"Hybrid ranking: {hybrid}")
    
    # 測試增強異常分數
    ts = pd.Series([0.1, 0.2, 0.5, 0.8, 1.0])
    score = enhanced_anomaly_score(ts, 4, config)
    print(f"Enhanced anomaly score: {score}")
    
    # 測試置信度計算
    anomaly_scores = {'A': 0.8, 'B': 0.3, 'C': 0.1}
    spot_scores = {'A': 0.9, 'B': 0.2, 'C': 0.05}
    
    confidence = compute_fallback_confidence(anomaly_scores, spot_scores)
    print(f"Fallback confidence: {confidence}")
    
    return True


if __name__ == "__main__":
    test_fallback_optimizer()

