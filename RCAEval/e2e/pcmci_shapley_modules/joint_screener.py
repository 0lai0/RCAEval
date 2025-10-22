"""
Joint Screener: 聯合篩選器模組

實現 Fallback 異常分數 + SPOT 極值理論的聯合篩選器
這是增強版 PCMCI-Shapley 的核心創新
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

from .utils import min_max_normalize

logger = logging.getLogger(__name__)


def joint_screening(
    node_anomaly: pd.Series, 
    node_spot: pd.Series, 
    config: Any
) -> Tuple[List[str], Dict[str, float]]:
    """
    聯合篩選器：Fallback 異常分數 + SPOT 罕見度
    
    Args:
        node_anomaly: 節點異常分數 Series
        node_spot: 節點 SPOT 罕見度分數 Series  
        config: 配置對象
        
    Returns:
        (selected_nodes, fusion_scores): 選中的節點列表和融合分數字典
    """
    # 1. 歸一化兩個分數
    anomaly_norm = min_max_normalize(node_anomaly.to_dict())
    spot_norm = min_max_normalize(node_spot.to_dict())
    
    # 2. 增強版加權融合（提升精度）
    all_nodes = set(anomaly_norm.keys()) | set(spot_norm.keys())
    fusion_scores = {}
    for node in all_nodes:
        score_a = anomaly_norm.get(node, 0.0)
        score_s = spot_norm.get(node, 0.0)
        
        # 基礎融合分數
        base_fusion = (
            config.joint_weight_fallback * score_a + 
            config.joint_weight_spot * score_s
        )
        
        # 增強：如果兩個分數都高，給予額外獎勵
        if score_a > 0.5 and score_s > 0.5:
            base_fusion *= 1.5  # 50%獎勵
        
        # 增強：如果異常分數特別高，給予額外權重
        if score_a > 0.8:
            base_fusion += score_a * 0.3
        
        # 應用非線性變換增強差異
        fusion_scores[node] = np.tanh(base_fusion * 1.5) * 1.2
    
    # 3. 排序並選擇 Top-N
    sorted_nodes = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, _ in sorted_nodes[:config.joint_top_n]]
    
    logger.info(f"Joint screener: {len(all_nodes)} -> {len(selected_nodes)} nodes")
    logger.info(f"Top-5 fusion scores: {dict(sorted_nodes[:5])}")
    logger.info(f"Fusion score range: min={min(fusion_scores.values()):.6f}, max={max(fusion_scores.values()):.6f}")
    logger.info(f"Selected nodes: {selected_nodes}")
    
    return selected_nodes, fusion_scores


def evaluate_screening_quality(
    selected_nodes: List[str], 
    ground_truth: str, 
    all_nodes: List[str]
) -> Dict[str, float]:
    """
    評估篩選器質量
    
    Args:
        selected_nodes: 被選中的節點列表
        ground_truth: 真實根因節點
        all_nodes: 所有候選節點
        
    Returns:
        metrics: {
            'recall': 是否包含真實根因,
            'reduction_rate': 節點減少比例,
            'top_rank': 真實根因的排名
        }
    """
    metrics = {}
    
    # Recall: 真實根因是否被選中
    metrics['recall'] = 1.0 if ground_truth in selected_nodes else 0.0
    
    # Reduction rate: 減少了多少節點
    metrics['reduction_rate'] = 1 - len(selected_nodes) / max(len(all_nodes), 1)
    
    # Top rank: 真實根因在選中節點中的排名
    if ground_truth in selected_nodes:
        metrics['top_rank'] = selected_nodes.index(ground_truth) + 1
    else:
        metrics['top_rank'] = len(selected_nodes) + 1
    
    return metrics


def visualize_screening(
    anomaly_scores: Dict[str, float], 
    spot_scores: Dict[str, float], 
    fusion_scores: Dict[str, float], 
    selected_nodes: List[str],
    ground_truth: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    可視化篩選過程
    
    Args:
        anomaly_scores: 異常分數字典
        spot_scores: SPOT 分數字典
        fusion_scores: 融合分數字典
        selected_nodes: 被選中的節點
        ground_truth: 真實根因（可選）
        save_path: 保存路徑（可選）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        # 創建子圖
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 準備數據
        all_nodes = set(anomaly_scores.keys()) | set(spot_scores.keys())
        nodes = list(all_nodes)
        
        anomaly_vals = [anomaly_scores.get(n, 0.0) for n in nodes]
        spot_vals = [spot_scores.get(n, 0.0) for n in nodes]
        fusion_vals = [fusion_scores.get(n, 0.0) for n in nodes]
        
        # 顏色編碼
        colors = []
        for node in nodes:
            if node == ground_truth:
                colors.append('red')  # 真實根因
            elif node in selected_nodes:
                colors.append('blue')  # 被選中
            else:
                colors.append('gray')  # 未被選中
        
        # 異常分數散點圖
        axes[0].scatter(range(len(nodes)), anomaly_vals, c=colors, alpha=0.7)
        axes[0].set_title('Anomaly Scores')
        axes[0].set_xlabel('Node Index')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=45)
        
        # SPOT 分數散點圖
        axes[1].scatter(range(len(nodes)), spot_vals, c=colors, alpha=0.7)
        axes[1].set_title('SPOT Scores')
        axes[1].set_xlabel('Node Index')
        axes[1].set_ylabel('Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 融合分數散點圖
        axes[2].scatter(range(len(nodes)), fusion_vals, c=colors, alpha=0.7)
        axes[2].set_title('Fusion Scores')
        axes[2].set_xlabel('Node Index')
        axes[2].set_ylabel('Score')
        axes[2].tick_params(axis='x', rotation=45)
        
        # 添加圖例
        red_patch = mpatches.Patch(color='red', label='Ground Truth')
        blue_patch = mpatches.Patch(color='blue', label='Selected')
        gray_patch = mpatches.Patch(color='gray', label='Not Selected')
        fig.legend(handles=[red_patch, blue_patch, gray_patch], loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Screening visualization saved to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")


def adaptive_joint_screening(
    node_anomaly: pd.Series,
    node_spot: pd.Series,
    config: Any,
    data_quality: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], Dict[str, float]]:
    """
    自適應聯合篩選器：根據數據質量動態調整權重
    
    Args:
        node_anomaly: 節點異常分數
        node_spot: 節點 SPOT 分數
        config: 配置對象
        data_quality: 數據質量指標（可選）
        
    Returns:
        (selected_nodes, fusion_scores): 選中的節點和融合分數
    """
    # 默認權重
    w_fallback = config.joint_weight_fallback
    w_spot = config.joint_weight_spot
    
    # 根據數據質量調整權重
    if data_quality:
        # 如果數據中有大量常數列，降低異常分數權重
        removed_constant_rate = data_quality.get('removed_constant_rate', 0.0)
        if removed_constant_rate > 0.3:  # 超過30%的列被移除
            w_fallback *= 0.7
            w_spot *= 1.3
            # 重新歸一化
            total = w_fallback + w_spot
            w_fallback /= total
            w_spot /= total
            
        # 如果數據長度較短，降低 SPOT 權重（SPOT 需要足夠的歷史數據）
        data_length = data_quality.get('data_length', 1000)
        if data_length < config.spot_init_window:
            w_fallback *= 1.2
            w_spot *= 0.8
            # 重新歸一化
            total = w_fallback + w_spot
            w_fallback /= total
            w_spot /= total
    
    # 創建臨時配置
    temp_config = type(config)()
    temp_config.__dict__.update(config.__dict__)
    temp_config.joint_weight_fallback = w_fallback
    temp_config.joint_weight_spot = w_spot
    
    logger.info(f"Adaptive weights: fallback={w_fallback:.3f}, spot={w_spot:.3f}")
    
    # 使用調整後的權重進行篩選
    return joint_screening(node_anomaly, node_spot, temp_config)


def validate_screening_inputs(
    node_anomaly: pd.Series,
    node_spot: pd.Series
) -> Tuple[bool, str]:
    """
    驗證篩選器輸入
    
    Args:
        node_anomaly: 異常分數
        node_spot: SPOT 分數
        
    Returns:
        (is_valid, error_message)
    """
    if node_anomaly.empty and node_spot.empty:
        return False, "Both anomaly and spot scores are empty"
    
    if node_anomaly.empty:
        logger.warning("Anomaly scores are empty, using only SPOT scores")
    elif node_spot.empty:
        logger.warning("SPOT scores are empty, using only anomaly scores")
    
    # 檢查是否有有效分數
    valid_anomaly = node_anomaly.dropna()
    valid_spot = node_spot.dropna()
    
    if valid_anomaly.empty and valid_spot.empty:
        return False, "No valid scores found"
    
    return True, ""


# 測試函數
def test_joint_screener():
    """測試聯合篩選器"""
    # 創建測試數據
    np.random.seed(42)
    
    nodes = [f"service_{i}" for i in range(20)]
    
    # 模擬異常分數（大部分為0，少數較高）
    anomaly_scores = pd.Series(
        np.random.exponential(0.1, 20), 
        index=nodes
    )
    anomaly_scores.iloc[5] = 2.0  # 設置一個明顯的異常
    
    # 模擬 SPOT 分數（大部分為0，少數較高）
    spot_scores = pd.Series(
        np.random.exponential(0.05, 20),
        index=nodes
    )
    spot_scores.iloc[3] = 1.5  # 設置一個明顯的罕見值
    
    # 創建測試配置
    from .config import PCMCIShapleyConfig
    config = PCMCIShapleyConfig()
    config.joint_weight_fallback = 0.6
    config.joint_weight_spot = 0.4
    config.joint_top_n = 10
    
    # 測試聯合篩選
    selected, scores = joint_screening(anomaly_scores, spot_scores, config)
    
    print(f"Selected nodes: {selected[:5]}")
    print(f"Top scores: {dict(list(scores.items())[:5])}")
    
    # 測試質量評估
    metrics = evaluate_screening_quality(selected, "service_5", nodes)
    print(f"Quality metrics: {metrics}")
    
    return selected, scores, metrics


if __name__ == "__main__":
    test_joint_screener()

