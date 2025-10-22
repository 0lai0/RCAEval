"""
SPOT (Streaming Peaks-Over-Threshold) 極值理論異常檢測模組

基於極值理論的流式單變量時間序列異常檢測模型
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy.stats import genpareto
import logging

logger = logging.getLogger(__name__)


class SPOTDetector:
    """
    SPOT 異常檢測器
    
    基於極值理論的流式異常檢測，使用廣義帕累托分佈 (GPD) 建模超閾值數據
    """
    
    def __init__(self, q: float = 0.001, init_window: int = 200, depth: int = 10):
        """
        初始化 SPOT 檢測器
        
        Args:
            q: 風險參數，控制偽陽性數量 (0.001 = 0.1% 的極值)
            init_window: 初始訓練窗口大小
            depth: 極值池深度
        """
        self.q = q
        self.init_window = init_window
        self.depth = depth
        
        # 狀態變量
        self.threshold = None
        self.gpd_params = None
        self.is_fitted = False
        self.extremes_pool = []
        
    def fit(self, data: np.ndarray) -> None:
        """
        擬合 SPOT 模型
        
        Args:
            data: 訓練數據，用於估計閾值和 GPD 參數
        """
        if len(data) < self.init_window:
            logger.warning(f"Data length {len(data)} < init_window {self.init_window}, using all data")
            self.init_window = len(data)
        
        # 使用初始窗口估計閾值
        init_data = data[:self.init_window]
        
        # 設定初始閾值為 95% 分位數（降低閾值以捕捉更多異常）
        self.threshold = np.percentile(init_data, 95)
        
        # 計算超閾值數據
        excesses = init_data[init_data > self.threshold] - self.threshold
        
        if len(excesses) < 3:
            # 進一步降低閾值
            self.threshold = np.percentile(init_data, 90)
            excesses = init_data[init_data > self.threshold] - self.threshold
            
            if len(excesses) < 3:
                # 使用更保守的閾值
                self.threshold = np.percentile(init_data, 85)
                excesses = init_data[init_data > self.threshold] - self.threshold
        
        if len(excesses) >= 3:
            try:
                # 擬合廣義帕累托分佈
                self.gpd_params = genpareto.fit(excesses, floc=0)
                self.is_fitted = True
                logger.debug(f"SPOT fitted: threshold={self.threshold:.4f}, excesses={len(excesses)}")
            except Exception as e:
                logger.debug(f"GPD fitting failed: {e}, using conservative approach")
                self.gpd_params = (0.1, 0, 0)  # 保守參數
                self.is_fitted = True
        else:
            logger.debug(f"Insufficient excesses ({len(excesses)}) for GPD fitting, using conservative approach")
            self.gpd_params = (0.1, 0, 0)
            self.is_fitted = True
    
    def step(self, value: float) -> float:
        """
        在線更新：處理新的數據點
        
        Args:
            value: 新的數據點
            
        Returns:
            異常分數 (0-1)
        """
        if not self.is_fitted:
            raise ValueError("SPOT model not fitted. Call fit() first.")
        
        # 計算異常分數
        if value <= self.threshold:
            return 0.0
        
        # 計算超閾值
        excess = value - self.threshold
        
        try:
            # 計算 p-value (罕見度)
            p_value = 1 - genpareto.cdf(excess, *self.gpd_params)
            # 轉換為異常分數 (0-1)
            anomaly_score = min(1.0, max(0.0, 1 - p_value))
            
            # 更新極值池
            self.extremes_pool.append(value)
            if len(self.extremes_pool) > self.depth:
                self.extremes_pool.pop(0)
            
            return anomaly_score
            
        except Exception as e:
            logger.warning(f"SPOT step failed: {e}")
            return 0.0
    
    def detect_batch(self, data: np.ndarray) -> np.ndarray:
        """
        批量檢測異常分數
        
        Args:
            data: 待檢測的數據
            
        Returns:
            異常分數數組 (0-1)
        """
        if not self.is_fitted:
            raise ValueError("SPOT model not fitted. Call fit() first.")
        
        scores = np.zeros(len(data))
        
        for i, value in enumerate(data):
            if value <= self.threshold:
                scores[i] = 0.0
            else:
                excess = value - self.threshold
                try:
                    p_value = 1 - genpareto.cdf(excess, *self.gpd_params)
                    scores[i] = min(1.0, max(0.0, 1 - p_value))
                except:
                    scores[i] = 0.0
        
        return scores


def spot_anomaly_detection(data: pd.DataFrame, config: Any) -> pd.DataFrame:
    """
    對每個時間序列應用 SPOT 算法
    
    Args:
        data: 輸入 DataFrame，包含時間序列數據
        config: 配置對象，包含 SPOT 參數
        
    Returns:
        DataFrame: 每個指標的 SPOT 異常分數（罕見度）
    """
    spot_scores = {}
    
    # 獲取 SPOT 參數
    q = getattr(config, 'spot_risk_param', 0.001)
    init_window = getattr(config, 'spot_init_window', 200)
    depth = getattr(config, 'spot_depth', 10)
    
    logger.info(f"SPOT detection: q={q}, init_window={init_window}, depth={depth}")
    
    for col in data.columns:
        if col == 'time':
            continue
        
        series = data[col].values
        
        if len(series) < init_window:
            logger.warning(f"Column {col}: insufficient data ({len(series)} < {init_window})")
            spot_scores[col] = [0.0] * len(series)
            continue
        
        try:
            # 初始化 SPOT 檢測器
            spot = SPOTDetector(q=q, init_window=init_window, depth=depth)
            
            # 擬合模型
            spot.fit(series)
            
            # 批量檢測
            scores = spot.detect_batch(series)
            spot_scores[col] = scores.tolist()
            
            logger.debug(f"Column {col}: max_score={np.max(scores):.4f}, mean_score={np.mean(scores):.4f}")
            
        except Exception as e:
            logger.warning(f"SPOT detection failed for column {col}: {e}")
            spot_scores[col] = [0.0] * len(series)
    
    # 構建結果 DataFrame
    result = pd.DataFrame(spot_scores)
    if 'time' in data.columns:
        result.insert(0, 'time', data['time'].values)
    
    logger.info(f"SPOT detection completed: {len(spot_scores)} columns processed")
    
    return result


def aggregate_spot_scores(spot_df: pd.DataFrame, metric_map: Dict[str, List[str]]) -> pd.Series:
    """
    將指標級 SPOT 分數聚合到服務級
    
    Args:
        spot_df: SPOT 異常分數 DataFrame
        metric_map: 服務到指標的映射
        
    Returns:
        每個服務的 SPOT 罕見度分數
    """
    scores = {}
    
    for service, metrics in metric_map.items():
        cols = [m for m in metrics if m in spot_df.columns]
        if not cols:
            scores[service] = 0.0
            continue
        
        # 使用最後時間點的最大 SPOT 分數
        try:
            max_score = spot_df[cols].iloc[-1].max()
            scores[service] = float(max_score)
        except Exception as e:
            logger.warning(f"Failed to aggregate SPOT scores for {service}: {e}")
            scores[service] = 0.0
    
    return pd.Series(scores)


# 測試函數
def test_spot_detector():
    """測試 SPOT 檢測器"""
    np.random.seed(42)
    
    # 創建測試數據
    normal_data = np.random.normal(0, 1, 1000)
    # 添加異常點
    normal_data[950:] = 10
    
    # 測試 SPOT
    spot = SPOTDetector(q=0.001, init_window=200)
    spot.fit(normal_data)
    
    scores = spot.detect_batch(normal_data)
    
    print(f"Max score: {np.max(scores):.4f}")
    print(f"Anomaly scores at end: {scores[-10:]}")
    print(f"Normal scores: {scores[900:950]}")
    
    return scores


if __name__ == "__main__":
    test_spot_detector()

