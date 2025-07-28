# MMrca 分離式根因分析系統

## 概述

MMrca (Multi-Modal Causal Root Cause Analysis) 是一個支援離線訓練和線上分析的多模態因果根因分析系統。該系統專為生產環境設計，可以有效分離計算密集型的訓練過程和實時的分析過程。

## 系統架構

### 離線訓練部分 (Offline Training)
- **因果圖學習**：使用歷史數據進行 Granger 因果性測試，構建因果關係圖
- **基準建立**：使用正常數據訓練 scaler 模型，建立正常行為基準
- **模型參數優化**：調整異常檢測閾值、因果測試參數等
- **模型持久化**：將訓練好的模型保存為 pickle 文件

### 線上分析部分 (Online Analysis)
- **實時異常檢測**：使用預訓練的 scaler 對新數據進行標準化和異常檢測
- **根因定位**：基於預構建的因果圖進行根因排序
- **傳播路徑分析**：實時追溯異常傳播路徑
- **快速響應**：避免重複的因果圖學習，提供毫秒級分析結果

## 核心組件

### 1. MMRCAOfflineTrainer
離線訓練器，負責模型訓練和因果圖學習。

```python
from RCAEval.e2e.mmrca_split import MMRCAOfflineTrainer

# 創建訓練器
trainer = MMRCAOfflineTrainer(config={
    "scaler_function": RobustScaler,
    "granger_maxlag": 5,
    "granger_p_threshold": 0.05,
    "anomaly_threshold": 3.0,
    "metric_resample_rate": 15
})

# 執行訓練
results = trainer.train(historical_data, dataset="my_dataset")

# 保存模型
trainer.save_model("mmrca_model.pkl")
```

### 2. MMRCAOnlineAnalyzer
線上分析器，負責實時異常檢測和根因分析。

```python
from RCAEval.e2e.mmrca_split import MMRCAOnlineAnalyzer

# 創建分析器
analyzer = MMRCAOnlineAnalyzer(model_path="mmrca_model.pkl")

# 執行分析
results = analyzer.analyze(real_time_data, dataset="my_dataset")
```

## 支援的數據模態

MMrca 系統支援以下數據模態：

- **metric**: 系統指標數據（CPU、內存、網絡等）
- **logs**: 日誌數據
- **logts**: 日誌時間序列數據
- **traces**: 分散式追蹤數據
- **tracets_err**: 追蹤錯誤時間序列
- **tracets_lat**: 追蹤延遲時間序列

## 使用方法

### 方法1：類別導向使用

```python
# 離線訓練
trainer = MMRCAOfflineTrainer(config=my_config)
trainer.train(historical_data, dataset="production")
trainer.save_model("production_model.pkl")

# 線上分析
analyzer = MMRCAOnlineAnalyzer(model_path="production_model.pkl")
results = analyzer.analyze(real_time_data, dataset="production")
```

### 方法2：便利函數使用

```python
from RCAEval.e2e.mmrca_split import mmrca_train_offline, mmrca_analyze_online

# 離線訓練
mmrca_train_offline(
    historical_data=historical_data,
    model_save_path="model.pkl",
    dataset="production",
    config=my_config
)

# 線上分析
results = mmrca_analyze_online(
    data=real_time_data,
    model_path="model.pkl",
    dataset="production"
)
```

### 方法3：兼容接口使用

```python
from RCAEval.e2e.mmrca_split import mmrca_split

# 使用預訓練模型
results = mmrca_split(
    data=data,
    inject_time=anomaly_time,
    dataset="production",
    model_path="model.pkl"  # 如果提供，使用預訓練模型
)

# 不使用預訓練模型（退化為一體化分析）
results = mmrca_split(
    data=data,
    inject_time=anomaly_time,
    dataset="production"
    # 不提供 model_path，使用原始 mmrca 方法
)
```

## 配置參數

### 訓練配置
```python
config = {
    "scaler_function": RobustScaler,          # 標準化函數
    "granger_maxlag": 5,                      # Granger 測試最大滯後
    "granger_p_threshold": 0.05,              # Granger 測試 p 值閾值
    "anomaly_threshold": 3.0,                 # 異常檢測閾值
    "metric_resample_rate": 15,               # 指標重採樣率
    "min_data_points": 20                     # 最小數據點數
}
```

### 數據格式
每個模態的數據應該是包含 'time' 列的 pandas DataFrame：

```python
data = {
    "metric": pd.DataFrame({
        'time': timestamps,
        'cpu_usage': cpu_values,
        'memory_usage': memory_values,
        # ... 其他指標
    }),
    "logts": pd.DataFrame({
        'time': timestamps,
        'error_count': error_counts,
        'warning_count': warning_counts,
        # ... 其他日誌指標
    }),
    # ... 其他模態
}
```

## 輸出格式

### 訓練結果
```python
{
    "timestamp": "2024-01-01T00:00:00",
    "config": {...},
    "num_features": 50,
    "num_scalers": 50,
    "graph_stats": {
        "num_nodes": 45,
        "num_edges": 120,
        "density": 0.0606,
        "is_dag": True,
        "potential_root_nodes": [...],
        "top_influential_nodes": [...]
    },
    "training_status": "completed"
}
```

### 分析結果
```python
{
    "anomaly_detection": {
        "anomaly_scores": {
            "metric_cpu_usage": {
                "score": 4.2,
                "anomalous_points": 15,
                "total_points": 100,
                "anomaly_ratio": 0.15
            },
            # ... 其他異常特徵
        },
        "threshold": 3.0,
        "detection_timestamp": "2024-01-01T00:00:00"
    },
    "root_cause_analysis": {
        "ranked_root_causes": ["metric_cpu_usage", "logts_error_count", ...],
        "anomaly_propagation_paths": {
            "metric_cpu_usage": [
                {
                    "target": "metric_memory_usage",
                    "path": ["metric_cpu_usage", "metric_memory_usage"],
                    "path_length": 1,
                    "anomaly_score": 3.8,
                    "anomaly_ratio": 0.12
                }
            ]
        },
        "analysis_summary": "識別出 5 個潛在根因，3 個傳播路徑"
    },
    "ranked_root_causes": [...],  # 兼容字段
    "ranks": [...],               # 兼容字段
    "detailed_insights": {...}
}
```

## 性能優勢

### 離線訓練
- **一次性計算**：因果圖學習只需執行一次
- **批量處理**：可以使用大量歷史數據進行訓練
- **參數優化**：有充足時間進行模型調優

### 線上分析
- **快速響應**：避免重複的因果圖學習
- **資源節省**：只需要進行標準化和圖搜索
- **可擴展性**：支援高頻率的實時分析

## 部署建議

### 生產環境部署
1. **離線訓練**：定期（如每週）使用最新歷史數據重新訓練模型
2. **模型版本管理**：維護多個模型版本，支援回滾
3. **監控指標**：監控模型性能和異常檢測準確率
4. **自動化流程**：建立自動化的訓練和部署流程

### 資源配置
- **訓練階段**：需要較多 CPU 和內存資源
- **分析階段**：資源需求較低，可以高頻率運行
- **存儲需求**：模型文件通常在 1-10MB 範圍內

## 故障排除

### 常見問題
1. **模型載入失敗**：檢查模型文件路徑和權限
2. **特徵不匹配**：確保訓練和分析數據具有相同的特徵結構
3. **因果圖為空**：檢查 Granger 測試參數和數據質量
4. **異常檢測敏感度**：調整 `anomaly_threshold` 參數

### 調試技巧
```python
# 啟用詳細日誌
import logging
logging.basicConfig(level=logging.DEBUG)

# 檢查模型狀態
trainer = MMRCAOfflineTrainer()
trainer.load_model("model.pkl")
print(f"特徵數: {len(trainer.scaler_models)}")
print(f"因果關係數: {trainer.causal_graph.number_of_edges()}")
```

## 示例代碼

完整的使用示例請參考 `mmrca_example.py` 文件，其中包含：
- 離線訓練完整流程
- 線上分析示例
- 便利函數使用
- 兼容接口示例

## 技術細節

### 因果發現算法
- 使用 Granger 因果性測試
- 支援時間序列平穩性檢測
- 自動差分處理非平穩序列

### 異常檢測方法
- 基於 Z-score 的統計異常檢測
- 支援 RobustScaler 和 StandardScaler
- 可配置的異常閾值

### 根因分析策略
- 基於因果圖的上游節點分析
- 因果影響力分數計算
- 異常傳播路徑追溯

## 擴展性

系統設計支援以下擴展：
- 新的因果發現算法
- 更多的異常檢測方法
- 額外的數據模態支援
- 自定義的根因排序策略

## 版本兼容性

- Python 3.7+
- pandas >= 1.0.0
- numpy >= 1.18.0
- scikit-learn >= 0.22.0
- networkx >= 2.0
- statsmodels >= 0.11.0 