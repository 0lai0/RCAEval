# GRUMVGC: 多模態統一表示學習與多變量格蘭傑因果分析

## 概述

GRUMVGC (Graph-based Root cause analysis using Unified Multimodal representation learning and Variational Granger Causality) 是一個創新的微服務根因分析方法，結合了：

1. **多模態統一表示學習**：通過對比學習將異構監控數據映射到統一嵌入空間
2. **多變量格蘭傑因果分析**：在統一空間中識別因果關係
3. **圖分析根因定位**：基於因果圖和異常分數進行根因排序

## 架構設計

```
原始多模態數據 → 統一表示學習 → 因果分析 → 根因排序
    ↓                ↓            ↓         ↓
Metrics/Logs/   →  GRU編碼器   →  MVGC   →  PageRank
Traces             對比學習       分析      + 異常分數
```

## 核心模組

### 1. MultiModalDataProcessor
負責多模態數據的預處理和對齊：
- 時間對齊與重採樣
- 缺失值處理
- 異常檢測

### 2. UnifiedRepresentationLearner
實現統一表示學習：
- 多模態GRU編碼器
- InfoNCE對比學習
- 跨模態語義對齊

### 3. MVGCAnalyzer
執行多變量格蘭傑因果分析：
- 高維嵌入降維
- 格蘭傑因果檢驗
- 多重檢驗校正

### 4. CausalGraphAnalyzer
因果圖分析和根因定位：
- 因果圖構建
- PageRank計算
- 綜合根因排序

## 使用方法

### 基本用法

```python
from RCAEval.e2e.grumvgc import grumvgc

# 準備多模態數據
multimodal_data = {
    'metric': metric_df,    # DataFrame with time column
    'logts': logts_df,      # DataFrame with time column  
    'tracets_err': trace_err_df,  # Optional
    'tracets_lat': trace_lat_df   # Optional
}

# 運行GRUMVGC
result = grumvgc(
    data=multimodal_data,
    inject_time=inject_time,
    dataset="your_dataset",
    embedding_dim=128,      # 嵌入維度
    seq_len=60,            # 序列長度
    num_epochs=50,         # 訓練輪數
    batch_size=32          # 批次大小
)

# 獲取結果
root_causes = result['ranks']           # 根因排序列表
adjacency_matrix = result['adj']        # 因果關係鄰接矩陣
node_names = result['node_names']       # 節點名稱列表
causal_edges = result['causal_edges']   # 因果邊列表
```

### 參數說明

| 參數 | 類型 | 默認值 | 說明 |
|-----|------|--------|------|
| `data` | dict/DataFrame | - | 多模態數據字典或單模態DataFrame |
| `inject_time` | int/float | - | 異常注入時間點 |
| `dataset` | str | None | 數據集名稱 |
| `embedding_dim` | int | 128 | 嵌入向量維度 |
| `seq_len` | int | 60 | 時間序列窗口長度 |
| `num_epochs` | int | 50 | 對比學習訓練輪數 |
| `batch_size` | int | 32 | 訓練批次大小 |
| `dk_select_useful` | bool | False | 是否選擇有用特徵 |

### 數據格式要求

#### Metrics數據
```python
metric_df = pd.DataFrame({
    'time': [1, 2, 3, ...],
    'service_a_cpu': [0.5, 0.6, 0.4, ...],
    'service_a_memory': [0.7, 0.8, 0.6, ...],
    'service_b_cpu': [0.3, 0.4, 0.5, ...],
    # ... 更多指標
})
```

#### Logs時間序列數據
```python
logts_df = pd.DataFrame({
    'time': [1, 2, 3, ...],
    'service_a_error_count': [2, 1, 5, ...],
    'service_a_request_count': [100, 120, 90, ...],
    'service_b_error_count': [0, 1, 2, ...],
    # ... 更多日誌計數
})
```

## 實驗結果格式

```python
{
    'ranks': ['service_a_cpu', 'service_a_memory', ...],  # 根因排序
    'adj': numpy.ndarray,                                 # 鄰接矩陣
    'node_names': ['metric_service_a_cpu', ...],         # 節點名稱
    'causal_edges': [('source', 'target', weight), ...], # 因果邊
    'embeddings_info': {                                  # 嵌入信息
        'num_features': 50,
        'embedding_dim': 128
    }
}
```

## 性能優化建議

### 1. 計算資源優化
```python
# 對於大規模數據，使用較小的參數
result = grumvgc(
    data=data,
    inject_time=inject_time,
    embedding_dim=64,    # 減小嵌入維度
    seq_len=30,          # 減小序列長度
    num_epochs=20,       # 減少訓練輪數
    batch_size=16        # 減小批次大小
)
```

### 2. 特徵選擇
```python
# 啟用特徵選擇以減少計算量
result = grumvgc(
    data=data,
    inject_time=inject_time,
    dk_select_useful=True  # 自動選擇有用特徵
)
```

### 3. GPU加速
確保安裝了CUDA版本的PyTorch以啟用GPU加速：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 測試與驗證

運行測試腳本驗證實現：
```bash
python3 simple_test_grumvgc.py
```

## 技術特點

### 優勢
1. **多模態融合**：有效整合metrics、logs、traces等異構數據
2. **語義對齊**：通過對比學習實現跨模態語義一致性
3. **因果推理**：基於格蘭傑因果性識別真實因果關係
4. **可解釋性**：提供因果圖和根因排序的可解釋結果

### 適用場景
- 微服務系統根因分析
- 多模態監控數據分析
- 複雜系統故障診斷
- 性能異常定位

### 限制
1. **計算複雜度**：對比學習訓練需要較多計算資源
2. **數據要求**：需要足夠的時間序列數據進行訓練
3. **參數敏感**：嵌入維度和序列長度需要根據數據調優

## 與其他方法的比較

| 方法 | 多模態支持 | 因果推理 | 可解釋性 | 計算複雜度 |
|------|-----------|----------|----------|-----------|
| GRUMVGC | ✅ | ✅ | ✅ | 高 |
| Granger PageRank | ❌ | ✅ | ✅ | 中 |
| CausalRCA | ✅ | ✅ | ✅ | 高 |
| MSCRED | ❌ | ❌ | ❌ | 中 |

## 故障排除

### 常見問題

1. **內存不足**
   ```python
   # 減小批次大小和嵌入維度
   result = grumvgc(data, inject_time, embedding_dim=32, batch_size=8)
   ```

2. **訓練不收斂**
   ```python
   # 增加訓練輪數或調整學習率
   result = grumvgc(data, inject_time, num_epochs=100)
   ```

3. **沒有找到因果關係**
   ```python
   # 調整格蘭傑因果檢驗的參數
   # 在MVGCAnalyzer中修改max_lag和p_threshold
   ```

## 貢獻與擴展

歡迎貢獻代碼和改進建議！可能的擴展方向：

1. **更多編碼器**：支持Transformer、LSTM等其他編碼器
2. **動態圖分析**：支持時變因果關係分析
3. **在線學習**：支持流式數據的在線根因分析
4. **可視化工具**：添加因果圖可視化功能

## 引用

如果您在研究中使用了GRUMVGC，請引用：

```bibtex
@inproceedings{grumvgc2024,
  title={GRUMVGC: Graph-based Root Cause Analysis using Unified Multimodal Representation Learning and Variational Granger Causality},
  author={Your Name},
  booktitle={Proceedings of ...},
  year={2024}
}
``` 