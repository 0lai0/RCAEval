# MULAN: Multi-Modal Causal Structure Learning for RCAEval

## 概述

MULAN (Multi-Modal Causal Structure Learning for Effective Root Cause Analysis) 是一個專為微服務系統設計的多模態根因分析方法，現已整合到 RCAEval 基準框架中。

## 🎯 核心特性

### 四大核心模組
1. **日誌表示提取** - 使用語言模型將日誌轉為時間序列表示
2. **對比學習** - 提取模態共用與特定表示，生成因果圖
3. **KPI注意力融合** - 根據與KPI的相關性融合多模態因果圖
4. **隨機遊走定位** - 基於融合圖進行根因排序

### 支援的數據集
- **RE1**: 單模態（僅指標數據）- 375個案例
- **RE2**: 多模態（指標+日誌+追蹤）- 270個案例  
- **RE3**: 多模態（指標+日誌+追蹤）- 90個案例，支援程式碼層級故障

## 📊 數據處理

### RE1 數據結構
```python
# 單模態數據（DataFrame）
data = pd.DataFrame({
    'time': timestamps,
    'frontend_latency': [...],  # KPI指標
    'service_1_cpu': [...],     # 其他指標
    'service_1_memory': [...],
    # ... 49-212個指標欄位
})
```

### RE2/RE3 數據結構
```python
# 多模態數據（字典）
data = {
    "metric": metric_df,        # 指標時間序列 (77-376個指標)
    "logts": logts_df,          # 日誌時間序列 (預處理後)
    "logs": logs_df,            # 原始日誌 (可選)
    "tracets_lat": trace_lat_df, # 追蹤延遲時間序列
    "tracets_err": trace_err_df, # 追蹤錯誤時間序列
    "traces": traces_df         # 原始追蹤數據 (可選)
}
```

## 🚀 使用方法

### 1. 基本使用

#### RE1 單模態分析
```bash
python main.py --method mulan --dataset re1-ob --length 20
```

#### RE2/RE3 多模態分析
```bash
python main.py --method mulan --dataset re2-tt --length 20
python main.py --method mulan --dataset re3-tk --length 20
```

### 2. 程式化使用

#### RE1 範例
```python
from RCAEval.e2e.mulan import mulan
import pandas as pd

# 準備RE1數據
data = pd.read_csv("path/to/re1_data.csv")
inject_time = "2023-01-01 12:00:00"

# 執行MULAN
results = mulan(
    data=data,
    inject_time=inject_time,
    dataset="re1",
    sli="frontend_latency",  # 指定KPI
    num_epochs=100,
    learning_rate=0.001
)

print("根因排序:", results['ranks'][:5])
```

#### RE2/RE3 範例
```python
# 準備多模態數據
mmdata = {
    "metric": metric_df,
    "logts": logts_df,
    "tracets_lat": tracets_lat_df,
    "tracets_err": tracets_err_df
}

# 執行MULAN（不使用追蹤）
results = mulan(
    data=mmdata,
    inject_time=inject_time,
    dataset="re2",
    sli="frontend_latency",
    use_traces=False,  # 僅使用指標+日誌
    num_epochs=100
)

# 執行MULAN（完整多模態）
results = mulan(
    data=mmdata,
    inject_time=inject_time,
    dataset="re2",
    sli="frontend_latency",
    use_traces=True,   # 使用指標+日誌+追蹤
    num_epochs=100
)
```

## ⚙️ 參數配置

### 主要參數
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `data` | DataFrame/dict | - | 輸入數據 |
| `inject_time` | str/datetime | None | 故障注入時間 |
| `dataset` | str | None | 數據集名稱 (re1/re2/re3) |
| `sli` | str | None | KPI指標名稱 |
| `use_traces` | bool | False | 是否使用追蹤數據 |

### 訓練參數
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `num_epochs` | int | 100 | 訓練輪數 |
| `learning_rate` | float | 0.001 | 學習率 |
| `lambda_1` | float | 1.0 | VAR損失權重 |
| `lambda_2` | float | 1.0 | 正交損失權重 |
| `lambda_3` | float | 1.0 | 節點對比損失權重 |
| `lambda_4` | float | 1.0 | 邊預測損失權重 |
| `lambda_5` | float | 0.1 | 稀疏正則化權重 |
| `beta` | float | 0.85 | 隨機遊走阻尼因子 |

### 進階參數
```python
results = mulan(
    data=data,
    inject_time=inject_time,
    dataset="re2",
    sli="frontend_latency",
    
    # 訓練參數
    num_epochs=200,
    learning_rate=0.001,
    
    # 損失函數權重
    lambda_1=1.0,    # VAR預測損失
    lambda_2=1.0,    # 正交約束損失
    lambda_3=1.0,    # 節點對比損失
    lambda_4=1.0,    # 邊預測損失
    lambda_5=0.1,    # L1正則化
    
    # 模型結構
    hidden_dim=64,   # 隱藏層維度
    repr_dim=32,     # 表示維度
    
    # 其他參數
    beta=0.85,       # 隨機遊走阻尼
    use_traces=True, # 使用追蹤數據
    dk_select_useful=False
)
```

## 📈 輸出結果

### 返回值結構
```python
{
    "ranks": ["service_1", "service_2", ...],  # 排序的根因列表
    "adj_matrix": np.ndarray,                  # 融合的鄰接矩陣
    "ranking_scores": np.array,                # 各實體的排序分數
    "node_names": ["service_1", ...]          # 實體名稱列表
}
```

### 評估指標
- **Avg@5**: 前5個預測中命中真實根因的平均比例
- **MRR**: 平均倒數排名
- **MAP@K**: 平均精確度@K

## 🔧 測試與驗證

### 運行測試
```bash
python test_mulan.py
```

### 測試內容
1. **RE1 單模態測試** - 驗證僅使用指標數據的功能
2. **RE2 多模態測試** - 驗證指標+日誌的組合
3. **RE3 追蹤測試** - 驗證完整三模態功能
4. **性能評估測試** - 計算Avg@5等指標

## 🎯 最佳實踐

### 1. KPI選擇
```python
# 自動檢測前端延遲作為KPI
sli = "frontend_latency"

# 或指定其他KPI
sli = "response_time_90th"
sli = "error_rate"
```

### 2. 資料前處理
```python
# 使用RCAEval的預處理工具
from RCAEval.io.time_series import preprocess, drop_constant

# 清理數據
data = preprocess(data, dataset="re2", dk_select_useful=True)
data = drop_constant(data)  # 移除常數列
```

### 3. 超參數調優
```python
# 對於大型數據集，增加epochs
num_epochs = 200

# 對於多模態數據，調整損失權重
lambda_2 = 0.5  # 降低正交約束
lambda_3 = 2.0  # 增強模態對比學習
```

### 4. 計算資源優化
```python
# 檢查GPU可用性
import torch
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print("Using CPU - consider reducing model complexity")
```

## ⚠️ 注意事項

### 數據品質要求
1. **時間對齊**: 確保多模態數據的時間軸對齊
2. **完整性**: 避免過多的缺失值
3. **一致性**: 保持實體命名的一致性

### 性能考量
1. **日誌量**: RE2數據集包含8.6-26.9百萬行日誌，需要足夠的計算資源
2. **內存使用**: 大型數據集可能需要16GB+內存
3. **訓練時間**: 完整訓練可能需要數小時

### 故障排除
1. **CUDA錯誤**: 確保PyTorch版本與CUDA兼容
2. **內存不足**: 減少batch大小或模型維度
3. **數據格式**: 確保數據符合預期的DataFrame/dict格式

## 📚 參考資料

- [RCAEval GitHub](https://github.com/phamquiluan/RCAEval)
- [MULAN論文](論文連結)
- [多模態RCA示例](docs/multi-source-rca-demo.ipynb)
