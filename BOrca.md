# BOrca: Bayesian Online Change Point Detection for Root Cause Analysis

## 概述

BOrca (Bayesian Online Root Cause Analysis) 是一個先進的多模態根因分析演算法，專門設計用於分析複雜系統中的異常事件。該演算法結合了嚴格的貝葉斯理論、自適應參數管理和魯棒因果推斷技術，能夠在多種數據類型（指標、日誌、追蹤等）中準確識別異常的根本原因。

## 主要特性

### 🔍 **多模態數據支持**
- 支援指標數據 (metrics)
- 日誌數據 (logs)
- 時序日誌 (logts)
- 追蹤數據 (traces)
- 追蹤錯誤 (traces_err)
- 追蹤延遲 (traces_lat)

### 🧠 **智能算法核心**
- **貝葉斯在線變點檢測 (BOCPD)**: 基於 Normal-Inverse-Wishart 共軛先驗的嚴格貝葉斯實現
- **自適應參數管理**: 根據數據特性動態調整算法參數
- **魯棒因果圖構建**: 多算法融合的因果發現機制
- **增強異常評分**: 多維度自適應異常評分系統

### 🎯 **智能特徵選擇**
- 基於異常類型的優先級篩選
- 數據質量評估和信息量分析
- 相關性和冗餘度分析
- 多階段自適應特徵選擇

## 系統架構

```
BOrca 系統架構
├── 數據預處理模組
│   ├── 多模態數據整合
│   ├── 數據清理和標準化
│   └── 特徵工程
├── 核心分析引擎
│   ├── MultivariateBOCPD (變點檢測)
│   ├── AdaptiveParameterManager (參數管理)
│   ├── RobustCausalGraphBuilder (因果分析)
│   └── EnhancedAnomalyScorer (異常評分)
├── 智能特徵選擇
│   ├── IntelligentFeatureSelector
│   └── AnomalyTypeDetector
└── 結果輸出
    ├── 根因排名
    ├── 服務排名
    └── 詳細分析報告
```

## 核心組件詳解

### 1. MultivariateBOCPD
**多元貝葉斯在線變點檢測**

```python
class MultivariateBOCPD:
    def __init__(self, alpha=None, beta=None, kappa=None, nu=None, mu=None, 
                 max_run_length=200, min_run_length=None):
```

**主要功能:**
- 基於 Normal-Inverse-Wishart 共軛先驗的嚴格貝葉斯實現
- 自適應危險函數調整
- 數值穩定性保障和多重回退機制
- 動態閾值變點檢測

**關鍵方法:**
- `detect_changepoints()`: 檢測數據中的變點
- `_log_marginal_likelihood()`: 計算貝葉斯邊際似然
- `_hazard_function()`: 變點危險函數

### 2. AdaptiveParameterManager
**自適應參數管理器**

```python
class AdaptiveParameterManager:
    def analyze_data_characteristics(self, data, feature_names=None):
    def get_bocpd_parameters(self, data_characteristics=None):
    def get_causal_parameters(self, data_characteristics=None):
```

**主要功能:**
- 分析數據特性（樣本數、特徵數、缺失率、數值穩定性等）
- 計算綜合數據質量評分
- 根據數據特性動態調整算法參數
- 參數歷史記錄和追蹤

### 3. RobustCausalGraphBuilder
**魯棒因果圖構建器**

```python
class RobustCausalGraphBuilder:
    def learn_causal_structure(self, data, feature_names):
    def compute_causal_scores(self, anomaly_data, statistical_scores):
```

**主要功能:**
- 多算法融合因果發現（PC、FCI算法）
- 因果可行性評估
- 多重共線性處理
- 因果強度計算和圖拓撲分析

### 4. EnhancedAnomalyScorer
**增強異常評分器**

```python
class EnhancedAnomalyScorer:
    def learn_normal_distribution(self, data, feature_names):
    def calculate_anomaly_scores(self, anomaly_data, feature_names):
```

**主要功能:**
- 多維度異常評分（Z-score、IQR、範圍、變化）
- 魯棒統計方法
- 異常類型特定調整
- 自適應權重組合

### 5. IntelligentFeatureSelector
**智能特徵選擇器**

```python
class IntelligentFeatureSelector:
    def select_features(self, data, feature_names=None, max_features=None):
```

**主要功能:**
- 多階段特徵選擇策略
- 基於異常類型的優先級篩選
- 數據質量和信息量評估
- 冗餘特徵移除

## 使用方法

### 基本用法

```python
from RCAEval.e2e.BOrca import borca_multimodal

# 單一數據源
result = borca_multimodal(
    data=your_dataframe,
    inject_time=anomaly_timestamp,
    verbose=True
)

# 多模態數據
multimodal_data = {
    'metric': metrics_df,
    'logs': logs_df,
    'traces': traces_df
}

result = borca_multimodal(
    data=multimodal_data,
    inject_time=anomaly_timestamp,
    verbose=True
)
```

### 高級配置

```python
result = borca_multimodal(
    data=your_data,
    inject_time=anomaly_timestamp,
    use_causal_graph=True,           # 啟用因果分析
    max_features=50,                 # 最大特徵數
    bocpd_timeout=300,               # BOCPD超時設定
    dk_select_useful=True,           # 啟用有用特徵選擇
    verbose=True
)
```

### 輸出結果

```python
{
    "anomaly_detected": True,                    # 是否檢測到異常
    "estimated_anomaly_time": 1234,             # 估計異常時間
    "changepoints": [1200, 1234],               # 檢測到的變點
    "changepoint_probs": [0.3, 0.8],           # 變點概率
    "node_names": ["cpu_usage", "memory_usage"], # 特徵名稱
    "ranks": ["cpu_usage", "memory_usage"],      # 根因排名
    "scores": {"cpu_usage": 0.85, "memory_usage": 0.72}, # 異常分數
    "service_rankings": [("service_a", 0.85)],  # 服務排名
    "detected_anomaly_type": "CPU",             # 檢測到的異常類型
    "anomaly_type_confidence": 0.9,             # 異常類型置信度
    "causal_graph_used": True,                  # 是否使用因果圖
    "data_characteristics": {...},              # 數據特性分析
    "bocpd_parameters": {...},                  # BOCPD參數
    "selected_features": [...],                 # 選擇的特徵
    "causal_graph_reliability": 0.75,           # 因果圖可靠性
    "causal_graph_metrics": {...}               # 因果圖指標
}
```

## 異常類型支持

BOrca 支援多種異常類型的自動檢測和分析：

- **CPU**: CPU使用率、處理器負載相關異常
- **MEM**: 記憶體使用、堆記憶體、垃圾回收相關異常
- **DISK**: 磁碟I/O、存儲讀寫相關異常
- **DELAY**: 延遲、響應時間相關異常
- **LOSS**: 錯誤率、失敗率、丟包相關異常
- **UNKNOWN**: 未知類型異常

## 依賴項

### 必需依賴
```bash
numpy>=1.19.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
networkx>=2.6.0
```

### 可選依賴（因果分析）
```bash
causallearn>=0.1.3.0
```

## 安裝指南

1. 安裝基礎依賴：
```bash
pip install numpy pandas scipy scikit-learn networkx
```

2. 安裝因果學習庫（可選）：
```bash
pip install causal-learn
```

3. 將 BOrca.py 放置在適當的目錄中

## 性能優化

### 數據預處理建議
- 移除常數特徵和高度相關的特徵
- 使用魯棒標準化處理異常值
- 適當的特徵選擇可以大幅提升性能

### 參數調優
- `max_features`: 根據數據集大小調整（建議10-50）
- `bocpd_timeout`: 根據計算資源調整超時時間
- `alpha`: BOCPD敏感性參數（0.05-0.8）

### 計算複雜度
- 時間複雜度: O(T × R × D²)，其中T是時間步數，R是最大運行長度，D是特徵維度
- 空間複雜度: O(T × R)

## 最佳實踐

### 1. 數據準備
```python
# 確保數據質量
data = data.dropna()  # 移除缺失值
data = data.select_dtypes(include=[np.number])  # 選擇數值型特徵
```

### 2. 參數設置
```python
# 根據數據集大小調整參數
if len(data) < 100:
    max_features = 10
elif len(data) < 1000:
    max_features = 30
else:
    max_features = 50
```

### 3. 結果解釋
```python
# 檢查結果可靠性
if result['causal_graph_used'] and result['causal_graph_reliability'] > 0.6:
    print("因果分析結果可靠")
    
# 分析異常類型
if result['anomaly_type_confidence'] > 0.7:
    print(f"異常類型: {result['detected_anomaly_type']}")
```

## 故障排除

### 常見問題

1. **因果學習庫不可用**
   ```
   Warning: causallearn not available. Causal graph integration disabled.
   ```
   解決方案：安裝 `causal-learn` 庫

2. **數據質量評分過低**
   ```
   數據質量評分: 0.234
   ```
   解決方案：檢查數據完整性，移除異常值，增加數據預處理

3. **BOCPD超時**
   ```
   BOCPD timeout at t=150
   ```
   解決方案：增加 `bocpd_timeout` 參數或減少 `max_features`

4. **沒有檢測到變點**
   ```
   Warning: 沒有檢測到明顯的變點
   ```
   解決方案：調整 BOCPD 敏感性參數或檢查數據品質

### 調試模式

```python
# 啟用詳細輸出
result = borca_multimodal(
    data=your_data,
    verbose=True,  # 啟用詳細日誌
    inject_time=anomaly_time
)
```

## 擴展開發

### 添加新的異常類型
```python
# 在 AnomalyTypeDetector 中添加新的關鍵字
type_keywords = {
    'NETWORK': ['network', 'bandwidth', 'packet', 'connection'],
    # 其他類型...
}
```

### 自定義評分函數
```python
class CustomAnomalyScorer(EnhancedAnomalyScorer):
    def _calculate_custom_scores(self, feature_data, stats):
        # 實現自定義評分邏輯
        pass
```

## 引用和致謝

如果您在研究中使用了 BOrca，請考慮引用相關的學術論文和技術文檔。

## 許可證

請參考項目的許可證文件以了解使用條款和限制。

## 聯繫和支持

如有問題或建議，請通過以下方式聯繫：
- 提交 GitHub Issues
- 發送電子郵件至維護團隊
- 參與社區討論

---

**注意**: 本演算法適用於複雜系統的根因分析，建議在使用前充分了解您的數據特性和業務需求。
