# PCMCI-Shapley 根因分析方法實作計畫

**專案名稱**: Local PCMCI-lag Causal + Shapley Propagation Ranking  
**實作位置**: `RCAEval/e2e/pcmci_shapley.py` (主入口) + `RCAEval/e2e/pcmci_shapley_modules/` (模組資料夾)  
**建立日期**: 2025-10-15  
**實作狀態**: 規劃階段

---

## 架構設計總覽

```
RCAEval/e2e/
├── pcmci_shapley.py              # 主要入口檔案，提供 pcmci_shapley() 函數
└── pcmci_shapley_modules/        # 模組資料夾
    ├── __init__.py               # 模組初始化
    ├── preprocessing.py          # 步驟0: 資料預處理與異常偵測
    ├── node_isolation.py         # 步驟1: 局部節點建構
    ├── pcmci_local.py           # 步驟2: PCMCI+ 局部滯後因果檢定
    ├── edge_fusion.py           # 步驟3: 多源邊權融合
    ├── propagation.py           # 步驟4: 異常傳播模型
    ├── shapley.py               # 步驟5: Shapley Value 貢獻計算
    ├── scoring.py               # 步驟6-7: 可達性、時間一致性與綜合排名
    └── utils.py                 # 工具函數
```

---

## 50步驟實作計畫

### 階段 A: 專案結構建立 (Steps 1-8)

#### Step 1: 建立模組資料夾結構
- **任務**: 建立 `RCAEval/e2e/pcmci_shapley_modules/` 資料夾
- **檔案**: 資料夾
- **依賴**: 無
- **驗證**: 資料夾存在

#### Step 2: 建立模組 __init__.py
- **任務**: 建立模組初始化檔案，定義對外介面
- **檔案**: `pcmci_shapley_modules/__init__.py`
- **依賴**: Step 1
- **內容**: 匯入所有子模組的主要函數
- **驗證**: 可被 import

#### Step 3: 建立主入口檔案框架
- **任務**: 建立 `pcmci_shapley.py`，定義主函數簽名
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Step 2
- **內容**: 
  - 匯入必要套件
  - 定義 `@rca` 裝飾器
  - 定義 `pcmci_shapley(data, inject_time, dataset, **kwargs)` 函數框架
- **驗證**: 函數可被呼叫，返回 dummy 結果

#### Step 4: 建立 utils.py 工具模組
- **任務**: 建立通用工具函數模組
- **檔案**: `pcmci_shapley_modules/utils.py`
- **依賴**: Step 1
- **內容**:
  - `min_max_normalize()`: Min-Max 正規化
  - `robust_standardize()`: Robust 標準化
  - `ensure_2d()`: 確保數組為 2D
  - `safe_divide()`: 安全除法（避免除零）
- **驗證**: 單元測試通過

#### Step 5: 研究現有 PCMCI 實作
- **任務**: 分析 `RCAEval/graph_construction/pcmci.py`
- **檔案**: 無（研究）
- **依賴**: 無
- **內容**: 理解現有 PCMCI 實作方式，確認可重用部分
- **驗證**: 文件記錄重用策略

#### Step 6: 研究 trace graph 獲取方式
- **任務**: 分析專案中 trace graph 的資料結構
- **檔案**: 無（研究）
- **依賴**: 無
- **內容**: 查看 `tracerca.py` 等檔案，理解 trace 資料格式
- **驗證**: 文件記錄 trace graph 介面

#### Step 7: 確認依賴套件
- **任務**: 列出所需的外部套件
- **檔案**: 文件
- **依賴**: 無
- **內容**:
  - `tigramite`: PCMCI 演算法
  - `numpy`, `pandas`: 數據處理
  - `networkx`: 圖操作
  - `scikit-learn`: IsolationForest, PCA
  - `scipy`: 統計函數
- **驗證**: 確認 requirements.txt 已包含

#### Step 8: 建立配置參數類別
- **任務**: 定義方法的超參數配置
- **檔案**: `pcmci_shapley_modules/config.py`
- **依賴**: Step 1
- **內容**:
  - `PCMCIShapleyConfig` 類別
  - 所有超參數的預設值（τ_max, α, θ1/θ2/θ3, K, R, λ 等）
- **驗證**: 可實例化並訪問參數

---

### 階段 B: 資料預處理模組 (Steps 9-14)

#### Step 9: 建立 preprocessing.py 模組框架
- **任務**: 建立資料預處理模組
- **檔案**: `pcmci_shapley_modules/preprocessing.py`
- **依賴**: Step 4, Step 8
- **內容**: 模組結構與匯入

#### Step 10: 實作指標標準化函數
- **任務**: 實作 Robust 標準化（基於 MAD）
- **函數**: `robust_normalize(df: pd.DataFrame) -> pd.DataFrame`
- **依賴**: Step 9
- **公式**: `x̃ = (x - median(x)) / MAD(x)`
- **驗證**: 單元測試，確認標準化正確

#### Step 11: 實作異常偵測函數 (Robust Z-score)
- **任務**: 基於 Robust Z-score 的異常分數計算
- **函數**: `detect_anomaly_zscore(df: pd.DataFrame, threshold: float) -> pd.DataFrame`
- **依賴**: Step 10
- **公式**: `a_{s,m}(t) = |x̃| if |x̃| > threshold else 0`
- **驗證**: 測試異常值被正確標記

#### Step 12: 實作異常偵測函數 (SPOT 可選)
- **任務**: 實作基於 SPOT 演算法的異常偵測（進階選項）
- **函數**: `detect_anomaly_spot(df: pd.DataFrame) -> pd.DataFrame`
- **依賴**: Step 11
- **內容**: 可選實作，若時間不足可先使用 Robust Z-score
- **驗證**: 測試 SPOT 偵測效果

#### Step 13: 實作節點異常聚合函數
- **任務**: 將多個 metrics 的異常分數聚合為節點分數
- **函數**: `aggregate_node_anomaly(anomaly_df: pd.DataFrame, metric_map: dict) -> pd.Series`
- **依賴**: Step 11
- **公式**: `a_s(t) = (1/M_s) Σ_m a_{s,m}(t)`
- **驗證**: 測試聚合分數計算正確

#### Step 14: 整合預處理流程
- **任務**: 建立完整的預處理 pipeline
- **函數**: `preprocess_data(data: pd.DataFrame, config: Config) -> dict`
- **依賴**: Steps 10-13
- **輸出**: 
  ```python
  {
      'normalized_df': pd.DataFrame,
      'anomaly_scores': pd.DataFrame,  # a_{s,m}(t)
      'node_anomaly': pd.Series,       # a_s(t)
  }
  ```
- **驗證**: 整合測試

---

### 階段 C: 局部節點建構模組 (Steps 15-21)

#### Step 15: 建立 node_isolation.py 模組框架
- **任務**: 建立節點隔離與篩選模組
- **檔案**: `pcmci_shapley_modules/node_isolation.py`
- **依賴**: Step 4, Step 8

#### Step 16: 實作滯後互相關計算
- **任務**: 計算焦點節點與其他節點的滯後相關性
- **函數**: `compute_lagged_correlation(focus_series: pd.Series, other_series: pd.Series, tau_max: int) -> np.ndarray`
- **依賴**: Step 15
- **公式**: `ρ_{F,s}(τ) = corr(a_F(t), a_s(t-τ))`
- **驗證**: 測試相關性計算正確

#### Step 17: 實作統計鄰域擴展
- **任務**: 基於相關性篩選初步候選節點 U0
- **函數**: `statistical_neighborhood(focus_node: str, node_anomaly: dict, tau_max: int, top_m1: int) -> list`
- **依賴**: Step 16
- **輸出**: Top-M1 個相關節點
- **驗證**: 測試返回正確數量的節點

#### Step 18: 實作 Isolation-based 特徵建構
- **任務**: 為每個候選節點建構滯後相關特徵向量
- **函數**: `build_isolation_features(candidates: list, focus_node: str, node_anomaly: dict, tau_max: int) -> pd.DataFrame`
- **依賴**: Step 16
- **公式**: `z_s = [ρ_{F,s}(τ_1), ..., ρ_{F,s}(τ_max)]`
- **驗證**: 特徵矩陣維度正確

#### Step 19: 實作 Isolation Forest 篩選
- **任務**: 使用 IsolationForest 篩選節點
- **函數**: `isolation_forest_selection(features: pd.DataFrame, top_m2: int) -> tuple`
- **依賴**: Step 18
- **輸出**: (選中的節點列表 U1, 每個節點的 isolation 分數 I_s)
- **驗證**: 測試篩選機制

#### Step 20: 實作 Trace 補強
- **任務**: 基於 trace graph 補充鄰居節點
- **函數**: `trace_augmentation(selected_nodes: list, trace_graph: nx.Graph, focus_node: str, max_size: int) -> list`
- **依賴**: Step 6
- **公式**: `U = U1 ∪ trace_1hop(U1) ∪ {F}`
- **驗證**: 測試節點集合不超過 max_size

#### Step 21: 整合節點建構流程
- **任務**: 整合完整的局部節點建構 pipeline
- **函數**: `build_local_nodes(data: dict, focus_node: str, trace_graph: nx.Graph, config: Config) -> dict`
- **依賴**: Steps 15-20
- **輸出**:
  ```python
  {
      'local_nodes': list,      # U
      'isolation_scores': dict, # I_s
  }
  ```
- **驗證**: 整合測試

---

### 階段 D: PCMCI+ 局部因果檢定模組 (Steps 22-27)

#### Step 22: 建立 pcmci_local.py 模組框架
- **任務**: 建立 PCMCI 局部檢定模組
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **依賴**: Step 5, Step 8

#### Step 23: 實作時間序列矩陣建構
- **任務**: 從局部節點建構 PCMCI 輸入矩陣
- **函數**: `build_timeseries_matrix(data: pd.DataFrame, local_nodes: list, use_pca: bool) -> np.ndarray`
- **依賴**: Step 22
- **內容**: 可選 PCA 降維或直接使用 a_s(t)
- **驗證**: 矩陣維度正確

#### Step 24: 實作 PCMCI+ 檢定包裝器
- **任務**: 包裝 tigramite 的 PCMCI 演算法
- **函數**: `run_pcmci_plus(data_matrix: np.ndarray, tau_max: int, alpha: float) -> dict`
- **依賴**: Step 5, Step 23
- **內容**: 
  - 使用 ParCorr 檢定
  - 返回 p-values 和相關係數
- **驗證**: PCMCI 成功執行

#### Step 25: 實作顯著邊提取
- **任務**: 從 PCMCI 結果提取顯著因果邊
- **函數**: `extract_significant_edges(pcmci_result: dict, alpha: float) -> list`
- **依賴**: Step 24
- **公式**: `E_PCMCI = {(j→i, τ): p_{j→i}(τ) ≤ α, τ≥1}`
- **驗證**: 測試邊提取正確

#### Step 26: 實作時滯關聯強度計算
- **任務**: 計算邊的關聯強度
- **函數**: `compute_edge_strength(pcmci_result: dict, edges: list) -> dict`
- **依賴**: Step 25
- **公式**: `s_{j→i} = max_τ |r_{j→i|Z}(τ)|`
- **驗證**: 強度值在 [0, 1] 範圍

#### Step 27: 整合 PCMCI 檢定流程
- **任務**: 整合完整的 PCMCI 檢定 pipeline
- **函數**: `local_pcmci_causal_test(data: pd.DataFrame, local_nodes: list, config: Config) -> dict`
- **依賴**: Steps 22-26
- **輸出**:
  ```python
  {
      'edges': list,           # E_PCMCI
      'edge_strengths': dict,  # s_{i→j}
      'pcmci_graph': nx.DiGraph
  }
  ```
- **驗證**: 整合測試

---

### 階段 E: 多源邊權融合模組 (Steps 28-32)

#### Step 28: 建立 edge_fusion.py 模組框架
- **任務**: 建立邊權融合模組
- **檔案**: `pcmci_shapley_modules/edge_fusion.py`
- **依賴**: Step 8

#### Step 29: 實作 Trace 邊權提取
- **任務**: 從 trace graph 提取邊權重
- **函數**: `extract_trace_weights(trace_graph: nx.Graph, local_nodes: list) -> dict`
- **依賴**: Step 6, Step 28
- **輸出**: `w^{trace}_{i→j}` 字典
- **驗證**: 邊權重合理

#### Step 30: 實作三源權重融合
- **任務**: 融合 trace、PCMCI、isolation 三種來源的權重
- **函數**: `fuse_edge_weights(trace_weights: dict, pcmci_strengths: dict, isolation_scores: dict, config: Config) -> dict`
- **依賴**: Steps 21, 27, 29
- **公式**: `w_{i→j} = θ1·w^{trace}_{i→j} + θ2·s_{i→j} + θ3·I_i`
- **驗證**: 測試融合公式正確

#### Step 31: 實作方向衝突懲罰
- **任務**: 處理 trace 和 PCMCI 方向不一致的情況
- **函數**: `apply_conflict_penalty(weights: dict, pcmci_edges: list, trace_edges: list, gamma: float) -> dict`
- **依賴**: Step 30
- **公式**: `w_{i→j} = w_{i→j}(1 - γ·1[PCMCI(j→i)成立])`
- **驗證**: 測試懲罰應用正確

#### Step 32: 實作入邊歸一化
- **任務**: 對每個節點的入邊進行歸一化
- **函數**: `normalize_incoming_weights(weights: dict, local_nodes: list) -> dict`
- **依賴**: Step 31
- **公式**: `ŵ_{i→j} = w_{i→j} / Σ_{k∈N_in(j)} w_{k→j}`
- **驗證**: 每個節點入邊權重和為 1

---

### 階段 F: 異常傳播模型模組 (Steps 33-37)

#### Step 33: 建立 propagation.py 模組框架
- **任務**: 建立異常傳播模組
- **檔案**: `pcmci_shapley_modules/propagation.py`
- **依賴**: Step 8

#### Step 34: 實作初始異常強度計算
- **任務**: 計算每個節點的初始異常強度
- **函數**: `compute_initial_anomaly(node_anomaly: pd.Series, timestamp: int) -> dict`
- **依賴**: Step 33
- **公式**: `δ_s = log(1 + a_s(T))`
- **驗證**: 測試初始值計算

#### Step 35: 實作單步傳播更新
- **任務**: 實作一步異常傳播
- **函數**: `propagate_one_step(current_h: dict, edge_weights: dict, alpha_prop: float) -> dict`
- **依賴**: Step 34
- **公式**: `h_j^{(k+1)} = α_prop Σ_{i∈N_in(j)} ŵ_{i→j} h_i^{(k)}`
- **驗證**: 測試傳播計算正確

#### Step 36: 實作 K 步傳播
- **任務**: 執行 K 步傳播並累積影響
- **函數**: `propagate_k_steps(initial_delta: dict, edge_weights: dict, K: int, alpha_prop: float) -> dict`
- **依賴**: Step 35
- **輸出**: 
  ```python
  {
      'h_final': dict,        # h_s = Σ_k h_s^{(k)}
      'h_by_step': list       # 每一步的 h 值
  }
  ```
- **驗證**: K 步傳播正確

#### Step 37: 實作可視化傳播過程（可選）
- **任務**: 繪製傳播過程的動態圖
- **函數**: `visualize_propagation(h_by_step: list, graph: nx.DiGraph) -> None`
- **依賴**: Step 36
- **內容**: 可選功能，用於 debug
- **驗證**: 可視化輸出正確

---

### 階段 G: Shapley Value 計算模組 (Steps 38-42)

#### Step 38: 建立 shapley.py 模組框架
- **任務**: 建立 Shapley 值計算模組
- **檔案**: `pcmci_shapley_modules/shapley.py`
- **依賴**: Step 8

#### Step 39: 實作系統異常指標函數
- **任務**: 定義局部系統異常指標 v(C)
- **函數**: `compute_system_anomaly(coalition: set, edge_weights: dict, local_nodes: list, alpha_prop: float, K: int) -> float`
- **依賴**: Step 36
- **公式**: `v(C) = Σ_{j∈U} v_j · h_j^{(C)}(K)`
- **驗證**: 測試系統異常值合理

#### Step 40: 實作 Shapley 值抽樣計算
- **任務**: 使用蒙特卡洛抽樣近似 Shapley 值
- **函數**: `compute_shapley_sampling(local_nodes: list, edge_weights: dict, config: Config) -> dict`
- **依賴**: Step 39
- **公式**: `φ̂_s = (1/R) Σ_r [v(C^{(r)}_{-s} ∪ {s}) - v(C^{(r)}_{-s})]`
- **驗證**: Shapley 值總和接近 v(U)

#### Step 41: 實作精確 Shapley 值計算（可選）
- **任務**: 針對小規模節點集合的精確計算
- **函數**: `compute_shapley_exact(local_nodes: list, edge_weights: dict, config: Config) -> dict`
- **依賴**: Step 39
- **內容**: 當 |U| ≤ 15 時使用精確計算
- **驗證**: 與抽樣結果比較

#### Step 42: 實作 Shapley 值正規化
- **任務**: 正規化 Shapley 值到 [0, 1]
- **函數**: `normalize_shapley(shapley_values: dict) -> dict`
- **依賴**: Step 40
- **驗證**: 正規化後值在合理範圍

---

### 階段 H: 評分與排名模組 (Steps 43-47)

#### Step 43: 建立 scoring.py 模組框架
- **任務**: 建立評分與排名模組
- **檔案**: `pcmci_shapley_modules/scoring.py`
- **依賴**: Step 8

#### Step 44: 實作可達性計算
- **任務**: 計算從每個節點到焦點節點的可達性
- **函數**: `compute_reachability(node: str, focus_node: str, edge_weights: dict, graph: nx.DiGraph) -> float`
- **依賴**: Step 43
- **公式**: `r_s = max_{path(s→F)} Π_{(i→j)∈path} ŵ_{i→j}`
- **驗證**: 測試路徑權重計算

#### Step 45: 實作時間一致性懲罰
- **任務**: 檢測並懲罰違反時間順序的路徑
- **函數**: `compute_temporal_penalty(node: str, focus_node: str, anomaly_timestamps: dict, graph: nx.DiGraph, lambda_penalty: float) -> float`
- **依賴**: Step 44
- **公式**: `p_s = exp(-λ Σ_{violated paths} s_{i→j})`
- **驗證**: 測試懲罰計算

#### Step 46: 實作綜合分數計算
- **任務**: 整合 Shapley、可達性、異常分數
- **函數**: `compute_comprehensive_score(shapley_values: dict, reachability: dict, node_anomaly: dict, config: Config) -> dict`
- **依賴**: Steps 42, 44
- **公式**: `Score_s = α1·φ̂_s + α2·r̂_s + α3·â_s`
- **驗證**: 測試分數計算

#### Step 47: 實作最終排名
- **任務**: 應用時間懲罰並排序
- **函數**: `compute_final_ranking(scores: dict, penalties: dict) -> list`
- **依賴**: Steps 45, 46
- **公式**: `Rank(s) = argsort(Score_s · p_s)`
- **驗證**: 測試排名順序

---

### 階段 I: 主流程整合 (Steps 48-50)

#### Step 48: 實作主函數邏輯
- **任務**: 在 `pcmci_shapley.py` 中實作完整流程
- **函數**: 完善 `pcmci_shapley(data, inject_time, dataset, **kwargs)` 函數
- **依賴**: Steps 14, 21, 27, 32, 36, 42, 47
- **流程**:
  1. 資料預處理
  2. 局部節點建構
  3. PCMCI+ 檢定
  4. 邊權融合
  5. 異常傳播
  6. Shapley 計算
  7. 評分排名
- **輸出**:
  ```python
  {
      'adj': np.ndarray,           # 融合後的鄰接矩陣
      'node_names': list,          # 所有節點名稱
      'ranks': list,               # 根因排名列表
      'scores': dict,              # 詳細分數
      'shapley_values': dict,      # Shapley 值
      'local_graph': nx.DiGraph,   # 局部因果圖
  }
  ```
- **驗證**: 端到端測試

#### Step 49: 註冊到 RCAEval 系統
- **任務**: 在 `__init__.py` 和 `main.py` 中註冊新方法
- **檔案**: 
  - `RCAEval/e2e/__init__.py`
  - `main.py`
- **依賴**: Step 48
- **內容**: 
  - 在 `__init__.py` 中 import `pcmci_shapley`
  - 在 `main.py` 中添加到方法列表
- **驗證**: 可透過 main.py 呼叫

#### Step 50: 建立範例與文件
- **任務**: 建立使用範例和說明文件
- **檔案**: `RCAEval/e2e/pcmci_shapley_modules/README.md`
- **依賴**: Steps 48, 49
- **內容**:
  - 方法說明
  - 參數說明
  - 使用範例
  - 超參數調整指南
- **驗證**: 文件完整清晰

---

## 實作優先順序建議

### 第一階段（核心功能）
- Steps 1-8: 建立專案結構
- Steps 9-14: 實作預處理
- Steps 22-27: 實作 PCMCI 檢定
- Steps 33-36: 實作異常傳播
- Steps 43, 44, 47: 實作基礎評分排名
- Step 48: 整合主流程（簡化版）

### 第二階段（完整功能）
- Steps 15-21: 實作節點隔離
- Steps 28-32: 實作邊權融合
- Steps 38-42: 實作 Shapley 值計算
- Steps 45, 46: 實作進階評分

### 第三階段（優化與整合）
- Step 49: 系統註冊
- Step 50: 文件與範例
- Steps 12, 37, 41: 可選功能

---

## 技術難點與解決方案

### 難點 1: PCMCI 計算複雜度
- **問題**: PCMCI 在大規模節點上計算緩慢
- **解決**: 
  1. 嚴格限制局部節點數量 |U| ≤ 40
  2. 使用 Isolation Forest 提前篩選
  3. 可選：實作平行化

### 難點 2: Shapley 值計算成本
- **問題**: 精確 Shapley 值需要指數時間
- **解決**:
  1. 使用蒙特卡洛抽樣 (R=500-1000)
  2. 小規模時 (|U| ≤ 15) 使用精確計算
  3. 可考慮 TreeSHAP 等近似方法

### 難點 3: Trace Graph 整合
- **問題**: 不同資料集的 trace graph 格式可能不同
- **解決**:
  1. 定義統一的 trace graph 介面
  2. 實作多種 trace 來源的轉換器
  3. 若無 trace，降級為純統計方法

### 難點 4: 時間窗口與滯後選擇
- **問題**: τ_max 和窗口大小 W 需要調整
- **解決**:
  1. 提供自適應選擇機制
  2. 基於資料頻率自動推薦 τ_max
  3. 實作網格搜索工具

---

## 測試策略

### 單元測試
- 每個模組的核心函數都需要單元測試
- 使用 pytest 框架
- 測試覆蓋率目標 > 80%

### 整合測試
- 使用 RCAEval 現有的測試資料集
- 測試端到端流程
- 驗證輸出格式符合規範

### 性能測試
- 測試不同規模資料集的執行時間
- 記錄記憶體使用情況
- 確保在合理時間內完成（< 5 分鐘）

### 準確性測試
- 使用已知根因的資料集驗證
- 計算 Avg@K, Precision@K 等指標
- 與 baseline 方法比較

---

## 預期挑戰與風險

1. **資料品質問題**: 真實資料可能存在大量缺失值和噪音
   - 緩解：強化預處理和異常偵測

2. **超參數敏感性**: 方法涉及多個超參數
   - 緩解：提供自適應選擇和調參指南

3. **計算資源限制**: Shapley 計算可能消耗大量時間
   - 緩解：實作抽樣近似和早停機制

4. **方法泛化性**: 可能對某些類型的故障效果不佳
   - 緩解：在多個資料集上測試和調整

---

## 時間估算

- **階段 A**: 2-3 小時
- **階段 B**: 4-5 小時
- **階段 C**: 5-6 小時
- **階段 D**: 4-5 小時
- **階段 E**: 3-4 小時
- **階段 F**: 3-4 小時
- **階段 G**: 5-6 小時
- **階段 H**: 4-5 小時
- **階段 I**: 2-3 小時

**總計**: 約 32-41 小時（4-5 個工作日）

---

## 驗收標準

1. 所有 50 個步驟完成
2. 單元測試通過率 > 80%
3. 整合測試全部通過
4. 可透過 main.py 正常呼叫
5. 在至少 3 個資料集上測試成功
6. 文件完整，包含使用範例
7. 程式碼符合專案風格規範
8. 性能符合預期（執行時間 < 5 分鐘）

---

## 附錄：依賴關係圖

```
Step 1 → Step 2 → Step 3
         ↓        ↓
Step 4   Step 8   Step 48 → Step 49 → Step 50
↓        ↓        ↑
Steps    Steps    └─ Steps 14, 21, 27, 32, 36, 42, 47
9-14     15-47    
```

---

**計畫建立時間**: 2025-10-15  
**計畫版本**: v1.0  
**計畫狀態**: 待確認

---

## 下一步行動

1. 確認此計畫可行性
2. 設定開發環境
3. 建立 Git 分支 `feature/pcmci-shapley`
4. 開始階段 A 的實作

---

## 每步驟補充說明（Inputs/Outputs/Interfaces/Done/Risks）

為每個步驟補充可執行細節，便於協作與驗收。

### Steps 1–8：專案結構建立

- Step 1: 建立模組資料夾結構
  - Inputs: 目錄 `RCAEval/e2e/`
  - Outputs: 新增 `RCAEval/e2e/pcmci_shapley_modules/`
  - Interfaces: 無
  - Params: 無
  - Done: 目錄存在於 repo，納入 git
  - Risks: 權限/忽略規則導致未入庫

- Step 2: 建立模組 __init__.py
  - Inputs: 模組子檔名清單
  - Outputs: `__all__` 與主要類/函數匯出
  - Interfaces: `from .preprocessing import ...` 等
  - Params: 無
  - Done: `from RCAEval.e2e.pcmci_shapley_modules import *` 可成功
  - Risks: 循環匯入（避免在模組頂層做重運算）

- Step 3: 建立主入口檔案框架
  - Inputs: 既有 `@rca` 裝飾器模式（見 `RCAEval/e2e/pc_pagerank.py`）
  - Outputs: `pcmci_shapley(data, inject_time, dataset, **kwargs)` 雛形
  - Interfaces: 使用 `RCAEval.io.time_series.preprocess`
  - Params: `dk_select_useful`, `config`, `trace_graph`, `focus_node`
  - Done: 可回傳 dummy 結構 `{"adj":[],"node_names":[],"ranks":[]}`
  - Risks: 參數名需與 `main.py` 調用一致

- Step 4: 建立 utils.py 工具模組
  - Inputs: numpy/pandas 結構
  - Outputs: `min_max_normalize`, `robust_standardize`, `ensure_2d`, `safe_divide`
  - Interfaces: 純函數，無 side-effect
  - Params: `eps`, `clip_range`
  - Done: 單元測試通過、被其他模組成功 import
  - Risks: 浮點數精度/空集合處理

- Step 5: 研究現有 PCMCI 實作
  - Inputs: `RCAEval/graph_construction/pcmci.py`
  - Outputs: 記錄可重用 API（ParCorr 包裝、矩陣產生）
  - Interfaces: 保留 τ 資訊的需求差異
  - Params: `tau_max`, `pc_alpha`
  - Done: 文檔化重用與改造差異
  - Risks: 現有實作不保留 τ，需擴展

- Step 6: 研究 trace graph 獲取方式
  - Inputs: `RCAEval/e2e/tracerca.py`、資料集元資料
  - Outputs: 統一的 trace 介面定義（有向圖、可選權重）
  - Interfaces: `networkx.DiGraph`，節點命名與資料欄位對齊
  - Params: 無
  - Done: 能從資料集取得/模擬 trace 子圖
  - Risks: 各資料集節點命名不一致

- Step 7: 確認依賴套件
  - Inputs: `requirements.txt`
  - Outputs: 依賴列表與版本範圍
  - Interfaces: tigramite, sklearn, networkx
  - Params: 無
  - Done: `pip install -r requirements.txt` 成功
  - Risks: 版本衝突、M1/M2 架構相容性

- Step 8: 建立配置參數類別
  - Inputs: `plan.md` 公式中的超參數
  - Outputs: `PCMCIShapleyConfig`（含驗證）
  - Interfaces: 模組間傳遞 config 實例
  - Params: τ_max, α, θ1..θ3, K, R, λ, α_prop 等
  - Done: `config.validate()` 無錯
  - Risks: 參數和內部函數預設不一致

### Steps 9–14：資料預處理模組

- Step 9: 建立 preprocessing.py 模組框架
  - Inputs: pandas DataFrame（含 time 欄）
  - Outputs: 檔案骨架與型別標註
  - Interfaces: `RCAEval.io.time_series.preprocess`
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 10: 指標標準化函數
  - Inputs: 宽表 DataFrame
  - Outputs: 同維度標準化 DataFrame
  - Interfaces: 無
  - Params: `mad_eps` 防 0 MAD
  - Done: 單元測試涵蓋常數列、缺失值
  - Risks: 非數值欄位/全零方差

- Step 11: 異常偵測（Z-score）
  - Inputs: 標準化後 DataFrame, `threshold`
  - Outputs: 同維度 anomaly 分數 DataFrame
  - Interfaces: 無
  - Params: `threshold`、是否雙側
  - Done: 邊界值測試、NaN 處理
  - Risks: 極端值引發溢出

- Step 12: 異常偵測（SPOT 可選）
  - Inputs: 時序序列
  - Outputs: SPOT 分數 DataFrame
  - Interfaces: 第三方或自實作
  - Params: 初始窗口、置信水準
  - Done: 可被關閉回退至 Z-score
  - Risks: 依賴額外套件/參數敏感

- Step 13: 節點異常聚合
  - Inputs: anomaly_df、`metric_map: service→metrics`
  - Outputs: `a_s(t)` Series 或 DataFrame（索引為時間）
  - Interfaces: 提供聚合策略 hook（平均/加權/最大）
  - Params: 聚合方法與權重
  - Done: 對齊時間索引，無缺失
  - Risks: 指標命名映射錯誤

- Step 14: 預處理整合
  - Inputs: 原始 `data`、`config`
  - Outputs: `{normalized_df, anomaly_scores, node_anomaly}`
  - Interfaces: 後續模組統一讀取
  - Params: `anomaly_method` 選擇
  - Done: 對齊 `inject_time` 分界可用
  - Risks: 計算量偏大需向量化

### Steps 15–21：局部節點建構

- Step 15: node_isolation.py 骨架
  - Inputs: 無
  - Outputs: 模組檔與函數簽名
  - Interfaces: numpy, pandas, sklearn, networkx
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 16: 滯後互相關計算
  - Inputs: `a_F(t)`、`a_s(t)`、`tau_max`
  - Outputs: size=τ_max 的相關陣列
  - Interfaces: `numpy.corrcoef`
  - Params: 是否去均值/缺失處理
  - Done: 單元測試涵蓋短序列與 NaN
  - Risks: 低變異導致 NaN

- Step 17: 統計鄰域擴展
  - Inputs: 所有節點 `a_s(t)`、`focus_node`、`top_m1`
  - Outputs: 候選節點 `U0`
  - Interfaces: 排序選擇（最大值/平均值）
  - Params: 打分策略（max/mean over τ）
  - Done: U0 尺寸符合
  - Risks: 焦點節點缺失/對齊錯位

- Step 18: Isolation 特徵
  - Inputs: U0、`tau_max`、各節點滯後相關曲線
  - Outputs: `features` DataFrame（行=節點, 列=τ）
  - Interfaces: 給 IsolationForest 使用
  - Params: 是否標準化特徵
  - Done: 無 NaN/Inf
  - Risks: 特徵共線性影響模型

- Step 19: Isolation Forest 篩選
  - Inputs: `features`, `top_m2`
  - Outputs: U1、`I_s` 字典（[0,1]）
  - Interfaces: `sklearn.ensemble.IsolationForest`
  - Params: `n_estimators`, `contamination`, `random_state`
  - Done: 穩定可復現（固定 random_state）
  - Risks: contamination 不易估

- Step 20: Trace 補強
  - Inputs: U1、`trace_graph`、`u_max`
  - Outputs: U = U1 ∪ 1-hop ∪ {F}（裁剪至 `u_max`）
  - Interfaces: `networkx` 取 1-hop 鄰居
  - Params: 裁剪策略（優先度：U1 > 1-hop）
  - Done: |U| ≤ u_max
  - Risks: trace 缺失時退化處理

- Step 21: 節點建構整合
  - Inputs: 預處理結果、focus、trace、config
  - Outputs: `{local_nodes, isolation_scores}`
  - Interfaces: 後續 PCMCI 使用節點清單
  - Params: top_m1、top_m2、u_max
  - Done: local_nodes 與資料欄名一一對齊
  - Risks: 命名不一致導致落空

### Steps 22–27：PCMCI 局部因果檢定

- Step 22: pcmci_local.py 骨架
  - Inputs: 無
  - Outputs: 模組檔與函數簽名
  - Interfaces: tigramite.PCMCI
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 23: 建構時間序列矩陣
  - Inputs: `data`, `local_nodes`, `use_pca`
  - Outputs: `X` 矩陣（變數×時間）或 tigramite DataFrame
  - Interfaces: `tigramite.data_processing.DataFrame`
  - Params: PCA 維度、標準化
  - Done: 矩陣尺寸 > τ_max
  - Risks: 序列長度不足

- Step 24: PCMCI+ 包裝器
  - Inputs: `X`, `tau_max`, `alpha`
  - Outputs: `p_matrix`, `val_matrix`, `link_matrix`
  - Interfaces: ParCorr、`PCMCI.run_pcmci`
  - Params: `max_conds_dim`, `pc_alpha`
  - Done: 成功返回且耗時可接受
  - Risks: 常數列/奇異矩陣需健壯處理

- Step 25: 顯著邊提取
  - Inputs: `p_matrix`, `alpha`
  - Outputs: `E_PCMCI = [(src, dst, τ), ...]`
  - Interfaces: 與圖結構組裝
  - Params: 多重檢定（可選 FDR）
  - Done: τ≥1 的邊被正確收錄
  - Risks: 偽陽性過多

- Step 26: 邊強度計算
  - Inputs: `val_matrix`, `edges`
  - Outputs: `s_{i→j}` 字典
  - Interfaces: 供融合與懲罰
  - Params: 聚合策略 max/mean over τ
  - Done: 值域規範至 [0,1]
  - Risks: 噪音敏感

- Step 27: PCMCI 整合
  - Inputs: `data`, `local_nodes`, `config`
  - Outputs: `{edges, edge_strengths, pcmci_graph}`
  - Interfaces: `networkx.DiGraph`
  - Params: `tau_max`, `pcmci_alpha`
  - Done: 圖節點對齊 local_nodes
  - Risks: 方向解讀錯誤

### Steps 28–32：多源邊權融合

- Step 28: edge_fusion.py 骨架
  - Inputs: 無
  - Outputs: 模組檔
  - Interfaces: 無
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 29: Trace 邊權提取
  - Inputs: `trace_graph`, `local_nodes`
  - Outputs: `w_trace[(i,j)]`
  - Interfaces: 有向圖邊權（無權則設 1）
  - Params: 正規化策略
  - Done: 僅輸出 U×U 子圖邊
  - Risks: 方向與命名不一致

- Step 30: 三源權重融合
  - Inputs: `w_trace`, `s_{i→j}`, `I_s`
  - Outputs: `w[(i,j)]`
  - Interfaces: 後續正規化
  - Params: θ1, θ2, θ3（和為 1）
  - Done: 缺失來源視為 0
  - Risks: 權重比例敏感

- Step 31: 方向衝突懲罰
  - Inputs: `w`, `pcmci_edges`, `trace_edges`
  - Outputs: 懲罰後 `w`
  - Interfaces: 布林衝突旗標（可回傳）
  - Params: γ（0–1）
  - Done: 有衝突的反向邊被下調
  - Risks: 過度懲罰導致斷邊

- Step 32: 入邊歸一化
  - Inputs: `w`, `local_nodes`
  - Outputs: `ŵ`（每節點入邊和=1）
  - Interfaces: 供傳播
  - Params: `eps` 防零除
  - Done: 數值穩定
  - Risks: 孤立節點處理

### Steps 33–37：異常傳播

- Step 33: propagation.py 骨架
  - Inputs: 無
  - Outputs: 模組檔
  - Interfaces: 無
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 34: 初始異常強度
  - Inputs: `node_anomaly`, `T`
  - Outputs: `δ_s` 字典
  - Interfaces: log1p 安全計算
  - Params: 是否 clip
  - Done: 所有 local_nodes 皆有 δ
  - Risks: T 對齊錯誤

- Step 35: 單步傳播
  - Inputs: `h^(k)`, `ŵ`, `α_prop`
  - Outputs: `h^(k+1)`
  - Interfaces: 稀疏運算（可選）
  - Params: `alpha_prop` 0–1
  - Done: 尺寸與節點對齊
  - Risks: 浮點累積誤差

- Step 36: K 步傳播
  - Inputs: `δ`, `ŵ`, `K`, `α_prop`
  - Outputs: `{h_final, h_by_step}`
  - Interfaces: 停止條件（可選收斂/早停）
  - Params: `K` 5–10 建議
  - Done: h_final 非負且有限
  - Risks: 震盪/發散（歸一化與 α 控制）

- Step 37: 視覺化（可選）
  - Inputs: `h_by_step`, `graph`
  - Outputs: 圖檔或動畫
  - Interfaces: matplotlib/networkx
  - Params: 佈局方式
  - Done: 可視化生成成功
  - Risks: 大圖渲染緩慢

### Steps 38–42：Shapley 計算

- Step 38: shapley.py 骨架
  - Inputs: 無
  - Outputs: 模組檔
  - Interfaces: 無
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 39: 系統異常指標 v(C)
  - Inputs: 聯盟 C、`ŵ`、`K`、`α_prop`
  - Outputs: 單一標量 v(C)
  - Interfaces: 調用傳播模組（僅啟動 C 節點）
  - Params: 節點權重 `v_j`（預設 1）
  - Done: v(∅)=0，v(U)>0
  - Risks: 演算重複需快取

- Step 40: Shapley 抽樣
  - Inputs: `local_nodes`, `value_func`, `R`
  - Outputs: `φ̂_s` 字典
  - Interfaces: 隨機排列/子集抽樣
  - Params: `R`、亂數種子
  - Done: Σφ ≈ v(U)-v(∅)
  - Risks: 方差過大 → 提高 R

- Step 41: Shapley 精確（可選）
  - Inputs: 同上（|U|≤15）
  - Outputs: 精確 `φ_s`
  - Interfaces: 子集枚舉/動態規劃
  - Params: 無
  - Done: 與抽樣誤差可接受
  - Risks: 組合爆炸（需門檻）

- Step 42: Shapley 正規化
  - Inputs: `φ/φ̂`
  - Outputs: `φ̂ ∈ [0,1]`
  - Interfaces: utils.min_max_normalize
  - Params: 是否穩健正規化（去極值）
  - Done: 無 NaN/Inf
  - Risks: 極端值造成壓縮

### Steps 43–47：評分與排名

- Step 43: scoring.py 骨架
  - Inputs: 無
  - Outputs: 模組檔
  - Interfaces: 無
  - Params: 無
  - Done: 可 import
  - Risks: 無

- Step 44: 可達性計算 r_s
  - Inputs: `graph`, `ŵ`, `focus_node`
  - Outputs: `r_s` 字典
  - Interfaces: 最長路徑（乘積→log 轉加法）
  - Params: 最大路徑長度（防循環）
  - Done: r_s ∈ [0,1]
  - Risks: 環路需處理（DAG 化或限制步數）

- Step 45: 時間一致性懲罰 p_s
  - Inputs: 節點異常最早時間、因果路徑
  - Outputs: `p_s ∈ (0,1]`
  - Interfaces: 路徑掃描，累計違規邊權
  - Params: λ（懲罰力度）
  - Done: 無資料時 p_s=1
  - Risks: 時間戳對齊/時區

- Step 46: 綜合分數 Score_s
  - Inputs: `φ̂`, `r̂`, `â̂`（三者已正規化）
  - Outputs: `Score_s`
  - Interfaces: utils.min_max_normalize（前處理）
  - Params: α1, α2, α3（和=1）
  - Done: 值域合理，單調性符合期望
  - Risks: 權重需網格搜尋

- Step 47: 最終排名
  - Inputs: `Score_s`, `p_s`
  - Outputs: `ranks`（降序）
  - Interfaces: 與 `main.py` 輸出格式一致
  - Params: 是否輸出前 K
  - Done: 字串節點名與 `node_names` 對齊
  - Risks: 穩定排序與 ties 處理

### Steps 48–50：整合與註冊

- Step 48: 主函數整合
  - Inputs: `data`, `inject_time`, `dataset`, `trace_graph`, `config`
  - Outputs: `{adj, node_names, ranks, scores, shapley_values, local_graph}`
  - Interfaces: 串接前述所有模組
  - Params: 允許覆寫預設超參數
  - Done: 在樣例資料上端到端可跑
  - Risks: 例外處理需完整（@rca 包覆）

- Step 49: 系統註冊
  - Inputs: `__init__.py`, `main.py`
  - Outputs: 新方法可被 CLI/測試調用
  - Interfaces: `from RCAEval.e2e import pcmci_shapley`
  - Params: 方法名稱鍵值（如 'pcmci_shapley'）
  - Done: `python main.py --method pcmci_shapley ...` 可執行
  - Risks: import 條件分支（py310 檢查）

- Step 50: 範例與文件
  - Inputs: API 與超參數清單
  - Outputs: README、範例程式、FAQ
  - Interfaces: 連結 `plan.md` 與公式參考
  - Params: 無
  - Done: 新人可依文件跑通
  - Risks: 文件與程式碼漂移需同步維護
  - Status: ✅完成驗證

---

## 階段 J: PCMCI 穩健性增強與性能優化 (Steps 51-65)

### 問題診斷與根因分析

從實作過程中發現的關鍵問題：
1. **PCMCI 失敗頻繁**：遇到 "nans after standardizing, possibly constant array!" 錯誤
2. **Fallback 機制過度觸發**：導致所有資料集使用相同的 anomaly 排序，準確度完全一致
3. **數據質量問題**：常數列、NaN 值、低變異性數據未被妥善處理
4. **評分機制失效**：當圖為空時，所有節點評分都是 0（除了 focus_node）
5. **時間性能問題**：大資料集（31 節點）需要 2 小時以上

### Step 51: 實作 PCMCI 數據品質檢查器
- **任務**: 在 PCMCI 執行前進行全面的數據品質檢查
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `validate_pcmci_input(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, list, dict]`
- **依賴**: Step 27
- **內容**:
  - 檢測並移除常數列（標準差 < 1e-10）
  - 檢測並處理 NaN 值（插值或填充 0）
  - 檢測低變異性列（變異係數 < 閾值）
  - 檢測共線性列（相關係數 > 0.99）
  - 返回清理後的數據、可用列名、診斷報告
- **輸出**: 
  ```python
  {
      'cleaned_df': pd.DataFrame,
      'valid_columns': list,
      'diagnostics': {
          'removed_constant': list,
          'removed_low_var': list,
          'removed_collinear': list,
          'nan_filled': int
      }
  }
  ```
- **驗證**: 測試各種邊緣情況（全常數、部分 NaN、高共線性）
- **Risks**: 過度清理導致節點數量過少

### Step 52: 實作 PCMCI 多策略執行器
- **任務**: 實作多個備選 PCMCI 執行策略，逐步降級
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `run_pcmci_with_fallback(df: pd.DataFrame, columns: list, config: Config) -> dict`
- **依賴**: Step 51
- **策略優先序**:
  1. **標準 PCMCI+**: ParCorr + pc_alpha=0.05, tau_max=5
  2. **寬鬆 PCMCI**: ParCorr + pc_alpha=0.1, tau_max=3
  3. **極寬鬆 PCMCI**: ParCorr + pc_alpha=0.2, tau_max=2
  4. **簡單相關性**: Pearson 相關 + 顯著性檢定
  5. **Granger 因果**: 基於 VAR 模型
  6. **空圖降級**: 返回空邊集合但保留節點
- **內容**:
  - 每個策略有獨立的 try-except
  - 記錄使用的策略等級
  - 提供詳細的失敗診斷
- **驗證**: 模擬各種失敗情況，確保有策略可用
- **Risks**: 過度降級影響方法準確性

### Step 53: 實作數據預處理增強
- **任務**: 改善數據標準化和異常處理
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `robust_data_preparation(df: pd.DataFrame) -> pd.DataFrame`
- **依賴**: Step 51
- **內容**:
  - **Robust 標準化**: 使用中位數和 MAD 而非均值和標準差
  - **異常值裁剪**: 裁剪極端值到 [μ-3σ, μ+3σ]
  - **差分處理**: 對非平穩序列進行一階差分
  - **平滑處理**: 移動平均或高斯濾波降噪
  - **對數轉換**: 對正偏數據進行 log1p 轉換
- **驗證**: 測試改善標準化後的穩定性
- **Risks**: 過度處理損失原始信號

### Step 54: 實作替代因果推斷方法
- **任務**: 當 PCMCI 完全失敗時的替代方案
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `alternative_causal_inference(df: pd.DataFrame, columns: list, method: str) -> dict`
- **依賴**: Step 52
- **方法**:
  1. **Granger Causality**: 基於 VAR 模型的 F 檢定
  2. **Transfer Entropy**: 信息論方法
  3. **CCM (Convergent Cross Mapping)**: 非線性因果
  4. **簡單滯後相關**: 最大滯後相關作為代理
- **輸出格式**: 與 PCMCI 一致的邊和強度
- **驗證**: 與 PCMCI 結果比較一致性
- **Risks**: 替代方法的因果解釋性較弱

### Step 55: 實作邊融合降級策略
- **任務**: 當 PCMCI 邊為空時的智能降級
- **檔案**: `pcmci_shapley_modules/edge_fusion.py`
- **函數**: `fallback_edge_construction(trace_weights: dict, isolation_scores: dict, node_anomaly: dict, local_nodes: list) -> dict`
- **依賴**: Step 32
- **策略**:
  1. **僅 Trace**: 若有 trace_graph，使用 θ1=1.0
  2. **Trace + Isolation**: θ1=0.7, θ3=0.3
  3. **Isolation + Anomaly**: 基於異常相關性建邊
  4. **K-NN Graph**: 基於特徵相似度
  5. **完全圖降級**: 均勻權重連接所有節點
- **內容**: 動態調整 θ 權重
- **驗證**: 確保至少有部分邊存在
- **Risks**: 引入虛假邊

### Step 56: 實作評分機制健壯化
- **任務**: 改善空圖情況下的評分邏輯
- **檔案**: `pcmci_shapley_modules/scoring.py`
- **函數**: `robust_comprehensive_score(shapley: dict, reachability: dict, anomaly: dict, graph_empty: bool, config: Config) -> dict`
- **依賴**: Step 46
- **內容**:
  - **空圖模式**: 當 `total_weight == 0` 時
    - Shapley 權重 α1 = 0
    - 異常權重 α3 = 0.7
    - 新增「鄰域異常傳播」權重 α4 = 0.3
  - **稀疏圖模式**: 當邊數 < |U|/2 時
    - 動態調整權重：降低 Shapley，提高 Anomaly
  - **正常模式**: 使用標準權重
- **驗證**: 測試不同圖密度下的評分合理性
- **Risks**: 權重調整過於複雜

### Step 57: 實作鄰域異常傳播（無圖版本）
- **任務**: 在無因果圖時使用統計鄰域傳播
- **檔案**: `pcmci_shapley_modules/propagation.py`
- **函數**: `statistical_propagation(node_anomaly_ts: dict, focus_node: str, local_nodes: list, tau_max: int) -> dict`
- **依賴**: Step 36
- **內容**:
  - 基於滯後相關性構建虛擬傳播權重
  - 使用相關係數作為傳播強度
  - 考慮時間順序（只允許過去→現在）
- **公式**: `w_{i→j}^{stat} = max_τ |corr(a_i(t-τ), a_j(t))|`
- **驗證**: 與真實圖傳播結果比較
- **Risks**: 相關性≠因果性

### Step 58: 實作 PCMCI 參數自適應調整
- **任務**: 根據數據特性自動調整 PCMCI 參數
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `adaptive_pcmci_params(df: pd.DataFrame, config: Config) -> dict`
- **依賴**: Step 51
- **內容**:
  - **tau_max 調整**: 
    - 基於 ACF 衰減確定最大有效滯後
    - 小資料集 (T<500): tau_max = min(3, T/100)
    - 大資料集: tau_max = 5
  - **alpha 調整**:
    - 高維 (|U|>20): alpha = 0.1（寬鬆）
    - 低維: alpha = 0.05
  - **max_conds_dim 調整**:
    - 限制條件變數數量避免過擬合
- **驗證**: 測試自適應參數的效果
- **Risks**: 自動選擇可能不最優

### Step 59: 實作節點數量動態調整
- **任務**: 根據 PCMCI 性能動態調整局部節點數量
- **檔案**: `pcmci_shapley_modules/node_isolation.py`
- **函數**: `dynamic_node_selection(U0: list, U1: list, pcmci_success: bool, config: Config) -> list`
- **依賴**: Step 21, Step 52
- **策略**:
  - PCMCI 成功: 保持當前節點數
  - PCMCI 失敗且 |U|>20: 減少到 top 15
  - PCMCI 仍失敗且 |U|>10: 減少到 top 10
  - 記錄調整歷史供調試
- **驗證**: 測試迭代調整機制
- **Risks**: 過度縮減丟失重要節點

### Step 60: 實作時間性能優化
- **任務**: 優化 PCMCI 和 Shapley 計算的性能
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`, `shapley.py`
- **內容**:
  1. **PCMCI 優化**:
     - 使用 `max_conds_dim` 限制條件集大小
     - 啟用 PCMCI 的並行化選項
     - 設置合理的 `max_combinations`
  2. **Shapley 優化**:
     - 大節點集 (|U|>20): 強制使用抽樣，R=500
     - 中等節點集 (10<|U|≤20): R=1000
     - 小節點集 (|U|≤10): 精確計算
     - 實作早停：當 Shapley 值收斂時停止
  3. **快取機制**:
     - 快取重複的傳播計算
     - 快取 v(C) 函數調用
- **驗證**: 測試時間降低到可接受範圍（<10分鐘）
- **Risks**: 過度優化影響準確性

### Step 61: 實作超參數網格搜索工具
- **任務**: 提供自動化超參數調優工具
- **檔案**: `pcmci_shapley_modules/hyperopt.py` (新文件)
- **函數**: `grid_search_hyperparams(datasets: list, param_grid: dict) -> dict`
- **依賴**: Step 48
- **內容**:
  - 定義參數搜索空間
  - 使用交叉驗證評估參數組合
  - 支持並行搜索
  - 輸出最佳參數和性能報告
- **參數空間**:
  ```python
  {
      'tau_max': [2, 3, 5],
      'pcmci_alpha': [0.05, 0.1, 0.15],
      'theta1': [0.0, 0.3, 0.6],
      'theta2': [0.3, 0.5, 0.7],
      'theta3': [0.1, 0.2, 0.3],
      'K': [5, 8, 10],
      'alpha_prop': [0.7, 0.8, 0.9],
  }
  ```
- **驗證**: 在已知根因的資料集上驗證
- **Risks**: 計算量巨大

### Step 62: 實作參數推薦系統
- **任務**: 基於資料集特徵推薦參數
- **檔案**: `pcmci_shapley_modules/hyperopt.py`
- **函數**: `recommend_params(data: pd.DataFrame, dataset_type: str) -> PCMCIShapleyConfig`
- **依賴**: Step 61
- **規則**:
  - **微服務系統** (online-boutique, sock-shop):
    - 有 trace: θ1=0.6, θ2=0.3, θ3=0.1
    - 無 trace: θ1=0.0, θ2=0.8, θ3=0.2
  - **單體系統** (train-ticket):
    - tau_max=5, K=10
  - **高頻數據** (採樣 <10s):
    - tau_max=7-10
  - **低頻數據** (採樣 >60s):
    - tau_max=2-3
- **驗證**: 推薦參數的性能優於預設
- **Risks**: 規則過於簡化

### Step 63: 實作診斷與可視化工具
- **任務**: 提供詳細的診斷信息和可視化
- **檔案**: `pcmci_shapley_modules/diagnostics.py` (新文件)
- **函數**: `generate_diagnostic_report(pipeline_state: dict) -> dict`
- **依賴**: Step 48
- **內容**:
  - PCMCI 執行狀態（成功/失敗/降級）
  - 數據品質報告
  - 圖密度和連通性分析
  - Shapley 值分布
  - 評分權重實際使用情況
  - 執行時間分析
- **輸出格式**: JSON + 可選 HTML 報告
- **驗證**: 報告信息完整準確
- **Risks**: 額外開銷

### Step 64: 實作自動化測試套件
- **任務**: 建立全面的測試來驗證穩健性
- **檔案**: `tests/test_pcmci_shapley_robustness.py` (新文件)
- **依賴**: Steps 51-63
- **測試案例**:
  1. 全常數數據
  2. 部分 NaN 數據
  3. 高共線性數據
  4. 極短序列 (T<100)
  5. 極長序列 (T>10000)
  6. 單節點場景
  7. 大規模節點 (|U|>40)
  8. 無 trace 場景
  9. 空異常場景
- **驗證**: 所有測試通過，無異常拋出
- **Risks**: 測試覆蓋不完整

### Step 65: 批量資料集驗證與參數調優
- **任務**: 在所有資料集上驗證並調優參數
- **檔案**: `experiments/validate_pcmci_shapley.py` (新文件)
- **依賴**: Steps 61-64
- **內容**:
  1. 在所有 RCAEval 資料集上運行
  2. 記錄每個資料集的：
     - PCMCI 成功率
     - Fallback 觸發率
     - 準確度指標
     - 執行時間
  3. 分析失敗模式
  4. 調整預設參數
  5. 生成對比報告
- **目標**: 
  - Fallback 觸發率 < 20%
  - 準確度變異係數 > 0.3（證明不同資料集有不同結果）
  - 平均執行時間 < 5 分鐘
- **驗證**: 與其他 RCA 方法對比
- **Risks**: 某些資料集仍可能失敗

---

## 階段 K: 性能與準確度優化 (補充)

### Step 66: 實作增量式 PCMCI
- **任務**: 對大規模資料實作增量/分塊 PCMCI
- **策略**: 時間窗口滑動，避免一次處理全部數據
- **預期提升**: 時間降低 50-70%

### Step 67: 實作智能早停機制
- **任務**: 當檢測到收斂時提前停止迭代
- **應用**: Shapley 抽樣、異常傳播
- **預期提升**: 時間降低 20-30%

### Step 68: 實作結果快取
- **任務**: 對重複查詢快取結果
- **策略**: 基於數據指紋的 LRU 快取
- **預期提升**: 重複實驗速度提升 10 倍

---

## 更新後的時間估算

- **階段 J (Steps 51-65)**: 12-15 小時
  - Steps 51-54 (PCMCI 穩健性): 4-5 小時
  - Steps 55-57 (降級策略): 3-4 小時
  - Steps 58-60 (性能優化): 3-4 小時
  - Steps 61-65 (調優與測試): 2-3 小時
- **階段 K (Steps 66-68, 可選)**: 3-4 小時

**新增總計**: 約 15-19 小時（2-3 個工作日）

**項目總計**: 約 47-60 小時（6-8 個工作日）

---

## 更新後的驗收標準

原有標準保持，新增：
9. PCMCI fallback 觸發率 < 20%
10. 不同資料集準確度變異係數 > 0.3
11. 平均執行時間 < 5 分鐘（online-boutique）
12. 平均執行時間 < 15 分鐘（train-ticket）
13. 所有邊緣情況測試通過
14. 穩健性測試覆蓋率 > 90%

---

## 補充的技術難點與解決方案

### 難點 5: PCMCI 在真實數據上頻繁失敗
- **問題**: 常數列、NaN、低變異性導致標準化失敗
- **解決**:
  1. 多層數據品質檢查
  2. 多策略降級執行
  3. 替代因果推斷方法
  4. 智能參數自適應

### 難點 6: Fallback 導致準確度相同
- **問題**: 空圖情況下評分機制失效
- **解決**:
  1. 改善邊融合降級策略
  2. 統計鄰域傳播作為替代
  3. 動態評分權重調整
  4. 保證至少部分圖結構存在

### 難點 7: 大資料集性能瓶頸
- **問題**: 31 節點需要 2+ 小時
- **解決**:
  1. 動態節點數量調整
  2. PCMCI 參數優化（max_conds_dim）
  3. Shapley 抽樣數量自適應
  4. 快取和早停機制

---

## 實作優先順序更新

### 第一階段（核心功能）- 已完成
- Steps 1-50: 基礎實作

### 第二階段（穩健性增強）- **當務之急**
- Steps 51-54: PCMCI 數據處理和降級
- Steps 55-57: 降級策略和替代方案
- Step 64: 自動化測試

### 第三階段（性能優化）
- Steps 58-60: 參數調整和性能優化
- Steps 66-68: 進階優化

### 第四階段（調優與驗證）
- Steps 61-63: 超參數調優
- Step 65: 批量驗證

