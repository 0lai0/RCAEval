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

## 階段 K: 聯合篩選器與高精度 Fallback 重構 (Steps 66-95)

### 核心理念

當前實作揭示了一個重要發現：
1. **PCMCI 頻繁失敗**導致 Fallback 機制被大量觸發
2. **Fallback 異常分數表現驚人**：Avg@5-DISK=1.0, Avg@5-SOCKET=0.97
3. **新策略**：將高精度 Fallback 與 SPOT 極值理論結合，形成「聯合篩選器」

### 設計目標

1. **修復 PCMCI 穩定性**：解決常數列和數據質量問題
2. **實現聯合篩選器**：Fallback 異常分數 + SPOT 極值理論
3. **優化工作流程**：篩選 → 因果建圖 → 歸因分析

---

### 子階段 K1: PCMCI 穩定性增強 (Steps 66-70) ✅

#### Step 66: 實作數據驗證前處理器
- **任務**: 在 PCMCI 執行前進行嚴格的數據驗證
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `validate_and_clean_for_pcmci(df: pd.DataFrame, columns: list, config: Config) -> tuple`
- **依賴**: Step 27
- **內容**:
  ```python
  def validate_and_clean_for_pcmci(df, columns, config):
      """
      返回: (cleaned_df, valid_columns, diagnostics)
      """
      diagnostics = {
          'removed_constant': [],
          'removed_low_variance': [],
          'removed_collinear': [],
          'nan_handled': 0,
          'input_shape': df.shape,
          'output_shape': None
      }
      
      # 1. 檢測並移除常數列（方差 < 1e-9）
      variances = df.var()
      constant_cols = variances[variances < 1e-9].index.tolist()
      diagnostics['removed_constant'] = constant_cols
      df_clean = df.drop(columns=constant_cols)
      
      # 2. 檢測並移除低變異列（變異係數 < 0.01）
      cv = df_clean.std() / (df_clean.mean().abs() + 1e-10)
      low_var_cols = cv[cv < 0.01].index.tolist()
      diagnostics['removed_low_variance'] = low_var_cols
      df_clean = df_clean.drop(columns=low_var_cols)
      
      # 3. 處理 NaN 值（前向填充 + 後向填充 + 零填充）
      nan_count = df_clean.isna().sum().sum()
      df_clean = df_clean.fillna(method='ffill').fillna(method='bfill').fillna(0)
      diagnostics['nan_handled'] = nan_count
      
      # 4. 檢測共線性（相關係數 > 0.99）
      if len(df_clean.columns) > 1:
          corr_matrix = df_clean.corr().abs()
          upper_tri = corr_matrix.where(
              np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
          )
          collinear_cols = [
              col for col in upper_tri.columns 
              if any(upper_tri[col] > 0.99)
          ]
          diagnostics['removed_collinear'] = collinear_cols[:len(collinear_cols)//2]
          df_clean = df_clean.drop(columns=diagnostics['removed_collinear'])
      
      diagnostics['output_shape'] = df_clean.shape
      valid_columns = df_clean.columns.tolist()
      
      return df_clean, valid_columns, diagnostics
  ```
- **目標**: 確保送入 PCMCI 的數據至少有 2 個有效變量，且無常數列
- **驗證**: 
  - 測試全常數數據：應返回空列表
  - 測試部分常數：應正確移除
  - 測試 NaN 數據：應完全填充
- **Risks**: 過度清理可能導致節點數不足
- **是否完成**: O

#### Step 67: 實作 PCMCI 多策略執行器
- **任務**: 實現漸進式降級策略
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `run_pcmci_with_progressive_fallback(df: pd.DataFrame, columns: list, config: Config) -> dict`
- **依賴**: Step 66
- **內容**:
  ```python
  def run_pcmci_with_progressive_fallback(df, columns, config):
      """
      策略優先序:
      1. 標準 PCMCI+ (alpha=0.05, tau_max=5, max_conds_dim=3)
      2. 寬鬆 PCMCI (alpha=0.1, tau_max=3, max_conds_dim=2)
      3. 極寬鬆 PCMCI (alpha=0.2, tau_max=2, max_conds_dim=1)
      4. 簡單滯後相關 (Pearson + 顯著性檢定)
      5. 空圖（返回空邊集合但保留節點）
      """
      strategies = [
          {'name': 'standard', 'alpha': 0.05, 'tau_max': 5, 'max_conds_dim': 3},
          {'name': 'relaxed', 'alpha': 0.1, 'tau_max': 3, 'max_conds_dim': 2},
          {'name': 'very_relaxed', 'alpha': 0.2, 'tau_max': 2, 'max_conds_dim': 1},
      ]
      
      for strategy in strategies:
          try:
              logger.info(f"Trying PCMCI strategy: {strategy['name']}")
              result = run_pcmci_plus(
                  df, 
                  tau_max=strategy['tau_max'],
                  alpha=strategy['alpha'],
                  max_conds_dim=strategy['max_conds_dim']
              )
              logger.info(f"PCMCI succeeded with strategy: {strategy['name']}")
              result['strategy_used'] = strategy['name']
              return result
          except Exception as e:
              logger.warning(f"PCMCI strategy {strategy['name']} failed: {e}")
              continue
      
      # 最後降級到簡單相關性
      logger.warning("All PCMCI strategies failed, using lagged correlation")
      return fallback_to_lagged_correlation(df, columns, config)
  ```
- **目標**: PCMCI 成功率 > 80%
- **驗證**: 在所有測試數據集上運行，記錄策略使用分佈
- **Risks**: 過度降級影響因果推斷質量
- **是否完成**: O

#### Step 68: 實作簡單相關性降級方法
- **任務**: 當 PCMCI 完全失敗時的因果代理
- **檔案**: `pcmci_shapley_modules/pcmci_local.py`
- **函數**: `fallback_to_lagged_correlation(df: pd.DataFrame, columns: list, config: Config) -> dict`
- **依賴**: Step 67
- **內容**:
  ```python
  def fallback_to_lagged_correlation(df, columns, config):
      """
      使用滯後相關性作為因果關係的代理
      保持與 PCMCI 相同的輸出格式
      """
      edges = []
      edge_strengths = {}
      
      for i, col_i in enumerate(columns):
          for j, col_j in enumerate(columns):
              if i == j:
                  continue
              
              # 計算滯後相關性
              max_corr = 0.0
              best_tau = 0
              for tau in range(1, config.tau_max + 1):
                  if len(df) <= tau:
                      continue
                  
                  x = df[col_i].values[:-tau]
                  y = df[col_j].values[tau:]
                  
                  if np.std(x) > 0 and np.std(y) > 0:
                      corr = abs(np.corrcoef(x, y)[0, 1])
                      if corr > max_corr:
                          max_corr = corr
                          best_tau = tau
              
              # 使用閾值篩選（相當於 alpha 檢定）
              threshold = 0.3  # 可調整
              if max_corr > threshold:
                  edges.append((i, j, best_tau))
                  edge_strengths[(i, j)] = max_corr
      
      return {
          'edges': edges,
          'edge_strengths': edge_strengths,
          'columns': columns,
          'strategy_used': 'lagged_correlation'
      }
  ```
- **目標**: 提供最後的安全網
- **驗證**: 與 PCMCI 結果比較一致性（在 PCMCI 可用時）
- **Risks**: 相關性不等於因果性
- **是否完成**: O

#### Step 69: 更新主流程中的 PCMCI 調用
- **任務**: 在 `pcmci_shapley.py` 中集成新的驗證和降級邏輯
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Steps 66-68
- **內容**:
  ```python
  # 替換原有的 PCMCI 調用
  service_df = pd.DataFrame({s: node_anomaly_ts.get(s, pd.Series(dtype=float)).values for s in U})
  
  # 新增：數據驗證和清理
  service_df_clean, valid_columns, diagnostics = pcmci_mod.validate_and_clean_for_pcmci(
      service_df, list(service_df.columns), cfg
  )
  logger.info(f"Data validation: {diagnostics}")
  
  if len(valid_columns) < 2:
      logger.warning(f"Insufficient valid columns ({len(valid_columns)}), triggering empty graph")
      pcmci_res = {'edges': [], 'edge_strengths': {}, 'columns': valid_columns, 'strategy_used': 'empty'}
  else:
      logger.info(f"PCMCI input shape after cleaning: {service_df_clean.shape}")
      pcmci_res = pcmci_mod.run_pcmci_with_progressive_fallback(
          service_df_clean, valid_columns, cfg
      )
      logger.info(f"PCMCI completed with strategy: {pcmci_res.get('strategy_used', 'unknown')}")
  ```
- **目標**: 減少 PCMCI 失敗率到 < 20%
- **驗證**: 運行完整測試套件，檢查失敗率
- **Risks**: 增加代碼複雜度
- **是否完成**: O

#### Step 70: 批量驗證 PCMCI 穩定性
- **任務**: 在所有數據集上測試改進效果
- **檔案**: `tests/test_pcmci_stability.py`
- **依賴**: Step 69
- **內容**:
  - 運行所有 RCAEval 數據集
  - 記錄每個數據集的 PCMCI 策略使用情況
  - 統計失敗率、降級率
  - 生成穩定性報告
- **目標**:
  - PCMCI 成功率（使用任意策略）> 80%
  - 空圖率 < 20%
- **驗證**: 與修改前的日誌對比
- **Risks**: 某些數據集可能天生不適合 PCMCI
- **是否完成**: O

---

### 子階段 K2: SPOT 極值理論集成 (Steps 71-75) ✅

#### Step 71: 研究並選擇 SPOT 實現
- **任務**: 選擇合適的 SPOT 算法實現
- **檔案**: 文檔
- **依賴**: 無
- **內容**:
  - 評估現有庫：`pyod`, `spot`, `adtk`
  - 決定自己實現或使用第三方庫
  - 確定 SPOT 參數：初始窗口大小、風險參數 q
- **目標**: 選定實現方案
- **驗證**: 在樣本數據上測試 SPOT 效果
- **Risks**: 第三方庫可能不穩定或不兼容
- **是否完成**: O

#### Step 72: 實作 SPOT 異常檢測模組
- **任務**: 實現 SPOT 極值理論異常檢測
- **檔案**: `pcmci_shapley_modules/spot_detector.py` (新文件)
- **函數**: `spot_anomaly_detection(data: pd.DataFrame, config: Config) -> pd.DataFrame`
- **依賴**: Step 71
- **內容**:
  ```python
  def spot_anomaly_detection(data: pd.DataFrame, config: Config) -> pd.DataFrame:
      """
      對每個時間序列應用 SPOT 算法
      
      Returns:
          DataFrame: 每個指標的 SPOT 異常分數（罕見度）
      """
      from pyod.models.spot import SPOT  # 假設使用 pyod
      
      spot_scores = {}
      for col in data.columns:
          if col == 'time':
              continue
          
          series = data[col].values
          
          # 初始化 SPOT
          spot = SPOT(q=config.spot_risk_param)  # q=0.001 表示 0.1% 的極值
          spot.fit(series[:config.spot_init_window])
          
          # 在線檢測
          scores = []
          for i in range(config.spot_init_window, len(series)):
              spot.step(series[i])
              score = spot.probability(series[i])  # 返回 p-value 或罕見度
              scores.append(1 - score)  # 轉為異常分數
          
          # 填充初始窗口
          full_scores = [0.0] * config.spot_init_window + scores
          spot_scores[col] = full_scores
      
      result = pd.DataFrame(spot_scores)
      if 'time' in data.columns:
          result.insert(0, 'time', data['time'].values)
      
      return result
  ```
- **目標**: 為每個指標生成 SPOT 異常分數
- **驗證**: 
  - 測試能夠捕捉極端值
  - 測試對正常波動的容忍度
- **Risks**: 參數敏感，需要調優
- **是否完成**: O

#### Step 73: 實作節點級 SPOT 聚合
- **任務**: 將指標級 SPOT 分數聚合到服務級
- **檔案**: `pcmci_shapley_modules/spot_detector.py`
- **函數**: `aggregate_spot_scores(spot_df: pd.DataFrame, metric_map: dict) -> pd.Series`
- **依賴**: Step 72
- **內容**:
  ```python
  def aggregate_spot_scores(spot_df: pd.DataFrame, metric_map: dict) -> pd.Series:
      """
      與 aggregate_node_anomaly 類似的邏輯
      使用最後時間點的分數（或最大值）
      """
      scores = {}
      for service, metrics in metric_map.items():
          cols = [m for m in metrics if m in spot_df.columns]
          if not cols:
              scores[service] = 0.0
              continue
          
          # 使用最後時間點的最大 SPOT 分數
          scores[service] = spot_df[cols].iloc[-1].max()
      
      return pd.Series(scores)
  ```
- **目標**: 每個服務有一個 SPOT 罕見度分數
- **驗證**: 分數分佈合理（大部分接近 0，少數接近 1）
- **Risks**: 聚合策略可能不最優
- **是否完成**: O

#### Step 74: 集成 SPOT 到預處理流程
- **任務**: 在 `preprocess_data` 中添加 SPOT 分析
- **檔案**: `pcmci_shapley_modules/preprocessing.py`
- **依賴**: Step 73
- **內容**:
  ```python
  def preprocess_data(data: pd.DataFrame, config: PCMCIShapleyConfig) -> Dict[str, Any]:
      # ... 現有邏輯 ...
      
      # 新增：SPOT 極值理論分析
      if config.enable_spot:
          from .spot_detector import spot_anomaly_detection, aggregate_spot_scores
          
          spot_scores_df = spot_anomaly_detection(norm, config)
          node_spot = aggregate_spot_scores(spot_scores_df, metric_map)
      else:
          spot_scores_df = None
          node_spot = pd.Series({s: 0.0 for s in metric_map.keys()})
      
      return {
          'normalized_df': norm,
          'anomaly_scores': anomalies,
          'node_anomaly': node_anomaly,
          'node_anomaly_ts': node_anomaly_ts,
          'metric_mapping': metric_map,
          'spot_scores': spot_scores_df,  # 新增
          'node_spot': node_spot,  # 新增
      }
  ```
- **目標**: 預處理輸出包含 SPOT 分數
- **驗證**: 檢查輸出結構正確
- **Risks**: 增加預處理時間
- **是否完成**: O

#### Step 75: 添加 SPOT 配置參數
- **任務**: 在配置類中添加 SPOT 相關參數
- **檔案**: `pcmci_shapley_modules/config.py`
- **依賴**: Step 71
- **內容**:
  ```python
  @dataclass
  class PCMCIShapleyConfig:
      # ... 現有參數 ...
      
      # SPOT 極值理論參數（新增）
      enable_spot: bool = True
      spot_risk_param: float = 0.001  # q 參數：0.001 表示捕捉 0.1% 的極值
      spot_init_window: int = 200  # 初始訓練窗口大小
      spot_depth: int = 10  # 極值池深度
      
      # 聯合篩選器參數（新增）
      joint_screener_enabled: bool = True
      joint_weight_fallback: float = 0.5  # Fallback 異常分數權重
      joint_weight_spot: float = 0.5  # SPOT 罕見度權重
      joint_top_n: int = 15  # 篩選後保留的節點數
  ```
- **目標**: 可配置 SPOT 行為
- **驗證**: 參數驗證通過
- **Risks**: 參數過多增加複雜度
- **是否完成**: O

---

### 子階段 K3: 聯合篩選器實現 (Steps 76-80) ✅

#### Step 76: 實作聯合篩選器核心邏輯
- **任務**: 實現 Fallback + SPOT 的加權融合
- **檔案**: `pcmci_shapley_modules/joint_screener.py` (新文件)
- **函數**: `joint_screening(node_anomaly: pd.Series, node_spot: pd.Series, config: Config) -> tuple`
- **依賴**: Steps 74, 75
- **內容**:
  ```python
  def joint_screening(
      node_anomaly: pd.Series, 
      node_spot: pd.Series, 
      config: Config
  ) -> tuple[list, dict]:
      """
      聯合篩選器：Fallback 異常分數 + SPOT 罕見度
      
      Returns:
          (selected_nodes, fusion_scores)
      """
      from .utils import min_max_normalize
      
      # 1. 歸一化兩個分數
      anomaly_norm = min_max_normalize(node_anomaly.to_dict())
      spot_norm = min_max_normalize(node_spot.to_dict())
      
      # 2. 加權融合
      all_nodes = set(anomaly_norm.keys()) | set(spot_norm.keys())
      fusion_scores = {}
      for node in all_nodes:
          score_a = anomaly_norm.get(node, 0.0)
          score_s = spot_norm.get(node, 0.0)
          fusion_scores[node] = (
              config.joint_weight_fallback * score_a + 
              config.joint_weight_spot * score_s
          )
      
      # 3. 排序並選擇 Top-N
      sorted_nodes = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
      selected_nodes = [node for node, _ in sorted_nodes[:config.joint_top_n]]
      
      logger.info(f"Joint screener: {len(all_nodes)} -> {len(selected_nodes)} nodes")
      logger.info(f"Top-5 fusion scores: {dict(sorted_nodes[:5])}")
      
      return selected_nodes, fusion_scores
  ```
- **目標**: 輸出高質量候選節點集
- **驗證**: 
  - 測試融合分數合理性
  - 測試選出的節點包含真實根因（在已知根因的數據集上）
- **Risks**: 權重選擇可能不最優
- **是否完成**: O

#### Step 77: 集成聯合篩選器到主流程
- **任務**: 在 `pcmci_shapley.py` 中使用聯合篩選器
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Step 76
- **內容**:
  ```python
  # 在步驟 2.5（剪枝）之後，步驟 3（焦點節點確定）之前
  
  # 2.7) 聯合篩選器（可選，與剪枝二選一或組合使用）
  if cfg.joint_screener_enabled:
      from .pcmci_shapley_modules import joint_screener as js_mod
      
      screened_nodes, fusion_scores = js_mod.joint_screening(
          pp.get("node_anomaly", pd.Series()),
          pp.get("node_spot", pd.Series()),
          cfg
      )
      
      logger.info(f"Joint screening: {len(pp.get('metric_mapping', {}))} -> {len(screened_nodes)} nodes")
      
      # 更新 node_anomaly_ts 只保留篩選後的節點
      node_anomaly_ts_original = pp.get("node_anomaly_ts", {})
      node_anomaly_ts = {k: v for k, v in node_anomaly_ts_original.items() 
                         if k in screened_nodes}
  else:
      # 使用原有的剪枝邏輯
      node_anomaly_ts = pp.get("node_anomaly_ts", {})
  ```
- **目標**: 減少送入 PCMCI 的節點數，提高質量
- **驗證**: 
  - 檢查篩選後節點數量合理
  - 檢查 PCMCI 成功率是否提升
- **Risks**: 可能誤殺重要節點
- **是否完成**: O

#### Step 78: 實作篩選器效果評估工具
- **任務**: 評估聯合篩選器的效果
- **檔案**: `pcmci_shapley_modules/joint_screener.py`
- **函數**: `evaluate_screening_quality(selected_nodes: list, ground_truth: str, all_nodes: list) -> dict`
- **依賴**: Step 76
- **內容**:
  ```python
  def evaluate_screening_quality(
      selected_nodes: list, 
      ground_truth: str, 
      all_nodes: list
  ) -> dict:
      """
      評估篩選器質量
      
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
  ```
- **目標**: 量化篩選器質量
- **驗證**: 在測試數據集上運行
- **Risks**: 需要已知根因的數據集
- **是否完成**: O

#### Step 79: 實作篩選器可視化
- **任務**: 可視化篩選過程
- **檔案**: `pcmci_shapley_modules/joint_screener.py`
- **函數**: `visualize_screening(anomaly_scores: dict, spot_scores: dict, fusion_scores: dict, selected_nodes: list) -> None`
- **依賴**: Step 76
- **內容**:
  - 繪製三個分數的散點圖
  - 標記被選中和未被選中的節點
  - 突出顯示真實根因（如果已知）
- **目標**: 幫助理解篩選邏輯
- **驗證**: 生成可讀的可視化圖表
- **Risks**: 可選功能，非必需
- **是否完成**: O

#### Step 80: 批量測試聯合篩選器
- **任務**: 在所有數據集上測試篩選器效果
- **檔案**: `tests/test_joint_screener.py`
- **依賴**: Steps 76-78
- **內容**:
  - 運行所有數據集
  - 記錄篩選前後的節點數
  - 記錄 Recall（真實根因保留率）
  - 記錄 PCMCI 成功率變化
  - 生成對比報告
- **目標**:
  - Recall > 95%（真實根因幾乎總是被保留）
  - 節點減少率 > 50%
  - PCMCI 成功率提升
- **驗證**: 與未使用篩選器的版本對比
- **Risks**: 某些數據集可能不適合激進篩選
- **是否完成**: O

---

### 子階段 K4: Fallback 機制優化 (Steps 81-85) ✅

#### Step 81: 重新設計 Fallback 觸發邏輯
- **任務**: 更智能的 Fallback 觸發條件
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Steps 69, 77
- **內容**:
  ```python
  # 原有觸發條件
  # fallback_triggered = (not metric_ranks or all(mr is None for mr in metric_ranks)) or total_weight == 0.0
  
  # 新的觸發條件（更細緻）
  def should_trigger_fallback(pcmci_res, norm_w, metric_ranks, cfg):
      """
      決定是否觸發 Fallback
      
      Returns:
          (should_fallback: bool, reason: str)
      """
      reasons = []
      
      # 1. PCMCI 完全失敗（空圖）
      if len(pcmci_res['edges']) == 0:
          reasons.append("pcmci_empty")
      
      # 2. 融合後權重總和為 0
      total_weight = sum(norm_w.values()) if norm_w else 0.0
      if total_weight == 0.0:
          reasons.append("total_weight_zero")
      
      # 3. 圖過於稀疏（邊數 < 節點數 / 2）
      num_nodes = len(set([i for i, j in norm_w.keys()] + [j for i, j in norm_w.keys()]))
      if len(norm_w) < num_nodes / 2:
          reasons.append("graph_too_sparse")
      
      # 4. metric_ranks 為空或全 None
      if not metric_ranks or all(mr is None for mr in metric_ranks):
          reasons.append("metric_ranks_invalid")
      
      # 5. 使用了非標準 PCMCI 策略
      if pcmci_res.get('strategy_used') in ['very_relaxed', 'lagged_correlation', 'empty']:
          reasons.append(f"pcmci_degraded_{pcmci_res.get('strategy_used')}")
      
      # 決策：如果有任何一個嚴重原因，觸發 Fallback
      severe_reasons = ['pcmci_empty', 'total_weight_zero', 'metric_ranks_invalid']
      should_fallback = any(r in reasons for r in severe_reasons)
      
      # 如果圖稀疏但不為空，可以考慮部分 Fallback（混合模式）
      if 'graph_too_sparse' in reasons and not should_fallback:
          should_fallback = cfg.fallback_on_sparse_graph
          reasons.append("sparse_graph_policy")
      
      reason_str = ", ".join(reasons) if reasons else "none"
      return should_fallback, reason_str
  ```
- **目標**: 更精確地判斷何時需要 Fallback
- **驗證**: 測試各種邊緣情況
- **Risks**: 邏輯過於複雜
- **是否完成**: O

#### Step 82: 實作混合模式（部分 Fallback）
- **任務**: 當圖稀疏但不為空時，混合使用 PCMCI 和 Fallback
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Step 81
- **內容**:
  ```python
  def hybrid_ranking(
      ranks_from_pcmci: list,
      ranks_from_fallback: list,
      pcmci_confidence: float,
      cfg: Config
  ) -> list:
      """
      混合排序：根據 PCMCI 信心度混合兩種排序
      
      Args:
          pcmci_confidence: 0-1，表示對 PCMCI 結果的信心
          cfg.hybrid_threshold: 當 confidence < threshold 時使用混合
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
          while i < len(ranks_from_pcmci) or j < len(ranks_from_fallback):
              if i < len(ranks_from_pcmci) and ranks_from_pcmci[i] not in hybrid:
                  hybrid.append(ranks_from_pcmci[i])
              i += 1
              if j < len(ranks_from_fallback) and ranks_from_fallback[j] not in hybrid:
                  hybrid.append(ranks_from_fallback[j])
              j += 1
          return hybrid
  ```
- **目標**: 更靈活地利用兩種方法的優勢
- **驗證**: 在稀疏圖數據上測試
- **Risks**: 增加複雜度，可能不一定提升效果
- **是否完成**: O

#### Step 83: 優化 Fallback 異常分數計算
- **任務**: 改進 Fallback 使用的異常分數
- **檔案**: `pcmci_shapley_modules/preprocessing.py`
- **依賴**: Step 14
- **內容**:
  - 考慮時間維度：不僅使用最後時間點，還考慮異常持續時間
  - 考慮異常強度：峰值 vs 平均值
  - 考慮異常模式：突增 vs 持續異常
  ```python
  def enhanced_anomaly_score(anomaly_ts: pd.Series, timestamp: int) -> float:
      """
      增強的異常分數計算
      
      考慮：
      1. 最後時間點分數（即時性）
      2. 近期平均分數（持續性）
      3. 近期最大分數（嚴重性）
      4. 異常突增程度（變化率）
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
          prev_avg = anomaly_ts.iloc[-window:-1].mean()
          change_rate = (instant_score - prev_avg) / (prev_avg + 1e-10)
          change_score = min(1.0, max(0.0, change_rate))
      else:
          change_score = 0.0
      
      # 加權融合
      final_score = (
          0.4 * instant_score +
          0.3 * persistent_score +
          0.2 * severity_score +
          0.1 * change_score
      )
      
      return final_score
  ```
- **目標**: 提高 Fallback 排序的準確性
- **驗證**: 對比原有和增強版本的 Avg@K 指標
- **Risks**: 過度工程化
- **是否完成**: O

#### Step 84: 實作 Fallback 置信度評估
- **任務**: 評估 Fallback 結果的可靠性
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Step 83
- **內容**:
  ```python
  def compute_fallback_confidence(
      anomaly_scores: dict,
      spot_scores: dict,
      data_quality: dict
  ) -> float:
      """
      計算 Fallback 結果的置信度
      
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
          data_conf = 1.0 - data_quality.get('removed_constant_rate', 0.0)
          confidence_factors.append(data_conf)
      
      # 綜合置信度
      if confidence_factors:
          return np.mean(confidence_factors)
      else:
          return 0.5  # 預設中等置信度
  ```
- **目標**: 為 Fallback 結果提供可信度指標
- **驗證**: 檢查置信度與實際準確率的相關性
- **Risks**: 置信度計算可能不準確
- **是否完成**: O

#### Step 85: 添加 Fallback 相關配置
- **任務**: 在配置中添加 Fallback 相關參數
- **檔案**: `pcmci_shapley_modules/config.py`
- **依賴**: Steps 81-84
- **內容**:
  ```python
  @dataclass
  class PCMCIShapleyConfig:
      # ... 現有參數 ...
      
      # Fallback 優化參數（新增）
      fallback_on_sparse_graph: bool = True  # 圖稀疏時是否觸發 Fallback
      fallback_sparse_threshold: float = 0.5  # 邊數/節點數 < 此值視為稀疏
      
      # 混合模式參數（新增）
      enable_hybrid_mode: bool = True
      hybrid_confidence_threshold: float = 0.7  # > 此值用 PCMCI，< 1-此值用 Fallback
      
      # 增強異常分數參數（新增）
      enhanced_anomaly_weights: tuple = (0.4, 0.3, 0.2, 0.1)  # (即時, 持續, 嚴重, 變化)
  ```
- **目標**: 可配置 Fallback 行為
- **驗證**: 參數驗證通過
- **Risks**: 參數過多
- **是否完成**: O

---

### 子階段 K5: 端到端集成與測試 (Steps 86-90) ✅

#### Step 86: 更新主流程邏輯
- **任務**: 整合所有新功能到主流程
- **檔案**: `RCAEval/e2e/pcmci_shapley.py`
- **依賴**: Steps 69, 77, 81-85
- **內容**: 完整重構主流程，整合：
  - 數據驗證與清理
  - SPOT 極值理論
  - 聯合篩選器
  - 多策略 PCMCI
  - 智能 Fallback 觸發
  - 混合模式排序
- **目標**: 流程清晰，邏輯正確
- **驗證**: 端到端測試通過
- **Risks**: 集成可能引入新 bug
- **是否完成**: X

#### Step 87: 實作全面的日誌記錄
- **任務**: 增強日誌記錄，便於調試和分析
- **檔案**: 所有相關模組
- **依賴**: Step 86
- **內容**:
  - 記錄每個階段的輸入輸出尺寸
  - 記錄 PCMCI 策略使用情況
  - 記錄篩選前後的節點數
  - 記錄 Fallback 觸發原因
  - 記錄置信度評估結果
- **目標**: 日誌完整詳細，易於分析
- **驗證**: 檢查日誌格式和內容
- **Risks**: 過多日誌影響性能
- **是否完成**: X

#### Step 88: 創建完整測試套件
- **任務**: 建立全面的測試
- **檔案**: `tests/test_pcmci_shapley_integrated.py`
- **依賴**: Step 86
- **內容**:
  - 單元測試：每個新增函數
  - 集成測試：完整流程
  - 回歸測試：確保原有功能不破壞
  - 性能測試：執行時間和內存使用
  - 邊緣測試：各種異常情況
- **目標**: 測試覆蓋率 > 85%
- **驗證**: pytest 全部通過
- **Risks**: 測試編寫耗時
- **是否完成**: X

#### Step 89: 批量驗證與性能測試
- **任務**: 在所有 RCAEval 數據集上驗證
- **檔案**: `experiments/validate_enhanced_pcmci_shapley.py`
- **依賴**: Steps 86-88
- **內容**:
  1. 運行所有數據集
  2. 記錄關鍵指標：
     - PCMCI 成功率（各策略分佈）
     - Fallback 觸發率
     - 聯合篩選器 Recall
     - Avg@K 準確度指標
     - 執行時間
  3. 與原版本對比
  4. 生成詳細報告
- **目標**:
  - PCMCI 失敗率 < 20%
  - Fallback 觸發率 < 30%
  - 整體 Avg@5 準確度 > 0.85
  - 平均執行時間 < 2 分鐘
- **驗證**: 達到目標指標
- **Risks**: 某些數據集可能仍有問題
- **是否完成**: X

#### Step 90: 撰寫完整文檔 ✅
- **任務**: 更新所有文檔
- **檔案**: `docs/PCMCI_SHAPLEY_TECHNICAL_DOCUMENTATION.md`, `docs/PCMCI_SHAPLEY_USER_GUIDE.md`, `docs/CONFIGURATION_GUIDE.md`, `docs/API_REFERENCE.md`
- **依賴**: Step 89
- **內容**:
  - 技術文檔：系統架構、核心組件、技術特性 ✅
  - 用戶指南：快速開始、數據格式、配置選項、結果解讀 ✅
  - 配置指南：參數分類、預設模板、調優指南、故障排除 ✅
  - API 參考：核心 API、配置 API、預處理 API、工具函數 ✅
- **目標**: 文檔完整清晰
- **驗證**: 新用戶能根據文檔使用
- **Risks**: 文檔與代碼不同步
- **是否完成**: O

---

### 子階段 K6: 高級優化與調優 (Steps 91-95)

#### Step 91: 實作超參數自動調優
- **任務**: 基於網格搜索或貝葉斯優化自動調參
- **檔案**: `pcmci_shapley_modules/auto_tuning.py` (新文件)
- **依賴**: Step 89
- **內容**:
  - 定義參數搜索空間
  - 實現網格搜索
  - 實現貝葉斯優化（可選）
  - 交叉驗證
  - 輸出最佳參數組合
- **目標**: 自動找到最佳參數
- **驗證**: 在驗證集上性能提升
- **Risks**: 計算成本高
- **是否完成**: X

#### Step 92: 實作數據集特徵提取器
- **任務**: 自動分析數據集特徵並推薦參數
- **檔案**: `pcmci_shapley_modules/dataset_profiler.py` (新文件)
- **依賴**: Step 91
- **內容**:
  ```python
  def profile_dataset(data: pd.DataFrame) -> dict:
      """
      分析數據集特徵
      
      Returns:
          profile: {
              'num_nodes': int,
              'time_series_length': int,
              'sampling_rate': float,
              'stationarity': float,
              'correlation_density': float,
              'anomaly_density': float,
              'system_type': str  # 'microservice', 'monolith', etc.
          }
      """
      # 實現特徵提取邏輯
      pass
  
  def recommend_config(profile: dict) -> PCMCIShapleyConfig:
      """
      基於數據集特徵推薦配置
      """
      config = PCMCIShapleyConfig()
      
      # 根據節點數調整
      if profile['num_nodes'] > 30:
          config.joint_top_n = 20
          config.u_max = 25
      
      # 根據採樣率調整
      if profile['sampling_rate'] < 10:  # 高頻
          config.tau_max = 7
      elif profile['sampling_rate'] > 60:  # 低頻
          config.tau_max = 3
      
      # 根據系統類型調整
      if profile['system_type'] == 'microservice':
          config.theta1 = 0.6  # 更依賴 trace
      
      return config
  ```
- **目標**: 自動推薦適合的參數
- **驗證**: 推薦配置優於預設
- **Risks**: 特徵提取可能不準確
- **是否完成**: X

#### Step 93: 實作在線學習與適應
- **任務**: 允許方法從歷史結果中學習
- **檔案**: `pcmci_shapley_modules/online_learning.py` (新文件)
- **依賴**: Step 92
- **內容**:
  - 記錄每次運行的結果
  - 記錄哪些參數組合效果好
  - 更新參數推薦模型
  - 持久化學習結果
- **目標**: 方法隨使用越來越智能
- **驗證**: 長期使用後性能提升
- **Risks**: 需要標註數據（根因標籤）
- **是否完成**: X

#### Step 94: 實作診斷和可視化工具
- **任務**: 提供豐富的診斷和可視化
- **檔案**: `pcmci_shapley_modules/diagnostics.py` (新文件)
- **依賴**: Step 86
- **內容**:
  - 生成 HTML 診斷報告
  - 可視化因果圖
  - 可視化 Shapley 值分佈
  - 可視化篩選過程
  - 可視化傳播過程
  - 對比 PCMCI 和 Fallback 結果
- **目標**: 幫助理解方法行為
- **驗證**: 生成可讀的報告
- **Risks**: 可視化工具複雜
- **是否完成**: X

#### Step 95: 最終驗收與文檔發布
- **任務**: 完成所有驗收標準
- **檔案**: 所有文檔
- **依賴**: Steps 86-94
- **內容**:
  1. 檢查所有 Steps 完成情況
  2. 運行完整測試套件
  3. 在所有數據集上驗證
  4. 生成最終性能報告
  5. 更新所有文檔
  6. 準備演示和教程
- **目標**: 達到所有驗收標準
- **驗收標準**:
  - 所有 Steps 66-95 完成
  - 測試覆蓋率 > 85%
  - PCMCI 成功率 > 80%
  - Fallback 觸發率 < 30%
  - 整體 Avg@5 > 0.85
  - 平均執行時間 < 2 分鐘
  - 文檔完整
- **Risks**: 時間壓力
- **是否完成**: X

---

## 實作優先順序（階段 K）

### 第一輪（核心穩定性）- 優先級最高
- Steps 66-70: PCMCI 穩定性增強
- Steps 71-75: SPOT 極值理論集成
- Steps 76-80: 聯合篩選器實現

### 第二輪（智能 Fallback）- 優先級高
- Steps 81-85: Fallback 機制優化
- Steps 86-90: 端到端集成與測試

### 第三輪（高級功能）- 優先級中
- Steps 91-93: 自動調優與在線學習
- Steps 94-95: 診斷工具與最終驗收

---

## 預期成果

完成階段 K 後，PCMCI-Shapley 方法將具備：

1. **高穩定性**：PCMCI 失敗率 < 20%
2. **高準確性**：Avg@5 > 0.85
3. **高魯棒性**：在各種數據質量下都能工作
4. **高智能度**：自動調參和推薦
5. **高可解釋性**：豐富的診斷和可視化 