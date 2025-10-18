# PCMCI-Shapley 穩健性增強計劃

**建立日期**: 2025-10-18  
**狀態**: 規劃階段  
**優先級**: 高

---

## 執行摘要

在實作 PCMCI-Shapley 方法的過程中，我們發現了關鍵的穩健性問題，導致：
1. **PCMCI 失敗率高**：約 100% 的情況下遇到數據標準化錯誤
2. **Fallback 機制過度觸發**：所有資料集都使用相同的 anomaly 排序
3. **準確度完全相同**：不同資料集的準確度指標完全一致（0.24 或 0.48）
4. **性能問題**：大資料集需要 2+ 小時執行時間

本計劃補充了 15 個新步驟（Steps 51-65）和 3 個可選優化步驟（Steps 66-68），預計需要 15-19 小時完成。

---

## 問題診斷詳情

### 問題 1: PCMCI 標準化失敗

**錯誤信息**:
```
ValueError: nans after standardizing, possibly constant array!
```

**根本原因**:
- 數據中存在常數列（標準差為 0）
- PCMCI 的 ParCorr 檢定無法處理常數或低變異性數據
- 數據清理不夠徹底

**影響**:
- PCMCI 完全無法執行
- 導致圖結構為空（0 條邊）
- 觸發 fallback 機制

### 問題 2: Fallback 機制設計缺陷

**當前行為**:
```python
if total_weight == 0.0:
    # 使用 anomaly_scores 的最後一行排序
    metric_ranks = anomaly_scores.iloc[-1].sort_values(ascending=False).index.tolist()
```

**問題**:
- 所有資料集都使用相同的排序邏輯
- 沒有考慮資料集的差異性
- 評分機制完全失效（所有節點評分為 0）

**結果**:
- `re2-tt`: Avg@5 = 0.0（所有指標）
- `re2-ob`: Avg@5 = 0.24（所有指標）
- `re2-ss`: Avg@5 = 0.48（所有指標）

### 問題 3: 評分機制在空圖下失效

**觀察到的評分**:
```
scores = {'adservice': 1.0, 'main': 0.0, 'redis': 0.0, ...}
shapley = {'adservice': 1.0, 'main': 0.0, 'redis': 0.0, ...}
reachability = {'adservice': 1.0, 'main': 0.0, 'redis': 0.0, ...}
anomaly = {'adservice': 1.0, 'main': 0.0, 'redis': 0.0, ...}
```

**問題**:
- 除了 focus_node，所有節點的所有評分都是 0
- 三個評分維度完全相同，失去區分度
- 無法有效排序和識別根因

### 問題 4: 性能瓶頸

**實測數據**:
- `re2-tt` (31 節點): 2 小時 4 分鐘 / 案例
- PCMCI 階段: 約 2 小時
- PCMCI 產生邊數: 75 萬條（異常多）

**原因**:
- 節點數量過多（31 個）
- tau_max 設置過大（5）
- PCMCI alpha 過於寬鬆（0.05）
- 產生大量假陽性邊

---

## 解決方案架構

### 層次 1: 數據品質保證（Steps 51-53）

**目標**: 確保 PCMCI 輸入數據的質量

1. **數據品質檢查器** (Step 51)
   - 檢測常數列、NaN、低變異性
   - 提供詳細診斷報告
   - 自動清理和修復

2. **多策略執行器** (Step 52)
   - 標準 PCMCI+ (alpha=0.05, tau=5)
   - 寬鬆 PCMCI (alpha=0.1, tau=3)
   - 極寬鬆 PCMCI (alpha=0.2, tau=2)
   - 簡單相關性
   - Granger 因果
   - 空圖降級

3. **數據預處理增強** (Step 53)
   - Robust 標準化（MAD）
   - 異常值裁剪
   - 差分處理
   - 平滑處理

### 層次 2: 降級與替代方案（Steps 54-57）

**目標**: 當 PCMCI 失敗時提供有效的替代方案

4. **替代因果推斷** (Step 54)
   - Granger Causality
   - Transfer Entropy
   - CCM
   - 滯後相關

5. **邊融合降級策略** (Step 55)
   - 僅 Trace (若可用)
   - Trace + Isolation
   - Isolation + Anomaly
   - K-NN Graph
   - 完全圖降級

6. **評分機制健壯化** (Step 56)
   - 空圖模式：降低 Shapley 權重
   - 稀疏圖模式：動態調整權重
   - 正常模式：標準權重

7. **統計鄰域傳播** (Step 57)
   - 基於滯後相關性的虛擬傳播
   - 替代因果圖傳播

### 層次 3: 性能與參數優化（Steps 58-60）

**目標**: 提升執行效率和參數自適應性

8. **PCMCI 參數自適應** (Step 58)
   - 自動調整 tau_max
   - 自動調整 alpha
   - 限制 max_conds_dim

9. **節點數量動態調整** (Step 59)
   - 根據 PCMCI 成功率調整
   - 迭代縮減策略

10. **時間性能優化** (Step 60)
    - PCMCI 並行化
    - Shapley 抽樣優化
    - 快取機制

### 層次 4: 調優與驗證（Steps 61-65）

**目標**: 自動化參數調優和全面驗證

11. **超參數網格搜索** (Step 61)
12. **參數推薦系統** (Step 62)
13. **診斷與可視化** (Step 63)
14. **自動化測試套件** (Step 64)
15. **批量資料集驗證** (Step 65)

---

## 實作優先順序

### 第一優先（立即實作）

**目標**: 讓 PCMCI 能夠成功執行

- [ ] Step 51: 數據品質檢查器
- [ ] Step 52: 多策略執行器
- [ ] Step 53: 數據預處理增強

**預期效果**:
- PCMCI 成功率從 0% 提升到 60-80%
- Fallback 觸發率降低到 20-40%

### 第二優先（重要）

**目標**: 改善空圖情況下的表現

- [ ] Step 55: 邊融合降級策略
- [ ] Step 56: 評分機制健壯化
- [ ] Step 57: 統計鄰域傳播

**預期效果**:
- 即使在空圖情況下也能提供有意義的排序
- 不同資料集的準確度有差異

### 第三優先（性能）

**目標**: 提升執行速度

- [ ] Step 58: PCMCI 參數自適應
- [ ] Step 59: 節點數量動態調整
- [ ] Step 60: 時間性能優化

**預期效果**:
- 執行時間從 2 小時降低到 10-15 分鐘
- 大資料集可以在合理時間內完成

### 第四優先（調優）

**目標**: 精細調優和驗證

- [ ] Step 61-65: 超參數調優和批量驗證

**預期效果**:
- 找到最佳參數組合
- 全面驗證穩健性

---

## 成功指標

### 定量指標

1. **PCMCI 成功率**: ≥ 80%
2. **Fallback 觸發率**: < 20%
3. **準確度變異係數**: > 0.3
4. **平均執行時間**: 
   - online-boutique: < 5 分鐘
   - re2-tt: < 15 分鐘
5. **穩健性測試通過率**: > 95%

### 定性指標

1. 不同資料集的準確度不再完全相同
2. 日誌能清楚顯示執行路徑和降級策略
3. 錯誤信息更加友好和可操作
4. 參數調優有明確指導

---

## 風險與挑戰

### 高風險

1. **過度降級影響準確性**
   - 緩解：保留策略等級記錄，分析不同策略的效果
   
2. **參數調優計算量大**
   - 緩解：使用分層搜索，先粗調後細調

3. **替代方法的因果解釋性弱**
   - 緩解：明確標記使用的方法，提供置信度指標

### 中風險

4. **快取機制可能引入bug**
   - 緩解：嚴格的快取失效策略和測試

5. **自適應參數可能不最優**
   - 緩解：保留手動覆蓋選項

### 低風險

6. **新增代碼增加維護成本**
   - 緩解：良好的文檔和模組化設計

---

## 時間線

### Week 1: 核心穩健性（Steps 51-57）
- Day 1-2: Steps 51-53 (數據處理)
- Day 3-4: Steps 54-57 (降級策略)
- Day 5: 整合測試

### Week 2: 性能與調優（Steps 58-65）
- Day 1-2: Steps 58-60 (性能優化)
- Day 3-4: Steps 61-63 (調優工具)
- Day 5: Steps 64-65 (測試與驗證)

### Week 3: 可選優化（Steps 66-68）
- 根據實際需求決定

---

## 依賴與前提條件

### 已完成
- ✅ Steps 1-50: 基礎實作
- ✅ 問題診斷和根因分析
- ✅ 詳細日誌和診斷工具

### 需要
- Python 3.10+
- tigramite, numpy, pandas, scikit-learn, networkx
- RCAEval 測試資料集
- 足夠的計算資源（至少 8GB RAM）

---

## 下一步行動

1. **立即**: 開始實作 Step 51（數據品質檢查器）
2. **本週內**: 完成 Steps 51-53
3. **下週**: 完成 Steps 54-60
4. **兩週內**: 完成全部並驗證

---

## 附錄 A: 診斷日誌範例

### 當前日誌（有問題）
```
[2025-10-18 11:59:01] INFO Starting pcmci_shapley pipeline
[2025-10-18 11:59:01] INFO Focus node: productcatalogservice
[2025-10-18 11:59:01] INFO Local set size: |U|=14
[2025-10-18 11:59:02] WARNING PCMCI failed: nans after standardizing
[2025-10-18 11:59:02] WARNING FALLBACK TRIGGERED!
[2025-10-18 11:59:02] INFO Metric-level Top-5: [...]
```

### 改進後日誌（目標）
```
[2025-10-18 11:59:01] INFO Starting pcmci_shapley pipeline
[2025-10-18 11:59:01] INFO Focus node: productcatalogservice
[2025-10-18 11:59:01] INFO Local set size: |U|=14
[2025-10-18 11:59:01] INFO Data quality check: removed 2 constant columns
[2025-10-18 11:59:01] INFO PCMCI Strategy 1 (standard) failed: constant array
[2025-10-18 11:59:02] INFO PCMCI Strategy 2 (relaxed) SUCCESS: 15 edges found
[2025-10-18 11:59:02] INFO Edge fusion: using PCMCI + Isolation (no trace)
[2025-10-18 11:59:02] INFO Graph density: 0.15 (sparse mode)
[2025-10-18 11:59:02] INFO Scoring weights: shapley=0.3, reach=0.4, anomaly=0.3
[2025-10-18 11:59:02] INFO Metric-level Top-5: [...]
```

---

## 附錄 B: 參數推薦規則

```python
def get_default_params(dataset_type: str, has_trace: bool, num_nodes: int, 
                       data_length: int) -> dict:
    """
    根據資料集特徵推薦參數
    """
    params = {
        'tau_max': 5,
        'pcmci_alpha': 0.05,
        'theta1': 0.6 if has_trace else 0.0,
        'theta2': 0.3 if has_trace else 0.8,
        'theta3': 0.1 if has_trace else 0.2,
        'K': 8,
        'alpha_prop': 0.8,
    }
    
    # 調整 tau_max
    if data_length < 500:
        params['tau_max'] = 3
    if num_nodes > 25:
        params['tau_max'] = min(params['tau_max'], 3)
    
    # 調整 pcmci_alpha
    if num_nodes > 20:
        params['pcmci_alpha'] = 0.1  # 更寬鬆
    
    # 調整傳播步數
    if dataset_type == 'train-ticket':
        params['K'] = 10
    elif dataset_type in ['online-boutique', 'sock-shop']:
        params['K'] = 8
    
    return params
```

---

**文件版本**: v1.0  
**最後更新**: 2025-10-18  
**負責人**: AI Assistant  
**審核狀態**: 待確認

