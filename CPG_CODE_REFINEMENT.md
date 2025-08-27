# CPG程式碼
-----

## 主要修改點

### 1.修正 `CorrelationCausalModel` 的核心邏輯 ✅

**問題**:

  - 原本兩個分支都在做相關性分析，邏輯混亂。
  - `if`分支計算的是候選者之間的相關性，而不是候選者與症狀的相關性，違背了因果推斷的目標。

**修改**:

```python
class CorrelationCausalModel:
    """基於相關性的因果發現模型"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
    def infer_causal(self, candidates: List[AggregatedEvent], target_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        推斷因果關係。
        透過計算每個候選事件與目標症狀之間的特徵相關性來實現。
        """
        edges = []
        
        for candidate in candidates:
            if len(candidate.features) > 0 and len(target_features) > 0:
                # 確保特徵向量長度相同才能計算相關性
                if len(candidate.features) == len(target_features):
                    correlation = np.corrcoef(candidate.features, target_features)[0, 1]
                    if not np.isnan(correlation):
                        confidence = abs(correlation)
                        if confidence > 0.05:
                            edges.append({
                                'source': candidate,
                                'confidence': confidence,
                                'strength': confidence
                            })
        
        return edges
```

**效果**: 統一了邏輯，只保留了正確的相關性計算方法。

-----

### 2.彙總根因分數至服務層級 ✅

**問題**:

  - 排名結果包含視窗後綴（如`ts-order-service_w3`）。
  - 使用者需要的是服務層級的根因，不是時間視窗層級。

**修改**:

```python
def quantify_root_causes(self, vertices: set, edges: set) -> Dict[str, float]:
    # ... PageRank計算 ...
    
    # 將視窗層級的分數彙總到服務層級
    service_scores = {}
    for window_name, score in windowed_scores.items():
        # 移除 '_w' + 數字 的後綴
        base_service_name = '_'.join(window_name.split('_')[:-1]) if '_w' in window_name else window_name
        
        # 彙總分數，這裡我們取最大值，代表該服務最異常的時刻
        if base_service_name not in service_scores or score > service_scores[base_service_name]:
            service_scores[base_service_name] = score
    
    return service_scores
```

**效果**: 最終排名顯示服務名稱而不是視窗名稱，更符合業務需求。

-----

### 3.清理無用程式碼與註釋 ✅

**修改**:

  - **刪除`AutoEncoder`類別**: 完全未使用，刪除以保持程式碼整潔。
  - **修正註釋**: 更新`PageRankImportanceCalculator`的docstring。
  - **優化實例化**: 將`CorrelationCausalModel`的實例化移到迴圈外。

<!-- end list -->

```python
# 刪除未使用的AutoEncoder
# class AutoEncoder(nn.Module): ...

class PageRankImportanceCalculator:
    """使用 PageRank 演算法來計算圖中節點重要性的計算器"""
    
def build_local_cpg(self, symptoms, all_events):
    # 優化：在迴圈外實例化模型以提高效率
    feature_dim = len(all_events[0].features) if all_events else 0
    correlation_model = CorrelationCausalModel(input_dim=feature_dim)
    
    for symptom in symptoms:
        # 使用已建立的模型實例
        causal_edges = correlation_model.infer_causal(candidates, symptom.features)
```

**效果**: 程式碼更整潔，效率更高。

-----

### 4.【中優先級】統一最終排名的實體 ✅

**問題**:

  - `ranks`包含服務名稱，`remaining_cols`包含欄位名稱，混合不一致。

**修改**:

```python
# 取得資料中所有唯一的服務名稱
all_service_names = set()
for col in data.columns:
    if col != 'time':
        service_name = cpg_framework._extract_service_from_column(col)
        if service_name != 'unknown' and not service_name.startswith('node-'):
            all_service_names.add(service_name)

# 新增未在CPG排名中的其他服務
ranked_services = set(ranks)
remaining_services = [s for s in all_service_names if s not in ranked_services]
ranks.extend(remaining_services)
```

**效果**: 最終排名列表中的所有元素都為服務名稱，保持一致性。

-----

## 預期改進效果

### 邏輯準確性

  - **因果推斷邏輯**：現在能正確計算候選事件與症狀的相關性。
  - **分數聚合**：服務層級的分數更符合根因分析需求。

### 程式碼整潔度

  - **移除冗餘**：刪除未使用的AutoEncoder類別。
  - **統一命名**：更正註釋與類別名稱。
  - **優化效能**：減少不必要的物件建立。

### 一致性

  - **資料類型一致**：最終排名全部為服務名稱。
  - **邏輯一致**：單一的因果推斷方法。

### 業務價值

  - **更直觀的結果**：使用者看到的是`ts-admin-service`而不是`ts-admin-service_w3`。
  - **更準確的分析**：正確的相關性計算提高了分析品質。
  - **更好的效能**：優化的程式碼執行效率更高。

-----

## 相容性保證

所有修改都保持了對外介面的相容性，不會影響現有的呼叫方式，同時保持了對所有資料集（sock-shop、online-boutique、train-ticket）的支援。