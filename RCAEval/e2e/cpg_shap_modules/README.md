# CPG-Shap Modules

This package implements the Local PCMCI-lag Causal + Shapley Propagation method.

- preprocessing.py: robust normalization, anomaly scoring, aggregation
- node_isolation.py: lagged correlation, isolation forest, trace augmentation
- cpg_shap_local.py: PCMCI+ wrapper and edge extraction
- edge_fusion.py: fuse trace/PCMCI/isolation and normalize
- propagation.py: K-step anomaly propagation
- shapley.py: coalition value and sampling-based Shapley
- scoring.py: reachability, temporal penalty, final ranking
- config.py: hyperparameters and validation

## Quick example

```python
from RCAEval.e2e.cpg_shap import cpg_shap
from RCAEval.e2e.cpg_shap_modules import CPGShapConfig
import pandas as pd

# df must include a `time` column and metric columns
result = cpg_shap(
    data=df,
    inject_time=inject_time,
    dataset="online-boutique",
    config=CPGShapConfig()
)
print(result["ranks"][0:10])
```
