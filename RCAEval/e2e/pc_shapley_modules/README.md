# PC-Shapley Modules

This package implements the PC-Shapley method for root cause analysis.

- preprocessing.py: robust normalization, anomaly scoring, aggregation
- node_isolation.py: lagged correlation, isolation forest, trace augmentation
- causal_discovery.py: PC algorithm wrapper and edge extraction
- edge_fusion.py: fuse trace/PC/isolation and normalize
- propagation.py: K-step anomaly propagation
- shapley.py: coalition value and sampling-based Shapley
- scoring.py: reachability, temporal penalty, final ranking
- config.py: hyperparameters and validation

## Quick example

```python
from RCAEval.e2e.pc_shapley import pc_shapley
from RCAEval.e2e.pc_shapley_modules import PCShapleyConfig
import pandas as pd

# df must include a `time` column and metric columns
result = pc_shapley(
    data=df,
    inject_time=inject_time,
    dataset="online-boutique",
    config=PCShapleyConfig()
)
print(result["ranks"][0:10])
```
