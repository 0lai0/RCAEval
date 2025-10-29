# Causal Discovery Module - User Guide

## Overview

The `causal_discovery` module provides a unified, simplified interface for building causal graphs using multiple algorithms. This replaces the previous complex `pcmci_local` implementation with a cleaner, more flexible approach.

## Supported Methods

1. **PCMCI** (default) - Time-series causal discovery with time lags
2. **PC** - Constraint-based causal discovery (instantaneous relationships)
3. **GES** - Score-based causal discovery (planned)
4. **FCI** - Causal discovery with latent confounders (planned)
5. **LiNGAM** - Linear non-Gaussian acyclic model (planned)

## Quick Start

### Basic Usage

```python
from RCAEval.e2e.pcmci_shapley_modules import causal_discovery, PCMCIShapleyConfig

# Create configuration
config = PCMCIShapleyConfig(
    causal_method="pcmci",  # or "pc"
    pcmci_alpha=0.05,
    tau_max=5
)

# Discover causal graph
result = causal_discovery.discover_causal_graph(
    data=your_dataframe,
    local_nodes=["service1", "service2", "service3"],
    config=config,
    method="pcmci"  # or "pc"
)

# Access results
edges = result["edges"]  # List of (cause, effect, lag) tuples
strengths = result["edge_strengths"]  # Dict of edge strengths
graph = result["pcmci_graph"]  # NetworkX DiGraph
columns = result["columns"]  # Column names used
```

### Using PC Algorithm

```python
config = PCMCIShapleyConfig(
    causal_method="pc",
    pcmci_alpha=0.05
)

result = causal_discovery.discover_causal_graph(
    data=df,
    local_nodes=nodes,
    config=config,
    method="pc"
)
```

### Integration with PCMCI-Shapley Pipeline

The main pipeline automatically uses the configured method:

```python
from RCAEval.e2e.pcmci_shapley import pcmci_shapley
from RCAEval.e2e.pcmci_shapley_modules import PCMCIShapleyConfig

config = PCMCIShapleyConfig(
    causal_method="pc",  # Switch to PC algorithm
    pcmci_alpha=0.05,
    # ... other parameters
)

result = pcmci_shapley(
    data=data,
    focus_node="frontend",
    trace_graph=trace_graph,
    config=config
)
```

## Architecture

### Module Structure

```
causal_discovery.py
├── prepare_data_matrix()      # Data preparation
├── clean_data_matrix()         # Preprocessing & cleaning
├── run_pcmci()                 # PCMCI+ algorithm
├── run_pc()                    # PC algorithm
├── extract_edges_and_strengths() # Unified result extraction
└── discover_causal_graph()    # Main entry point
```

### Key Functions

#### `prepare_data_matrix(data, local_nodes, use_pca, pca_components)`
- Extracts relevant columns for selected nodes
- Handles missing values intelligently
- Optional PCA dimensionality reduction
- Returns: `(data_matrix, column_names)`

#### `clean_data_matrix(X, min_std)`
- Removes constant variables
- Adds minimal noise to near-constant variables
- Returns: `(cleaned_matrix, variable_mask)`

#### `run_pcmci(X, tau_max, alpha, max_conds_dim)`
- Runs PCMCI+ algorithm for time-series causal discovery
- Considers time lags up to `tau_max`
- Returns: `{p_matrix, val_matrix, graph}`

#### `run_pc(X, alpha, max_conds_dim)`
- Runs PC algorithm for instantaneous causal relationships
- Uses conditional independence tests
- Returns: `{graph, p_matrix, val_matrix}`

#### `discover_causal_graph(data, local_nodes, config, method)`
- **Main entry point** - unified interface for all methods
- Automatically handles data preparation, algorithm selection, and result extraction
- Returns standardized result format

## Comparison: PCMCI vs PC

| Feature | PCMCI | PC |
|---------|-------|-----|
| **Time lags** | Yes (captures temporal dynamics) | No (instantaneous only) |
| **Speed** | Slower (more complex) | Faster (simpler) |
| **Data requirements** | Longer time series needed | Works with shorter series |
| **Use case** | When temporal dynamics matter | When only contemporaneous relationships matter |
| **Directionality** | Strong (time-based) | Weaker (constraint-based) |

## Configuration Options

### Key Parameters

```python
PCMCIShapleyConfig(
    # Method selection
    causal_method="pcmci",  # "pcmci" or "pc"
    
    # Significance testing
    pcmci_alpha=0.05,       # p-value threshold
    
    # PCMCI-specific
    tau_max=5,              # Maximum time lag (PCMCI only)
    
    # Optimization
    pcmci_max_conds_dim=3,  # Max conditioning set size
    use_pca=False,          # PCA dimensionality reduction
    pca_components=10,      # Number of PCA components
)
```

## Advantages of New Design

### 1. Simplicity
- **Before**: Complex multi-step process scattered across multiple functions
- **After**: Single entry point `discover_causal_graph()`

### 2. Flexibility
- **Before**: Hardcoded to PCMCI only
- **After**: Easy to switch between algorithms via `causal_method` parameter

### 3. Maintainability
- **Before**: Algorithm-specific logic mixed with data handling
- **After**: Clean separation of concerns (data prep, algorithm, result extraction)

### 4. Extensibility
- **Before**: Adding new algorithms requires modifying existing code
- **After**: New algorithms can be added as separate functions following the template

### 5. Robustness
- **Before**: Multiple failure points with inconsistent error handling
- **After**: Unified error handling with consistent fallback behavior

## Migration Guide

### Old Approach
```python
from .pcmci_shapley_modules import pcmci_local as pcmci_mod

pcmci_res = pcmci_mod.local_pcmci_causal_test(
    service_df, 
    list(service_df.columns), 
    cfg
)
```

### New Approach
```python
from .pcmci_shapley_modules import causal_discovery as causal_mod

pcmci_res = causal_mod.discover_causal_graph(
    service_df,
    list(service_df.columns),
    cfg,
    method=cfg.causal_method  # "pcmci" or "pc"
)
```

The result format remains the same, ensuring backward compatibility.

## Performance Tips

1. **For fast iteration**: Use `causal_method="pc"` during development
2. **For production**: Use `causal_method="pcmci"` for better temporal modeling
3. **For large graphs**: Set `pcmci_max_conds_dim=2` or `3` to limit complexity
4. **For noisy data**: Increase `pcmci_alpha` to 0.1 for fewer spurious edges
5. **For high-dimensional data**: Enable `use_pca=True`

## Common Issues & Solutions

### Issue: "Not enough variables after cleaning"
**Solution**: Lower the pruning thresholds or increase `pruning_min_nodes`

### Issue: "Time series too short"
**Solution**: Either collect more data or use `causal_method="pc"` (no time lags)

### Issue: "Empty causal graph"
**Solution**: 
- Increase `pcmci_alpha` (e.g., 0.1)
- Check if variables have sufficient variation
- Verify data quality and preprocessing

### Issue: "PCMCI standardization error"
**Solution**: The module now handles this automatically by:
- Removing constant variables
- Adding minimal noise to near-constant variables
- Aligning time series lengths

## Future Extensions

Planned implementations:
- **GES**: Score-based approach using BIC/AIC
- **FCI**: Handle latent confounders
- **LiNGAM**: For linear non-Gaussian data
- **DAG-GNN**: Neural network-based causal discovery
- **DYNOTEARS**: Continuous optimization for DAG learning

## References

- PCMCI: Runge et al. (2019) "Detecting and quantifying causal associations in large nonlinear time series datasets"
- PC: Spirtes et al. (2000) "Causation, Prediction, and Search"
- Tigramite: https://github.com/jakobrunge/tigramite

