"""
DEPRECATED: This module is deprecated and will be removed in a future version.
Please use pc_shapley instead.

This is a backward compatibility wrapper that redirects to pc_shapley.
"""
import warnings
from typing import Any, Dict
import pandas as pd
import networkx as nx

from .pc_shapley import pc_shapley, PCShapleyConfig


def pcmci_shapley(
    data: pd.DataFrame,
    inject_time: int | None = None,
    dataset: str | None = None,
    dk_select_useful: bool = False,
    focus_node: str | None = None,
    trace_graph: nx.DiGraph | None = None,
    config: Any | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    DEPRECATED: Use pc_shapley instead.
    
    This function is a backward compatibility wrapper that calls pc_shapley.
    It will be removed in version 2.0.0.
    
    Migration guide:
    - Replace: from RCAEval.e2e.pcmci_shapley import pcmci_shapley
    - With: from RCAEval.e2e.pc_shapley import pc_shapley
    - Replace: PCMCIShapleyConfig -> PCShapleyConfig
    - Update config parameters: pcmci_alpha -> pc_alpha, pcmci_max_conds_dim -> pc_max_conds_dim
    """
    warnings.warn(
        "pcmci_shapley is deprecated and will be removed in version 2.0.0. "
        "Please use pc_shapley instead. "
        "See migration guide in docstring.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Convert old config to new config if needed
    if config is not None and hasattr(config, 'pcmci_alpha'):
        # Old config detected, convert it
        new_config = PCShapleyConfig()
        # Copy over compatible parameters
        for attr in dir(config):
            if not attr.startswith('_'):
                if attr == 'pcmci_alpha':
                    new_config.pc_alpha = getattr(config, attr)
                elif attr == 'pcmci_max_conds_dim':
                    new_config.pc_max_conds_dim = getattr(config, attr)
                elif hasattr(new_config, attr):
                    setattr(new_config, attr, getattr(config, attr))
        config = new_config
    
    return pc_shapley(
        data=data,
        inject_time=inject_time,
        dataset=dataset,
        dk_select_useful=dk_select_useful,
        focus_node=focus_node,
        trace_graph=trace_graph,
        config=config,
        **kwargs
    )


# Backward compatibility alias
PCMCIShapleyConfig = PCShapleyConfig
