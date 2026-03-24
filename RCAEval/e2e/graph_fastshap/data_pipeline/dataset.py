import os
import glob
import torch
import numpy as np
import pandas as pd
from RCAEval.io.time_series import preprocess
from .build_hetero_graph import build_hetero_graph

def get_sli(data_path, data, service):
    sli = None
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        elif "frontend_1" in data:
            sli = "frontend_1"
    else:
        sli = "unknown"
    return sli

def load_graph_dataset(dataset_name="online-boutique", root_dir=".", length=20, force_rebuild=False):
    """
    Loads or builds the entire dataset as a list of HeteroData objects.
    Computes static deviations and stores them inside the HeteroData.
    """
    cache_dir = os.path.join(root_dir, "cache", "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_dataset.pt")
    
    if not force_rebuild and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        return torch.load(cache_path)
    
    print(f"Building dataset graphs from scratch for {dataset_name}...")
    
    # Mirror the DATASET_MAP from main.py so logical names resolve correctly
    DATASET_MAP = {
        "online-boutique": "data/online-boutique",
        "sock-shop-1": "data/sock-shop-1",
        "sock-shop-2": "data/sock-shop-2",
        "train-ticket": "data/train-ticket",
        "re1-ob": "data/online-boutique",
        "re1-ss": "data/sock-shop-2",
        "re1-tt": "data/train-ticket",
        "re2-ob": "data/RE2/RE2-OB",
        "re2-ss": "data/RE2/RE2-SS",
        "re2-tt": "data/RE2/RE2-TT",
        "re3-ob": "data/RE3/RE3-OB",
        "re3-ss": "data/RE3/RE3-SS",
        "re3-tt": "data/RE3/RE3-TT",
    }
    
    if dataset_name in DATASET_MAP:
        dataset_path = os.path.join(root_dir, DATASET_MAP[dataset_name])
    else:
        dataset_path = os.path.join(root_dir, "data", dataset_name)
    data_paths = sorted(list(glob.glob(os.path.join(dataset_path, "**/data.csv"), recursive=True)))
    if not data_paths:
        data_paths = sorted(list(glob.glob(os.path.join(dataset_path, "**/simple_metrics.csv"), recursive=True)))
    
    data_list = []
    
    for i, data_path in enumerate(data_paths):
        data_dir = os.path.dirname(data_path)
        service = os.path.basename(os.path.dirname(os.path.dirname(data_path))).split("_")[0]
        case_name = os.path.basename(os.path.dirname(data_path))
        
        data = pd.read_csv(data_path)
        data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
        data = data.replace([float("inf"), float("-inf")], float("nan"))
        data = data.fillna(method="ffill").fillna(0)
        
        sli = get_sli(data_path, data, service)
        
        inject_time_path = os.path.join(data_dir, "inject_time.txt")
        if os.path.exists(inject_time_path):
            with open(inject_time_path) as f:
                inject_time = int(f.readlines()[0].strip())
        else:
            mid = len(data) // 2
            inject_time = data.iloc[mid]["time"] if "time" in data.columns else 0

        normal_df = data[data["time"] < inject_time].tail(length * 60 // 2)
        anomal_df = data[data["time"] >= inject_time].head(length * 60 // 2)
        
        if len(normal_df) == 0 or len(anomal_df) == 0:
            continue
            
        normal_df = preprocess(data=normal_df, dataset=dataset_name, dk_select_useful=False)
        anomal_df = preprocess(data=anomal_df, dataset=dataset_name, dk_select_useful=False)
        
        intersects = [c for c in normal_df.columns if c in anomal_df.columns]
        metric_cols = [c for c in intersects if c != "time"]
        if len(metric_cols) == 0:
            continue
            
        normal_df = normal_df[intersects]
        anomal_df = anomal_df[intersects]
        
        # Build causal graph
        hetero_data = build_hetero_graph(normal_df, anomal_df, metric_cols, dataset=dataset_name, sli=sli)
        
        # Pre-calculate deviations for fast soft-label during training
        normal_vals = normal_df[metric_cols].to_numpy(dtype=np.float64)
        normal_mu = np.mean(normal_vals, axis=0)
        normal_sigma = np.std(normal_vals, axis=0)
        
        dynamic_eps = float(np.median(normal_sigma))
        if dynamic_eps < 1e-5: 
            dynamic_eps = 1e-5
            
        anomal_mean = anomal_df[metric_cols].mean().to_numpy(dtype=np.float64)
        deviations = np.abs((anomal_mean - normal_mu) / (normal_sigma + dynamic_eps))
        
        # Store in HeteroData
        hetero_data.deviations = torch.tensor(deviations, dtype=torch.float32)
        hetero_data.case_id = f"{service}_{case_name}"
        hetero_data.metric_cols = metric_cols # Used for inference ranking mapping
        
        data_list.append(hetero_data)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(data_paths)} cases")
            
    print(f"Saving {len(data_list)} graphs to {cache_path}")
    torch.save(data_list, cache_path)
    return data_list
