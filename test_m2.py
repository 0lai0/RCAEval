import os
import torch
import pandas as pd
from RCAEval.io.time_series import preprocess
from RCAEval.e2e.graph_fastshap.data_pipeline.build_hetero_graph import build_hetero_graph
from RCAEval.e2e.graph_fastshap.models.surrogate_gnn import SurrogateGNN
from RCAEval.e2e.graph_fastshap.trainers.train_surrogate import train_surrogate

def test_m2():
    # 1. Load an online-boutique case
    dataset = "online-boutique"
    # Choose a specific case file we know has a clear root cause, or just the first one
    # The normal data paths are accessible via glob in main.py, let's use the same logic
    import glob
    data_paths = sorted(list(glob.glob(f"data/{dataset}/**/data.csv", recursive=True)))
    if not data_paths:
        data_paths = sorted(list(glob.glob(f"data/{dataset}/**/simple_metrics.csv", recursive=True)))
    
    if not data_paths:
        print("No data paths found!")
        return
    
    # Let's find a case where fault is 'cpu' for clear testing
    target_path = next((p for p in data_paths if "cpu_" in p), data_paths[0])
    print(f"Testing on case: {target_path}")
    
    # 2. Extract service and fault from path. e.g. data/online-boutique/adservice_cpu/1/data.csv
    # The fault folder is -3 from the end
    folder_name = target_path.split("/")[-3]
    parts = folder_name.split("_")
    fault_service = parts[0]
    fault_type = parts[1] if len(parts) > 1 else "unknown"
    
    # 3. Preprocess similar to main.py
    data = pd.read_csv(target_path)
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    data = data.replace([float("inf"), float("-inf")], float("nan")).fillna(method="ffill").fillna(0)
    
    with open(os.path.join(os.path.dirname(target_path), "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip())
        
    normal_df = data[data["time"] < inject_time].tail(20 * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(20 * 60 // 2)
    
    normal_df = preprocess(data=normal_df, dataset=dataset, dk_select_useful=False)
    anomal_df = preprocess(data=anomal_df, dataset=dataset, dk_select_useful=False)
    
    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    metric_cols = [c for c in intersects if c != "time"]
    
    sli = "frontend_latency"
    if f"{fault_service}_latency" in normal_df.columns:
        sli = f"{fault_service}_latency"
        
    print(f"SLI column identified as: {sli}")
    
    # 4. Build Graph
    print("Building causal graph...")
    hetero_data = build_hetero_graph(normal_df, anomal_df, metric_cols, dataset=dataset, sli=sli)
    feat_dim = hetero_data["metric"].x.shape[1]
    
    # 5. Initialize and Train Surrogate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Surrogate using {device}...")
    
    surrogate = SurrogateGNN(in_dim=feat_dim, hidden_dim=32, n_layers=2).to(device)
    
    # Train for 50 epochs just to ensure loss behaves correctly and model learns
    surrogate = train_surrogate(
        surrogate, hetero_data.to(device),
        normal_df=normal_df, anomal_df=anomal_df,
        metric_cols=metric_cols, sli=sli,
        n_epochs=100, n_samples=32, lr=1e-3, mu=0.1, device=device
    )
    print("Training finished.")
    
    # 6. Perturbation Test
    print("\n--- Perturbation Test ---")
    surrogate.eval()
    with torch.no_grad():
        n_met = hetero_data["metric"].x.shape[0]
        
        # Test 1: Full graph
        v_full = surrogate(hetero_data.to(device), s=torch.ones(n_met, device=device)).item()
        
        # Test 2: Mask the fault Root Cause service's CPU metric
        target_metric = f"{fault_service}_{fault_type}"
        v_masked = None
        if target_metric in metric_cols:
            idx = metric_cols.index(target_metric)
            s_masked = torch.ones(n_met, device=device)
            s_masked[idx] = 0.0 # MASK IT EXCLUSIVELY
            v_masked = surrogate(hetero_data.to(device), s=s_masked).item()
            masked_name = target_metric
        else:
            # Mask whatever metric had highest deviation
            devs = np.abs((anomal_df[metric_cols].mean() - normal_df[metric_cols].mean()) / (normal_df[metric_cols].std() + 1e-8)).values
            idx = np.argmax(devs)
            s_masked = torch.ones(n_met, device=device)
            s_masked[idx] = 0.0
            v_masked = surrogate(hetero_data.to(device), s=s_masked).item()
            masked_name = metric_cols[idx]
        
        # Test 3: Mask a random unaffected metric
        random_idx = (idx + 10) % n_met
        s_rand = torch.ones(n_met, device=device)
        s_rand[random_idx] = 0.0
        v_rand = surrogate(hetero_data.to(device), s=s_rand).item()
        
        print(f"Ground Truth/Max Deviation Root Cause: {masked_name}")
        print(f"Random Unaffected Node: {metric_cols[random_idx]}")
        print(f"1. v(All features present)        = {v_full:.4f}")
        print(f"2. v(Masking Random Unaffected)   = {v_rand:.4f}  (Should remain high)")
        print(f"3. v(Masking Root Cause ONLY)     = {v_masked:.4f}  (Should drop significantly)")
        
        if v_full - v_masked > 0.1:
            print("\n✅ SUCCESS: Surrogate successfully learned that the root cause drives the anomaly.")
        else:
            print("\n❌ FAILURE / WARNING: Drop was insignificant. Soft Label or Masking might not be propagating.")

if __name__ == "__main__":
    test_m2()
