import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from RCAEval.io.time_series import preprocess
from .graph_fastshap.data_pipeline.build_hetero_graph import build_hetero_graph
from .graph_fastshap.models.surrogate_gnn import SurrogateGNN
from .graph_fastshap.models.explainer_gnn import ExplainerGNN
from .graph_fastshap.trainers.train_surrogate import train_surrogate
from .graph_fastshap.trainers.train_explainer import train_explainer
from .graph_fastshap.inference import phi_to_ranks

def graph_fastshap_ablation(data, inject_time=None, dataset=None, sli=None, variant="no-causal", **kwargs):
    if variant == "baseline-random":
        # Random Baseline logic
        intersects = [c for c in data.columns if c != "time" and c in data.columns]
        metric_cols = intersects
        import random
        scores = [random.random() for _ in metric_cols]
        ranks = phi_to_ranks(scores, metric_cols)
        return {"ranks": ranks}

    # ==================================================================
    # 1. Preprocessing + split
    # ==================================================================
    if inject_time is not None and "time" in data.columns:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        mid = len(data) // 2
        normal_df = data.iloc[:mid]
        anomal_df = data.iloc[mid:]

    normal_df = preprocess(data=normal_df, dataset=dataset, dk_select_useful=False)
    anomal_df = preprocess(data=anomal_df, dataset=dataset, dk_select_useful=False)

    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]
    metric_cols = [c for c in intersects if c != "time"]

    if len(metric_cols) == 0:
        return {"ranks": []}

    # ==================================================================
    # 2. Build HeteroData graph (M1)
    # ==================================================================
    hetero_data = build_hetero_graph(
        normal_df, anomal_df, metric_cols,
        dataset=dataset, sli=sli,
    )

    # ==================================================================
    # 3. Hyper-parameters
    # ==================================================================
    device = "cpu"
    feat_dim = hetero_data["metric"].x.shape[1]
    hidden_dim = int(os.environ.get("GFS_HIDDEN_DIM", 64))
    n_layers = int(os.environ.get("GFS_N_LAYERS", 3))
    
    # Ablation overrides
    lambda_ = 1.0
    gamma = 0.01
    mu = 0.1
    
    if variant == "no-causal":
        lambda_ = 0.0
    elif variant == "no-eff":
        gamma = 0.0
        
    # Validation constraint
    if variant not in ["no-causal", "no-eff", "no-rank", "no-adapt"]:
        print(f"Warning: Unknown ablation variant '{variant}', using 'no-causal' defaults")
        variant = "no-causal"

    # ==================================================================
    # 4. Load Ablation Weights
    # ==================================================================
    ckpt_dir = f"checkpoints/ablation_{variant}"
    # Exception: no-adapt uses full baseline model or a specifically trained one?
    # Our pretrain_ablation saves it in checkpoints/ablation_no-adapt if trained,
    # or we can point it to the base checkpoints if they train the same way.
    # The run_ablations.sh will run pretrain_ablation --variant no-adapt 
    # so we just load from ablation_no-adapt dir.
    
    surr_path = f"{ckpt_dir}/{dataset}_surrogate.pt"
    expl_path = f"{ckpt_dir}/{dataset}_explainer.pt"

    if not os.path.exists(expl_path):
        raise FileNotFoundError(f"Ablation weights not found: {expl_path}")

    surrogate = SurrogateGNN(in_dim=feat_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    surrogate.load_state_dict(torch.load(surr_path, map_location=device))
    
    explainer = ExplainerGNN(in_dim=feat_dim, hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    explainer.load_state_dict(torch.load(expl_path, map_location=device))
    
    # ==================================================================
    # 5. Local Adaptation (Ablated if variant == "no-adapt")
    # ==================================================================
    ft_surr_epochs = int(os.environ.get("GFS_FT_SURR_EPOCHS", 5))
    ft_expl_epochs = int(os.environ.get("GFS_FT_EXPL_EPOCHS", 5))
    n_samples = int(os.environ.get("GFS_N_SAMPLES", 16))
    
    if variant == "no-adapt":
        ft_surr_epochs = 0
        ft_expl_epochs = 0
    
    if ft_surr_epochs > 0:
        surrogate = train_surrogate(
            surrogate, hetero_data,
            normal_df=normal_df, anomal_df=anomal_df,
            metric_cols=metric_cols, sli=sli,
            n_epochs=ft_surr_epochs, n_samples=n_samples,
            lr=1e-3, mu=mu, device=device,
        )
        
    if ft_expl_epochs > 0:
        explainer = train_explainer(
            explainer, surrogate, hetero_data,
            n_epochs=ft_expl_epochs, n_samples=n_samples,
            lr=1e-3, gamma=gamma, lambda_=lambda_, device=device,
        )

    # ==================================================================
    # 6. Inference + Ensemble
    # ==================================================================
    explainer.eval()
    with torch.no_grad():
        phi_hat = explainer(hetero_data.to(device)).cpu()

    # Z-score deviation blending
    normal_vals = normal_df[metric_cols].to_numpy(dtype=np.float64)
    normal_mu = np.mean(normal_vals, axis=0)
    normal_sigma = np.std(normal_vals, axis=0) + 1e-8
    anomal_mean = anomal_df[metric_cols].mean().to_numpy(dtype=np.float64)
    dev_scores = np.abs((anomal_mean - normal_mu) / normal_sigma)

    phi_numpy = phi_hat.numpy()
    phi_norm = (phi_numpy - phi_numpy.min()) / (phi_numpy.max() - phi_numpy.min() + 1e-8)
    dev_norm = (dev_scores - dev_scores.min()) / (dev_scores.max() - dev_scores.min() + 1e-8)

    ensemble_phi = 0.7 * phi_norm + 0.3 * dev_norm
    ranks = phi_to_ranks(ensemble_phi, metric_cols)

    return {"ranks": ranks}
