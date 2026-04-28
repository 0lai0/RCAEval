import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from RCAEval.e2e.graph_fastshap.data_pipeline.dataset import load_graph_dataset
from RCAEval.e2e.graph_fastshap.models.surrogate_gnn import SurrogateGNN
from RCAEval.e2e.graph_fastshap.models.explainer_gnn import ExplainerGNN
from RCAEval.e2e.graph_fastshap.trainers.train_explainer import _compute_prior

def _compute_soft_label_fast(deviations, s):
    masked_devs = deviations * s
    full_dev_sum = deviations.sum() + 1e-8
    return float((masked_devs.sum() / full_dev_sum).item())

def fastshap_loss_ablation(
    phi_hat, s, s_bar, v_0, v_1, v_s, v_s_bar, prior=None,
    gamma=0.01, lambda_=1.0, alpha=0.5
):
    """Modified loss that returns components for logging."""
    scale = ((v_1 - v_0).abs() + 1e-3).detach()

    pred_s = v_0 + torch.dot(s, phi_hat)
    pred_s_bar = v_0 + torch.dot(s_bar, phi_hat)
    l_wls = ((v_s - pred_s) / scale) ** 2 + ((v_s_bar - pred_s_bar) / scale) ** 2

    l_eff = ((v_1 - v_0 - phi_hat.sum()) / scale) ** 2

    n_met = len(phi_hat)
    BASELINE_N = 50.0

    if prior is not None:
        weight = 1.0 - prior
        l_asym = (weight * phi_hat ** 2).sum() / n_met * BASELINE_N
        
        _, indices = torch.sort(prior, descending=True)
        n_top = max(1, min(5, len(prior) // 2))
        top_idx = indices[:n_top]
        bottom_idx = indices[n_top:]
        if len(bottom_idx) > 0:
            phi_top = phi_hat[top_idx].unsqueeze(1)
            phi_bottom = phi_hat[bottom_idx].unsqueeze(0)
            margin = 0.1
            l_rank = F.relu(margin - (phi_top - phi_bottom)).mean()
        else:
            l_rank = torch.tensor(0.0, device=phi_hat.device)
    else:
        l_asym = torch.tensor(0.0, device=phi_hat.device)
        l_rank = torch.tensor(0.0, device=phi_hat.device)

    total_loss = l_wls + gamma * l_eff + lambda_ * l_asym + alpha * l_rank
    
    components = {
        "wls": l_wls.item(),
        "eff": (gamma * l_eff).item() if isinstance(l_eff, torch.Tensor) else 0.0,
        "causal": (lambda_ * l_asym).item() if isinstance(l_asym, torch.Tensor) else 0.0,
        "rank": (alpha * l_rank).item() if isinstance(l_rank, torch.Tensor) else 0.0
    }
    
    return total_loss, components

def pretrain_ablation(args):
    dataset_name = args.dataset
    variant = args.variant
    n_samples = 16
    device = "cpu"
    print(f"Using device: {device} | Variant: {variant}")
    
    data_list = load_graph_dataset(dataset_name=dataset_name)
    if not data_list:
        print("Dataset is empty. Exiting.")
        return
        
    feat_dim = data_list[0]["metric"].x.shape[1]
    
    # -------------------------------------------------------------
    # PHASE 1: Surrogate (Same as original, but we can load if exists)
    # -------------------------------------------------------------
    surrogate = SurrogateGNN(in_dim=feat_dim, hidden_dim=64, n_layers=3).to(device)
    surr_path = f"checkpoints/{dataset_name}_surrogate.pt"
    
    if os.path.exists(surr_path) and not args.force_retrain_surrogate:
        print(f"Loading existing surrogate from {surr_path} to save time...")
        surrogate.load_state_dict(torch.load(surr_path, map_location=device))
    else:
        print("=== Phase 1: Pre-training Surrogate ===")
        surr_opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3, weight_decay=1e-5)
        surr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            surr_opt, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        best_surr_loss = float("inf")
        best_surr_state = None
        
        for epoch in range(args.surr_epochs):
            epoch_loss = 0.0
            surrogate.train()
            
            for data in data_list:
                data = data.to(device)
                n_met = data["metric"].x.shape[0]
                label_full = torch.tensor([1.0], device=device)
                label_empty = torch.tensor([0.0], device=device)
                
                deviations = data.deviations.to(device)
                
                for _ in range(n_samples):
                    s = torch.bernoulli(0.5 * torch.ones(n_met, device=device))
                    v_s = surrogate(data, s=s)
                    v_full = surrogate(data, s=torch.ones(n_met, device=device))
                    v_empty = surrogate(data, s=torch.zeros(n_met, device=device))
                    
                    loss_pred = F.binary_cross_entropy(v_full.unsqueeze(0), label_full) + \
                                F.binary_cross_entropy(v_empty.unsqueeze(0), label_empty)
                    
                    sl_val = _compute_soft_label_fast(deviations, s)
                    label_s = torch.tensor([sl_val], device=device)
                    loss_pred += F.binary_cross_entropy(v_s.unsqueeze(0), label_s)
                    
                    loss_mono = torch.tensor(0.0, device=device)
                    nonzero_indices = s.nonzero(as_tuple=True)[0]
                    if len(nonzero_indices) > 0:
                        idx = nonzero_indices[torch.randint(len(nonzero_indices), (1,))]
                        s_prime = s.clone()
                        s_prime[idx] = 0.0
                        v_s_prime = surrogate(data, s=s_prime)
                        loss_mono = F.relu(v_s_prime - v_s + 1e-3)
                    
                    loss = loss_pred + 0.1 * loss_mono
                    
                    surr_opt.zero_grad()
                    loss.backward()
                    clip_grad_norm_(surrogate.parameters(), max_norm=5.0)
                    surr_opt.step()
                    epoch_loss += loss.item()
                    
            avg_loss = epoch_loss / len(data_list)
            surr_scheduler.step(avg_loss)
            if avg_loss < best_surr_loss:
                best_surr_loss = avg_loss
                best_surr_state = {k: v.clone() for k, v in surrogate.state_dict().items()}
        
        if best_surr_state is not None:
            surrogate.load_state_dict(best_surr_state)
            
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(surrogate.state_dict(), surr_path)

    for param in surrogate.parameters():
        param.requires_grad = False
    surrogate.eval()
    
    # -------------------------------------------------------------
    # PHASE 2: Explainer (Ablation)
    # -------------------------------------------------------------
    explainer = ExplainerGNN(in_dim=feat_dim, hidden_dim=64, n_layers=3).to(device)
    expl_opt = torch.optim.Adam(explainer.parameters(), lr=1e-3, weight_decay=1e-5)
    expl_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        expl_opt, T_0=20, T_mult=2, eta_min=1e-6
    )
    best_expl_loss = float("inf")
    best_expl_state = None
    
    print("=== Phase 2: Pre-training Explainer (Ablation mode) ===")
    for epoch in range(args.expl_epochs):
        epoch_loss = 0.0
        
        # Aggregated component logging
        comp_sums = {"wls": 0.0, "eff": 0.0, "causal": 0.0, "rank": 0.0}
        total_steps = 0
        
        explainer.train()
        
        warmup_fraction = min(1.0, epoch / max(1, args.expl_epochs * 0.5))
        dynamic_alpha = 0.0 if args.disable_rank else (0.5 * warmup_fraction)
        
        for data in data_list:
            data = data.to(device)
            n_met = data["metric"].x.shape[0]
            prior = _compute_prior(data).to(device)
            
            for _ in range(n_samples):
                phi_hat = explainer(data)
                
                s = torch.bernoulli(0.5 * torch.ones(n_met, device=device))
                s_bar = 1.0 - s
                
                with torch.no_grad():
                    v_0 = surrogate(data, s=torch.zeros(n_met, device=device))
                    v_1 = surrogate(data, s=torch.ones(n_met, device=device))
                    v_s = surrogate(data, s=s)
                    v_s_bar = surrogate(data, s=s_bar)
                    
                loss, comps = fastshap_loss_ablation(
                    phi_hat, s, s_bar, v_0, v_1, v_s, v_s_bar,
                    prior=prior, 
                    gamma=args.gamma_eff, 
                    lambda_=args.lambda_asym, 
                    alpha=dynamic_alpha
                )
                
                expl_opt.zero_grad()
                loss.backward()
                clip_grad_norm_(explainer.parameters(), max_norm=5.0)
                expl_opt.step()
                
                epoch_loss += loss.item()
                for k in comp_sums:
                    comp_sums[k] += comps[k]
                total_steps += 1
                
        avg_loss = epoch_loss / len(data_list)
        expl_scheduler.step()
        
        # Logging with components
        if total_steps > 0:
            avg_comps = {k: v / total_steps for k, v in comp_sums.items()}
            comp_str = f"[WLS: {avg_comps['wls']:.4f} | Eff: {avg_comps['eff']:.4f} | Causal: {avg_comps['causal']:.4f} | Rank: {avg_comps['rank']:.4f}]"
        else:
            comp_str = ""
            
        if avg_loss < best_expl_loss:
            best_expl_loss = avg_loss
            best_expl_state = {k: v.clone() for k, v in explainer.state_dict().items()}
            
        print(f"Explainer Epoch {epoch+1}/{args.expl_epochs} | Total Loss: {avg_loss:.4f} {comp_str}")
        
    if best_expl_state is not None:
        explainer.load_state_dict(best_expl_state)
        print(f"Restored best explainer (Loss: {best_expl_loss:.4f})")

    # Save Checkpoint to Ablation Dir
    ckpt_dir = os.path.join("checkpoints", f"ablation_{variant}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    expl_path = os.path.join(ckpt_dir, f"{dataset_name}_explainer.pt")
    surr_path_link = os.path.join(ckpt_dir, f"{dataset_name}_surrogate.pt")
    
    torch.save(explainer.state_dict(), expl_path)
    # Also save a copy of surrogate so inference script finds both easily
    torch.save(surrogate.state_dict(), surr_path_link)
    
    print(f"Pre-training complete! Ablation models saved for variant '{variant}' in {ckpt_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="online-boutique")
    parser.add_argument("--surr-epochs", type=int, default=100)
    parser.add_argument("--expl-epochs", type=int, default=150)
    parser.add_argument("--force-retrain-surrogate", action="store_true", help="Do not load existing surrogate")
    
    # Ablation Flags
    parser.add_argument("--variant", type=str, required=True, help="Name of ablation variant (e.g. no-causal)")
    parser.add_argument("--lambda-asym", type=float, default=1.0, help="Weight for causal penalty")
    parser.add_argument("--gamma-eff", type=float, default=0.01, help="Weight for efficiency penalty")
    parser.add_argument("--disable-rank", action="store_true", help="Disable contrastive ranking loss (alpha=0)")
    
    args = parser.parse_args()
    pretrain_ablation(args)
