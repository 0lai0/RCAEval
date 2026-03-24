import os
import argparse
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from RCAEval.e2e.graph_fastshap.data_pipeline.dataset import load_graph_dataset
from RCAEval.e2e.graph_fastshap.models.surrogate_gnn import SurrogateGNN
from RCAEval.e2e.graph_fastshap.models.explainer_gnn import ExplainerGNN
from RCAEval.e2e.graph_fastshap.trainers.train_explainer import _compute_prior
from RCAEval.e2e.graph_fastshap.trainers.custom_loss import fastshap_loss

def _compute_soft_label_fast(deviations, s):
    # Vectorized fast soft label directly entirely in PyTorch
    masked_devs = deviations * s
    full_dev_sum = deviations.sum() + 1e-8
    return float((masked_devs.sum() / full_dev_sum).item())

def pretrain(dataset_name="online-boutique", surrogate_epochs=100, explainer_epochs=150, n_samples=16):
    # Force CPU for pre-training because small ~20-node graphs bottleneck hard on CUDA kernel launches
    device = "cpu"
    print(f"Using device: {device}")
    
    data_list = load_graph_dataset(dataset_name=dataset_name)
    if not data_list:
        print("Dataset is empty. Exiting.")
        return
        
    feat_dim = data_list[0]["metric"].x.shape[1]
    
    surrogate = SurrogateGNN(in_dim=feat_dim, hidden_dim=64, n_layers=3).to(device)
    surr_opt = torch.optim.Adam(surrogate.parameters(), lr=1e-3, weight_decay=1e-5)
    surr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        surr_opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    best_surr_loss = float("inf")
    best_surr_state = None
    
    print("=== Phase 1: Pre-training Surrogate ===")
    for epoch in range(surrogate_epochs):
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
                
                loss_pred = (
                    F.binary_cross_entropy(v_full.unsqueeze(0), label_full) +
                    F.binary_cross_entropy(v_empty.unsqueeze(0), label_empty)
                )
                
                sl_val = _compute_soft_label_fast(deviations, s)
                label_s = torch.tensor([sl_val], device=device)
                loss_pred += F.binary_cross_entropy(v_s.unsqueeze(0), label_s)
                
                # Monotonicity hinge
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
        current_lr = surr_opt.param_groups[0]['lr']
        if avg_loss < best_surr_loss:
            best_surr_loss = avg_loss
            best_surr_state = {k: v.clone() for k, v in surrogate.state_dict().items()}
        print(f"Surrogate Epoch {epoch+1}/{surrogate_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Best: {best_surr_loss:.4f}")
        
    # Restore best surrogate weights
    if best_surr_state is not None:
        surrogate.load_state_dict(best_surr_state)
        print(f"Restored best surrogate (Loss: {best_surr_loss:.4f})")
    for param in surrogate.parameters():
        param.requires_grad = False
    surrogate.eval()
    
    explainer = ExplainerGNN(in_dim=feat_dim, hidden_dim=64, n_layers=3).to(device)
    expl_opt = torch.optim.Adam(explainer.parameters(), lr=1e-3, weight_decay=1e-5)
    expl_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        expl_opt, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    best_expl_loss = float("inf")
    best_expl_state = None
    
    print("=== Phase 2: Pre-training Explainer ===")
    for epoch in range(explainer_epochs):
        epoch_loss = 0.0
        explainer.train()
        
        warmup_fraction = min(1.0, epoch / max(1, explainer_epochs * 0.5))
        dynamic_alpha = 0.5 * warmup_fraction
        
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
                    
                loss = fastshap_loss(
                    phi_hat, s, s_bar, v_0, v_1, v_s, v_s_bar,
                    prior=prior, gamma=0.01, lambda_=1.0, alpha=dynamic_alpha
                )
                
                expl_opt.zero_grad()
                loss.backward()
                clip_grad_norm_(explainer.parameters(), max_norm=5.0)
                expl_opt.step()
                epoch_loss += loss.item()
                
        avg_loss = epoch_loss / len(data_list)
        expl_scheduler.step(avg_loss)
        current_lr = expl_opt.param_groups[0]['lr']
        if avg_loss < best_expl_loss:
            best_expl_loss = avg_loss
            best_expl_state = {k: v.clone() for k, v in explainer.state_dict().items()}
        print(f"Explainer Epoch {epoch+1}/{explainer_epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Best: {best_expl_loss:.4f}")
        
    # Restore best explainer weights
    if best_expl_state is not None:
        explainer.load_state_dict(best_expl_state)
        print(f"Restored best explainer (Loss: {best_expl_loss:.4f})")

    os.makedirs("checkpoints", exist_ok=True)
    surr_path = f"checkpoints/{dataset_name}_surrogate.pt"
    expl_path = f"checkpoints/{dataset_name}_explainer.pt"
    torch.save(surrogate.state_dict(), surr_path)
    torch.save(explainer.state_dict(), expl_path)
    print(f"Pre-training complete! Models saved to {surr_path} and {expl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="online-boutique")
    parser.add_argument("--surr-epochs", type=int, default=100)
    parser.add_argument("--expl-epochs", type=int, default=150)
    args = parser.parse_args()
    pretrain(dataset_name=args.dataset, surrogate_epochs=args.surr_epochs, explainer_epochs=args.expl_epochs)
