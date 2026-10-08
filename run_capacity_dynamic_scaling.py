"""
run_capacity_dynamic_scaling.py
===============================

Tests the capacity limits of the 64D Concept Core (with DoG ridge) on the unit sphere,
but applies **Dynamic Kernel Scaling** based on the density of the concepts.
"""

from __future__ import annotations

import os
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 64

def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_sentence_pool() -> list[str]:
    adjs = ["red", "blue", "large", "small", "happy", "sad", "fast", "slow", "bright", "dark", 
            "heavy", "light", "hot", "cold", "old", "new", "beautiful", "ugly", "clean", "dirty", 
            "loud", "quiet", "soft", "hard", "wild", "calm", "rough", "smooth"]
    nouns = ["car", "dog", "house", "tree", "river", "mountain", "computer", "book", "phone", "bird", 
             "fish", "city", "village", "street", "cloud", "sun", "moon", "star", "flower", "music", 
             "art", "science", "game", "food", "idea", "dream", "memory", "future"]
    
    sentences = []
    for a, n in itertools.product(adjs, nouns):
        sentences.append(f"A {a} {n}.")
        if len(sentences) >= 600:
            break
            
    # Shuffle predictably
    np.random.seed(42)
    np.random.shuffle(sentences)
    return sentences

def roll_marbles_batched(core, start_pts):
    # start_pts: (N, DIM)
    x = start_pts.clone().detach().float()
    v = torch.zeros_like(x)
    n_steps = torch.ones(x.shape[0]) * core.p.marble_steps
    active = torch.ones(x.shape[0], dtype=torch.bool)
    
    for step_i in range(core.p.marble_steps):
        if not active.any():
            break
            
        x_var = x.clone().detach().requires_grad_(True)
        H = core.potential(x_var)
        H.sum().backward()
        grad = x_var.grad.detach()
        
        v = core.p.marble_momentum * v - core.p.marble_lr * grad
        
        # terminal velocity
        v_norm = v.norm(dim=-1, keepdim=True)
        mask = v_norm > core.p.marble_max_v
        v = torch.where(mask, v * (core.p.marble_max_v / v_norm.clamp(min=1e-9)), v)
        
        x = x + v
        x = x / x.norm(dim=-1, keepdim=True)
        
        # Check stopping condition
        stopped = (v.norm(dim=-1) < core.p.marble_vel_thresh) & active
        n_steps[stopped] = step_i + 1
        active = active & ~stopped
        
    return x.detach(), n_steps

def get_mean_nearest_neighbor_distance(pts: torch.Tensor) -> float:
    """Computes the mean distance to the nearest neighbor for a set of points."""
    N = pts.shape[0]
    if N < 2:
        return 1.0
    dist_matrix = torch.cdist(pts, pts)
    # Fill diagonal with infinity to ignore self-distance
    dist_matrix.fill_diagonal_(float('inf'))
    min_dists, _ = dist_matrix.min(dim=1)
    return min_dists.mean().item()

def experiment_capacity_dynamic() -> None:
    _set_seed()
    print("="*65)
    print(f"STEP 4: Concept Core Capacity Limit with DYNAMIC SCALING ({DIM}D)")
    print("="*65)
    
    print("Generating diverse sentences and computing embeddings...")
    sentences = generate_sentence_pool()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_all_384 = torch.tensor(model.encode(sentences))
    
    # Use PCA on the FULL 500-sized dataset to create a fixed 64D semantic space
    pca = PCA(n_components=DIM)
    emb_all_64 = torch.tensor(pca.fit_transform(emb_all_384.numpy())).float()
    # Normalize all to unit sphere
    emb_all_64 = emb_all_64 / emb_all_64.norm(dim=1, keepdim=True)
    
    N_list = [10, 25, 50, 100, 150, 200, 300, 500]
    
    recall_rates = []
    unique_ratios = []
    avg_steps_list = []
    
    # Base Physics parameters (will be dynamically scaled)
    # The ratio of sigma_rep to sigma is maintained at 1.5 (0.45 / 0.3)
    alpha_rep = 0.4
    
    broken_N = None
    broken_core = None
    broken_obs = None
    
    for N in N_list:
        print(f"\nTesting N = {N} ...")
        obs = emb_all_64[:N]
        
        # DYNAMIC SCALING LOGIC
        # 1. Compute how close concepts are packed
        d_nn = get_mean_nearest_neighbor_distance(obs)
        # 2. Scale sigma to be a fraction of the nearest neighbor distance
        # To avoid overlapping, sigma should be around d_nn / 4
        # For N=10, d_nn might be large, giving sigma close to 0.3
        sigma_dyn = max(0.05, d_nn * 0.25)
        sigma_rep_dyn = sigma_dyn * 1.5
        
        print(f"  -> Mean Nearest Neighbor Dist: {d_nn:.4f}")
        print(f"  -> Dynamically scaled sigma: {sigma_dyn:.4f}, sigma_rep: {sigma_rep_dyn:.4f}")
        
        # Initialize Concept Core
        p = HDParams(
            dim=DIM, use_ridge=True, sigma=sigma_dyn, sigma_rep=sigma_rep_dyn, alpha_rep=alpha_rep,
            marble_lr=0.5, marble_momentum=0.8, marble_max_v=0.1, marble_steps=400
        )
        core = HighDimConceptCore(p)
        
        # Directly register the N concepts (perfect valleys)
        core._centers = obs.clone()
        core._weights = torch.ones(N) * 0.2  # Depth of 0.2 per concept
        
        # Test Recall
        # Add noise to starting positions
        noise = torch.randn(N, DIM) * 0.1
        start_pts = obs + noise
        start_pts = start_pts / start_pts.norm(dim=1, keepdim=True)
        
        final_positions, n_steps = roll_marbles_batched(core, start_pts)
        total_steps = n_steps.sum().item()
        
        # Check convergence
        dists = torch.norm(final_positions - obs, dim=1)
        successes = (dists < 0.1).sum().item()

        recall_rate = (successes / N) * 100.0
        
        # Count unique minimums (valleys found)
        # Cluster final positions with threshold 0.05
        unique_mins = []
        for fp in final_positions:
            is_new = True
            for um in unique_mins:
                if torch.norm(fp - um).item() < 0.05:
                    is_new = False
                    break
            if is_new:
                unique_mins.append(fp)
                
        unique_ratio = (len(unique_mins) / N) * 100.0
        avg_steps = total_steps / N
        
        print(f"  -> Recall Rate: {recall_rate:.1f}%")
        print(f"  -> Unique Minimums: {len(unique_mins)} / {N} ({unique_ratio:.1f}%)")
        print(f"  -> Avg Steps: {avg_steps:.1f}")
        
        recall_rates.append(recall_rate)
        unique_ratios.append(unique_ratio)
        avg_steps_list.append(avg_steps)
        
        # Detect failure point for visualization (if it still breaks)
        if recall_rate < 80.0 and broken_N is None:
            broken_N = N
            broken_core = core
            broken_obs = obs
            print(f"  [!] Capacity breakdown detected at N={N}")

    # Plot Capacity Limit Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_list, recall_rates, marker='o', lw=2, label="Recall Rate (%)")
    ax.plot(N_list, unique_ratios, marker='s', lw=2, linestyle='--', label="Unique Minimums Ratio (%)")
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Number of Concepts (N)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Dynamic Capacity Limit ({DIM}D, DoG Kernel with Dynamic Sigma)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_curve = os.path.join(OUTPUT_DIR, "capacity_dynamic_curve.png")
    fig.savefig(out_curve, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved dynamic capacity curve to {out_curve}")

if __name__ == "__main__":
    experiment_capacity_dynamic()
