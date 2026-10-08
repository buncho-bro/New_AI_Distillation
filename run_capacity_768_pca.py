"""
run_capacity_768_pca.py
=======================

Tests the capacity limits of the Concept Core in 768D (GPT-2 intrinsic dimension).
Applies PCA with whitening to solve the Anisotropy (Black Hole) problem,
spreading concepts isotropically across the 768D unit sphere.
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

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 768

def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_sentence_pool() -> list[str]:
    # Need at least > 768 sentences to perform PCA up to 768 dimensions!
    adjs = ["red", "blue", "large", "small", "happy", "sad", "fast", "slow", "bright", "dark", 
            "heavy", "light", "hot", "cold", "old", "new", "beautiful", "ugly", "clean", "dirty", 
            "loud", "quiet", "soft", "hard", "wild", "calm", "rough", "smooth", "sharp", "dull"]
    nouns = ["car", "dog", "house", "tree", "river", "mountain", "computer", "book", "phone", "bird", 
             "fish", "city", "village", "street", "cloud", "sun", "moon", "star", "flower", "music", 
             "art", "science", "game", "food", "idea", "dream", "memory", "future", "world", "person"]
    
    sentences = []
    for a, n in itertools.product(adjs, nouns):
        sentences.append(f"A {a} {n}.")
            
    print(f"Generated {len(sentences)} unique sentences (requires > 768 for full PCA).")
    
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

def experiment_capacity_768_pca() -> None:
    _set_seed()
    print("="*65)
    print("STEP 5: Concept Core Capacity Limit (768D + PCA Whitening)")
    print("="*65)
    
    print("Generating diverse sentences and computing 768D embeddings via GPT-2...")
    sentences = generate_sentence_pool()
    
    from transformers import GPT2Tokenizer, GPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2Model.from_pretrained("gpt2")
    
    # Batch encode
    emb_list = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            print(f"  Encoding batch {i//batch_size + 1} / {len(sentences)//batch_size + 1} ...")
            batch_sentences = sentences[i:i+batch_size]
            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            # Average pooling over sequence length
            mask = inputs.attention_mask.unsqueeze(-1).float()
            embeddings = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb_list.append(embeddings)
            
    emb_all_768 = torch.cat(emb_list, dim=0).float().numpy()
    
    print("\nApplying PCA with Whitening to equalize 768D space...")
    # whiten=True removes the anisotropy by scaling variance along all principal components to 1
    pca = PCA(n_components=DIM, whiten=True)
    emb_all_768_pca = torch.tensor(pca.fit_transform(emb_all_768)).float()
    
    # Normalize all to unit sphere
    emb_all_768_pca = emb_all_768_pca / emb_all_768_pca.norm(dim=1, keepdim=True)
    
    N_list = [10, 50, 100, 200, 300, 500, 700]
    
    recall_rates = []
    unique_ratios = []
    avg_steps_list = []
    
    # Base Physics parameters (Reverted to fixed kernel width for fair comparison)
    sigma = 0.3
    sigma_rep = 0.45
    alpha_rep = 0.4
    
    broken_N = None
    broken_core = None
    broken_obs = None
    
    for N in N_list:
        print(f"\nTesting N = {N} ...")
        obs = emb_all_768_pca[:N]
        
        # Initialize Concept Core
        p = HDParams(
            dim=DIM, use_ridge=True, sigma=sigma, sigma_rep=sigma_rep, alpha_rep=alpha_rep,
            marble_lr=0.5, marble_momentum=0.8, marble_max_v=0.1, marble_steps=400
        )
        core = HighDimConceptCore(p)
        
        # Directly register the N concepts
        core._centers = obs.clone()
        core._weights = torch.ones(N) * 0.2  
        
        # Test Recall
        noise = torch.randn(N, DIM) * 0.1
        start_pts = obs + noise
        start_pts = start_pts / start_pts.norm(dim=1, keepdim=True)
        
        final_positions, n_steps = roll_marbles_batched(core, start_pts)
        total_steps = n_steps.sum().item()
        
        # Check convergence
        dists = torch.norm(final_positions - obs, dim=1)
        successes = (dists < 0.1).sum().item()

        recall_rate = (successes / N) * 100.0
        
        # Count unique minimums
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

    # Plot Capacity Limit Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_list, recall_rates, marker='o', lw=2, label="Recall Rate (%)")
    ax.plot(N_list, unique_ratios, marker='s', lw=2, linestyle='--', label="Unique Minimums Ratio (%)")
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Number of Concepts (N)")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(f"Capacity Limit ({DIM}D, GPT-2 + PCA Whitening)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_curve = os.path.join(OUTPUT_DIR, "capacity_768_pca_curve.png")
    fig.savefig(out_curve, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved 768D PCA capacity curve to {out_curve}")

if __name__ == "__main__":
    experiment_capacity_768_pca()
