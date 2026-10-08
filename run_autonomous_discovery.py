"""
run_autonomous_discovery.py
===========================

Verification of Step 2: Autonomous Concept Discovery
(Unsupervised Clustering & Autonomous Flow Generation).

Scattered observational data (raindrops) is dropped onto the landscape.
They carve the potential field as they roll, clustering into deep 
valleys autonomously without any prior anchors or paths.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 64
N_CLUSTERS = 3
POINTS_PER_CLUSTER = 40
NOISE_STD = 1.2

def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_data() -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate scattered data around hidden ground-truth clusters."""
    # 3 distant cluster centers
    gt_centers = torch.randn(N_CLUSTERS, DIM) * 5.0
    
    observations = []
    labels = []
    for i in range(N_CLUSTERS):
        # Generate noisy points around the center.
        # In 64D, we must constrain the L2 norm of the noise, otherwise points 
        # will be too far apart and won't feel each other's gravity.
        noise = torch.randn(POINTS_PER_CLUSTER, DIM)
        noise = noise / noise.norm(dim=1, keepdim=True) * NOISE_STD
        pts = gt_centers[i] + noise
        observations.append(pts)
        labels.append(torch.full((POINTS_PER_CLUSTER,), i))
        
    obs_tensor = torch.cat(observations, dim=0)
    lbl_tensor = torch.cat(labels, dim=0)
    
    # Shuffle
    idx = torch.randperm(obs_tensor.shape[0])
    return obs_tensor[idx], lbl_tensor[idx], gt_centers


def experiment_autonomous_discovery() -> None:
    _set_seed()
    print("\n" + "="*65)
    print("STEP 2: Autonomous Concept Discovery (Unsupervised Clustering)")
    print("="*65)
    
    obs, labels, gt_centers = generate_data()
    print(f"  Generated {obs.shape[0]} unlabelled observations in {DIM}-D space.")
    print(f"  Hidden ground-truth clusters: {N_CLUSTERS}")
    
    # Initialize Core (using Mexican Hat ridge for stability)
    p = HDParams(
        dim=DIM, use_ridge=True, sigma=1.2, sigma_rep=1.8, alpha_rep=0.4,
        decay_rate=0.99, prune_thresh=0.01,
        marble_lr=0.8, marble_momentum=0.9, marble_max_v=0.5
    )
    core = HighDimConceptCore(p)
    
    print("\n  Simulating rain & erosion (autonomous flow accumulation) ...")
    # This might take a few seconds
    final_drops = core.rain_and_erode(obs, steps=80, depth_gain=0.1)
    
    # Clean up small noise centers via final weathering
    for _ in range(50):
        core.weather()
        
    stats = core.weight_stats()
    print(f"\n  Memory bank after discovery: {stats['n']} attractors formed.")
    print(f"  (Ideally close to {N_CLUSTERS} macro-valleys)")
    
    # Verify clustering: check if drops from same GT cluster converged together
    # We can measure the average distance of final_drops to their respective GT center
    # vs their initial distance.
    initial_dists = []
    final_dists = []
    for i in range(N_CLUSTERS):
        mask = (labels == i)
        start_pts = obs[mask]
        end_pts = final_drops[mask]
        gt_c = gt_centers[i]
        
        initial_dists.append(torch.cdist(start_pts, gt_c.unsqueeze(0)).mean().item())
        # The drops might not converge EXACTLY to gt_c (since it's unsupervised),
        # but they should tightly cluster together.
        # Let's measure their spread (std dev).
        cluster_center = end_pts.mean(dim=0)
        spread = torch.cdist(end_pts, cluster_center.unsqueeze(0)).mean().item()
        final_dists.append(spread)
        
    avg_init = sum(initial_dists)/len(initial_dists)
    avg_fin = sum(final_dists)/len(final_dists)
    
    print(f"\n  Average initial spread (noise): {avg_init:.2f}")
    print(f"  Average final spread (convergence): {avg_fin:.2f}")
    
    if avg_fin < avg_init * 0.3:
        print("  Clustering Check: PASS (Drops successfully condensed into distinct valleys!)")
    else:
        print("  Clustering Check: FAIL (Drops did not condense tightly)")
        
    # --- Visualization (PCA) ---
    print("\n  Generating PCA visualization of autonomous discovery ...")
    
    # Fit PCA on GT centers + initial points to get a good projection
    pca = PCA(n_components=2)
    fit_data = torch.cat([gt_centers, obs]).numpy()
    pca.fit(fit_data)
    
    obs_2d = pca.transform(obs.numpy())
    fin_2d = pca.transform(final_drops.numpy())
    cen_2d = pca.transform(core.centers.detach().numpy())
    gt_2d = pca.transform(gt_centers.numpy())
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Background potential
    pad = 2.0
    x_min, x_max = obs_2d[:,0].min()-pad, obs_2d[:,0].max()+pad
    y_min, y_max = obs_2d[:,1].min()-pad, obs_2d[:,1].max()+pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_64d = pca.inverse_transform(grid_2d)
    
    with torch.no_grad():
        H_grid = core.potential(torch.from_numpy(grid_64d).float()).numpy()
    H_grid = H_grid.reshape(xx.shape)
    
    cf = ax.contourf(xx, yy, H_grid, levels=30, cmap="terrain", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Potential H(x)")
    
    # Plot initial scattered data
    scatter = ax.scatter(obs_2d[:,0], obs_2d[:,1], c=labels.numpy(), cmap="Set1", 
                         alpha=0.3, s=20, label="Initial Raindrops")
    
    # Plot final converged data
    ax.scatter(fin_2d[:,0], fin_2d[:,1], c=labels.numpy(), cmap="Set1", 
               edgecolors='k', s=60, marker='*', label="Final Positions (after rolling)")
               
    # Plot learned attractors
    w_norm = core.weights.detach().numpy()
    w_sizes = 30 + 100 * (w_norm / w_norm.max())
    ax.scatter(cen_2d[:,0], cen_2d[:,1], facecolors='none', edgecolors='k', 
               s=w_sizes, linewidths=2, label="Learned Valley Centers")
               
    ax.set_title("Autonomous Concept Discovery (64D -> 2D PCA)", fontsize=14)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    # Custom legend
    handles, labels_leg = ax.get_legend_handles_labels()
    # Remove duplicate labels from scatter colormap
    unique = dict(zip(labels_leg, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right")
    
    out_path = os.path.join(OUTPUT_DIR, "autonomous_discovery.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


if __name__ == "__main__":
    experiment_autonomous_discovery()
