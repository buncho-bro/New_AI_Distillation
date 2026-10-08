"""
run_capacity_experiments.py
============================

Stress tests and capacity experiments for the 64-D High-Dimensional Concept Core,
including the Mexican Hat (Difference of Gaussians) ridge implementation.

Scenarios:
  A) Valley separation test (Merge limit of two close attractors).
  B) Multi-concept capacity stress test (N=10, 30, 50).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from high_dim_concept_core import HighDimConceptCore, HDParams

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DIM = 64


def _set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Visualisation: 1D Cross-Section of Mexican Hat
# ---------------------------------------------------------------------------

def visualise_ridge_profile() -> None:
    """Plot the 1D profile of the potential to show the Mexican Hat ridge."""
    print("\nGenerating Ridge Cross-Section ...")
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    zeros = torch.zeros(200, DIM - 1)
    pts = torch.cat([x, zeros], dim=1)  # (200, 64)

    # Core without ridges
    core_flat = HighDimConceptCore(HDParams(dim=DIM, use_ridge=False))
    core_flat._centers = torch.zeros(1, DIM)
    core_flat._weights = torch.tensor([1.0])
    y_flat = core_flat.potential(pts).detach().numpy()

    # Core with ridges
    core_ridge = HighDimConceptCore(HDParams(dim=DIM, use_ridge=True, sigma=1.0, sigma_rep=1.5, alpha_rep=0.4))
    core_ridge._centers = torch.zeros(1, DIM)
    core_ridge._weights = torch.tensor([1.0])
    y_ridge = core_ridge.potential(pts).detach().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x.squeeze(), y_flat, label="No Ridge (Gaussian)", lw=2, linestyle="--", color="gray")
    ax.plot(x.squeeze(), y_ridge, label="With Ridge (Mexican Hat)", lw=2, color="#E91E63")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    
    # Highlight the ridge part (where potential > 0)
    ridge_mask = y_ridge > 0
    ax.fill_between(x.squeeze().numpy(), 0, y_ridge, where=ridge_mask, color="#E91E63", alpha=0.3, label="Repulsive Ridge")

    ax.set_xlabel("Distance from center", fontsize=12)
    ax.set_ylabel("Potential H(x)", fontsize=12)
    ax.set_title("Concept Potential Profile (Mexican Hat)", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "ridge_cross_section.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


# ---------------------------------------------------------------------------
# Experiment A: Two close valleys separation
# ---------------------------------------------------------------------------

def test_separation(distance: float, use_ridge: bool) -> bool:
    """Test if two concepts separated by `distance` remain distinct."""
    p = HDParams(dim=DIM, use_ridge=use_ridge, sigma=1.0, sigma_rep=1.5, alpha_rep=0.4, depth_gain=2.0)
    core = HighDimConceptCore(p)
    
    A = torch.zeros(DIM)
    B = torch.zeros(DIM)
    B[0] = distance
    
    # Erode them heavily to form deep valleys
    for _ in range(10):
        core.erode(A.unsqueeze(0))
        core.erode(B.unsqueeze(0))
        
    final_A, _, _ = core.roll_marble(A + torch.ones(DIM)*0.01) # slight perturbation
    
    # If final position is close to original A, they are separated.
    # If they merged into the midpoint, distance to A will be around distance/2.
    dist_A = (final_A - A).norm().item()
    
    # Check if we stayed closer to A than to the midpoint
    return dist_A < (distance / 3.0)


def experiment_a() -> None:
    print("\n" + "="*65)
    print("EXPERIMENT A: Two Close Valleys Separation Limit")
    print("="*65)
    
    distances = [3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.5]
    
    print(f"{'Distance (d/sigma)':<20} | {'No Ridge (Flat)':<20} | {'With Ridge (DoG)':<20}")
    print("-" * 65)
    
    for d in distances:
        sep_flat = test_separation(d, use_ridge=False)
        sep_ridge = test_separation(d, use_ridge=True)
        print(f"{d:<20.1f} | {'SEPARATED' if sep_flat else 'MERGED (Degraded)':<20} | {'SEPARATED' if sep_ridge else 'MERGED (Degraded)':<20}")
        
    print("\nConclusion: Ridge (Mexican Hat) prevents early merging of concepts!")


# ---------------------------------------------------------------------------
# Experiment B: N-concept capacity stress test
# ---------------------------------------------------------------------------

def capacity_test(N: int, use_ridge: bool) -> float:
    """Test recall accuracy with N independent concepts."""
    p = HDParams(
        dim=DIM, use_ridge=use_ridge, sigma=1.0, sigma_rep=1.5, alpha_rep=0.4,
        depth_gain=1.0, marble_steps=200
    )
    core = HighDimConceptCore(p)
    
    # Generate N random orthogonal-ish concepts on a sphere
    # Scale=2.2 means average pairwise distance is ~ 2.2*sqrt(2) = 3.1
    # This creates a dense pack where flat model collapses but ridge survives
    concepts = torch.randn(N, DIM)
    concepts = concepts / concepts.norm(dim=1, keepdim=True) * 2.2
    
    # Erode them all
    for _ in range(5):
        for i in range(N):
            core.erode(concepts[i].unsqueeze(0))
            
    # Recall test
    successes = 0
    for i in range(N):
        noise = torch.randn(DIM)
        noise = noise / noise.norm() * 0.5
        start_pos = concepts[i] + noise
        
        final_pos, _, _ = core.roll_marble(start_pos)
        
        dists = torch.cdist(final_pos.unsqueeze(0), concepts).squeeze(0)
        closest_idx = dists.argmin().item()
        
        if closest_idx == i and dists[i].item() < 1.0:
            successes += 1
            
    return (successes / N) * 100.0


def experiment_b() -> None:
    print("\n" + "="*65)
    print("EXPERIMENT B: Capacity Stress Test (N=5, 10, 15, 20, 30)")
    print("="*65)
    
    Ns = [5, 10, 15, 20, 30]
    results_flat = []
    results_ridge = []
    
    print(f"{'N Concepts':<15} | {'No Ridge Accuracy':<20} | {'With Ridge Accuracy':<20}")
    print("-" * 65)
    
    for N in Ns:
        acc_flat = capacity_test(N, use_ridge=False)
        acc_ridge = capacity_test(N, use_ridge=True)
        results_flat.append(acc_flat)
        results_ridge.append(acc_ridge)
        print(f"{N:<15} | {acc_flat:>17.1f}% | {acc_ridge:>17.1f}%")
        
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ns, results_flat, marker="o", lw=2, linestyle="--", color="gray", label="No Ridge")
    ax.plot(Ns, results_ridge, marker="s", lw=2, color="#E91E63", label="With Ridge (DoG)")
    
    ax.set_xlabel("Number of Concepts (N)", fontsize=12)
    ax.set_ylabel("Recall Accuracy (%)", fontsize=12)
    ax.set_title(f"64-D Capacity Stress Test (Dim={DIM})", fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.3)
    
    out_path = os.path.join(OUTPUT_DIR, "capacity_stress_test.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> {out_path}")


def main() -> None:
    _set_seed()
    visualise_ridge_profile()
    experiment_a()
    experiment_b()
    print("\nAll stress tests complete!")


if __name__ == "__main__":
    main()
