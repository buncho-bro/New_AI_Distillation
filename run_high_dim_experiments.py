"""
run_high_dim_experiments.py
============================

Automated verification of the 64-D High-Dimensional Concept Core.

Scenarios:
  A) Valley formation via repeated erosion along A->C and B->C
  B) Marble inference: convergence from (A+B)/2 toward C (origin)
  C) Weathering resilience: noise centre pruned after decay cycles

Produces a PCA-projected landscape visualisation saved as high_dim_result.png.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from sklearn.decomposition import PCA

from high_dim_concept_core import HighDimConceptCore, HDParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SEED = 42
DIM = 64
N_EROSION_CYCLES = 50
TRAJ_POINTS = 30
NORM_TARGET = 5.0


def _set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def make_anchors(dim: int = DIM) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create landmark vectors A, B, C in R^dim.

    C = origin, A and B are random unit vectors scaled to NORM_TARGET.
    """
    C = torch.zeros(dim)
    A = torch.randn(dim)
    A = A / A.norm() * NORM_TARGET
    B = torch.randn(dim)
    B = B / B.norm() * NORM_TARGET
    return A, B, C


# ---------------------------------------------------------------------------
# Scenario A: Valley formation
# ---------------------------------------------------------------------------

def scenario_a(
    core: HighDimConceptCore,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Erode A->C and B->C trajectories repeatedly."""
    print("\n" + "=" * 65)
    print("SCENARIO A: Valley Formation  (64-D Erosion)")
    print("=" * 65)
    print(f"  dim         = {DIM}")
    print(f"  ||A||       = {A.norm():.2f}")
    print(f"  ||B||       = {B.norm():.2f}")
    print(f"  ||A - B||   = {(A - B).norm():.2f}")
    print(f"  cycles      = {N_EROSION_CYCLES}")

    traj_AC = HighDimConceptCore.interpolate(A, C, TRAJ_POINTS)
    traj_BC = HighDimConceptCore.interpolate(B, C, TRAJ_POINTS)

    for cycle in range(1, N_EROSION_CYCLES + 1):
        core.erode(traj_AC)
        core.erode(traj_BC)
        if cycle % 5 == 0:
            core.weather()

    stats = core.weight_stats()
    print(f"\n  Memory bank : {stats['n']} centres")
    print(f"  Weight range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Weight total: {stats['total']:.2f}")

    # Validate: potential at C should be much lower than at a random point
    with torch.no_grad():
        H_at_C = core.potential(C.unsqueeze(0)).item()
        rand_pt = torch.randn(1, DIM) * NORM_TARGET
        H_at_rand = core.potential(rand_pt).item()

    print(f"\n  H(C)        = {H_at_C:.4f}")
    print(f"  H(random)   = {H_at_rand:.6f}")

    passed = H_at_C < H_at_rand - 0.5
    status = "PASS" if passed else "FAIL"
    print(f"\n  Valley depth check: {status}")
    print(f"    (C is {abs(H_at_C - H_at_rand):.3f} deeper than random background)")

    return traj_AC, traj_BC


# ---------------------------------------------------------------------------
# Scenario B: Marble inference
# ---------------------------------------------------------------------------

def scenario_b(
    core: HighDimConceptCore,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> Tuple[List[torch.Tensor], float]:
    """Roll marble from midpoint of A and B toward C."""
    print("\n" + "=" * 65)
    print("SCENARIO B: Marble Inference  (64-D Attractor Pull-in)")
    print("=" * 65)

    start = (A + B) / 2.0
    print(f"  Start       = midpoint(A, B),  ||start|| = {start.norm():.2f}")
    print(f"  Target      = C (origin),       ||C||    = {C.norm():.2f}")
    print(f"  Start->C    = {(start - C).norm():.2f}")

    final, path, n_steps = core.roll_marble(start)
    dist_to_C = (final - C).norm().item()

    print(f"\n  Final ||pos||  = {final.norm():.4f}")
    print(f"  Dist to C     = {dist_to_C:.4f}")
    print(f"  Steps taken   = {n_steps}")

    # Also try from a point on the A-channel
    start_A = A * 0.6  # 60% along A->C
    final_A, path_A, n_A = core.roll_marble(start_A)
    dist_A = (final_A - C).norm().item()
    print(f"\n  From A-channel (60%): dist to C = {dist_A:.4f}  ({n_A} steps)")

    # From a point on the B-channel
    start_B = B * 0.6
    final_B, path_B, n_B = core.roll_marble(start_B)
    dist_B = (final_B - C).norm().item()
    print(f"  From B-channel (60%): dist to C = {dist_B:.4f}  ({n_B} steps)")

    threshold = 2.0
    passed = dist_to_C < threshold and dist_A < threshold and dist_B < threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n  Convergence check (dist < {threshold}): {status}")

    return path, dist_to_C


# ---------------------------------------------------------------------------
# Scenario C: Weathering resilience
# ---------------------------------------------------------------------------

def scenario_c() -> None:
    """Add a noise centre and verify weathering removes it."""
    print("\n" + "=" * 65)
    print("SCENARIO C: Weathering Resilience  (64-D Noise Removal)")
    print("=" * 65)

    core_noise = HighDimConceptCore()

    # Random noise point
    D = torch.randn(DIM) * 3.0
    noise_traj = torch.stack([D, D + torch.randn(DIM) * 0.01])
    core_noise.erode(noise_traj, depth_gain=0.1)

    n_before = core_noise.num_centers
    w_before = core_noise.weights.clone()
    print(f"  Noise at ||D|| = {D.norm():.2f}")
    print(f"  Centres after noise erosion: {n_before}")
    print(f"  Weight: {w_before.sum().item():.4f}")

    N_WEATHER = 300
    for _ in range(N_WEATHER):
        core_noise.weather()

    n_after = core_noise.num_centers
    print(f"\n  After {N_WEATHER} weathering steps:")
    print(f"  Centres remaining: {n_after}")

    passed = n_after == 0
    status = "PASS" if passed else "FAIL"
    print(f"\n  Noise erasure check (0 centres remain): {status}")

    if not passed and core_noise.num_centers > 0:
        print(f"    Remaining weight: {core_noise.weights.sum().item():.6f}")


# ---------------------------------------------------------------------------
# Visualisation: PCA projection of 64-D landscape
# ---------------------------------------------------------------------------

def visualise_landscape(
    core: HighDimConceptCore,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    traj_AC: torch.Tensor,
    traj_BC: torch.Tensor,
    marble_path: List[torch.Tensor],
    marble_dist: float,
) -> None:
    """Create a comprehensive PCA-projected visualisation."""
    print("\n  Generating PCA visualisation ...")

    # ---- Collect all points for PCA fitting ----
    landmarks = torch.stack([A, B, C])
    centres_np = core.centers.detach().numpy()
    traj_AC_np = traj_AC.detach().numpy()
    traj_BC_np = traj_BC.detach().numpy()
    marble_np = torch.stack(marble_path).detach().numpy()
    landmarks_np = landmarks.detach().numpy()

    all_pts = np.vstack([centres_np, traj_AC_np, traj_BC_np, marble_np, landmarks_np])

    pca = PCA(n_components=2)
    pca.fit(all_pts)
    var_explained = pca.explained_variance_ratio_.sum() * 100

    # ---- Project everything to 2-D ----
    centres_2d = pca.transform(centres_np)
    traj_AC_2d = pca.transform(traj_AC_np)
    traj_BC_2d = pca.transform(traj_BC_np)
    marble_2d = pca.transform(marble_np)
    A_2d = pca.transform(A.numpy().reshape(1, -1)).squeeze()
    B_2d = pca.transform(B.numpy().reshape(1, -1)).squeeze()
    C_2d = pca.transform(C.numpy().reshape(1, -1)).squeeze()
    mid_2d = pca.transform(((A + B) / 2).numpy().reshape(1, -1)).squeeze()

    # ---- Background potential on a 2-D grid ----
    pad = 1.5
    x_min = min(centres_2d[:, 0].min(), marble_2d[:, 0].min(),
                A_2d[0], B_2d[0], C_2d[0]) - pad
    x_max = max(centres_2d[:, 0].max(), marble_2d[:, 0].max(),
                A_2d[0], B_2d[0], C_2d[0]) + pad
    y_min = min(centres_2d[:, 1].min(), marble_2d[:, 1].min(),
                A_2d[1], B_2d[1], C_2d[1]) - pad
    y_max = max(centres_2d[:, 1].max(), marble_2d[:, 1].max(),
                A_2d[1], B_2d[1], C_2d[1]) + pad

    grid_res = 120
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_64d = pca.inverse_transform(grid_2d)

    with torch.no_grad():
        H_grid = core.potential(torch.from_numpy(grid_64d).float()).numpy()
    H_grid = H_grid.reshape(xx.shape)

    # ---- Figure: 2-panel layout ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [3, 2]})

    # -- Panel 1: PCA landscape --
    ax = axes[0]
    cf = ax.contourf(xx, yy, H_grid, levels=40, cmap="terrain", alpha=0.9)
    ax.contour(xx, yy, H_grid, levels=40, colors="k", linewidths=0.2, alpha=0.3)
    plt.colorbar(cf, ax=ax, label="Potential  H(x)  [PCA slice]", shrink=0.85)

    # Centres (scaled by weight)
    w_norm = core.weights.detach().numpy()
    w_sizes = 20 + 80 * (w_norm / w_norm.max())
    sc = ax.scatter(
        centres_2d[:, 0], centres_2d[:, 1],
        c=w_norm, cmap="hot", s=w_sizes, edgecolors="k", linewidths=0.5,
        zorder=4, label="Kernel centres",
    )

    # Erosion trajectories
    ax.plot(traj_AC_2d[:, 0], traj_AC_2d[:, 1], "--", color="#90CAF9",
            lw=1.5, alpha=0.7, label="Erosion A->C")
    ax.plot(traj_BC_2d[:, 0], traj_BC_2d[:, 1], "--", color="#CE93D8",
            lw=1.5, alpha=0.7, label="Erosion B->C")

    # Marble trajectory
    ax.plot(marble_2d[:, 0], marble_2d[:, 1], "-", color="#FF1744",
            lw=2.5, alpha=0.9, zorder=5)
    ax.plot(marble_2d[0, 0], marble_2d[0, 1], "o", color="#FF1744",
            ms=12, zorder=6, label="Marble start")
    ax.plot(marble_2d[-1, 0], marble_2d[-1, 1], "*", color="#FF1744",
            ms=18, markeredgecolor="k", markeredgewidth=0.8, zorder=6,
            label="Marble end")

    # Landmarks
    for name, pos, color in [
        ("A (Dog)", A_2d, "#2196F3"),
        ("B (Cat)", B_2d, "#9C27B0"),
        ("C (Quadruped)", C_2d, "#4CAF50"),
    ]:
        ax.plot(pos[0], pos[1], "s", color=color, ms=13, markeredgecolor="k",
                markeredgewidth=2, zorder=7)
        ax.annotate(
            name, (pos[0], pos[1]),
            textcoords="offset points", xytext=(10, 10),
            fontsize=11, fontweight="bold", color=color,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

    ax.set_title(
        f"64-D Concept Core  (PCA projection, {var_explained:.1f}% var. explained)",
        fontsize=13, pad=10,
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_aspect("equal")

    # -- Panel 2: Convergence curve --
    ax2 = axes[1]
    dists = [
        (torch.stack(marble_path)[i] - C).norm().item()
        for i in range(len(marble_path))
    ]
    ax2.plot(dists, color="#FF1744", lw=2)
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Simulation step", fontsize=12)
    ax2.set_ylabel("||position - C||", fontsize=12)
    ax2.set_title("Marble Convergence to C", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # Annotate final distance
    ax2.annotate(
        f"Final dist = {marble_dist:.3f}",
        xy=(len(dists) - 1, dists[-1]),
        xytext=(-120, 40), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#FF1744"),
        fontsize=11, fontweight="bold", color="#FF1744",
    )

    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, "high_dim_result.png")
    fig.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> saved {path_out}")

    # ---- Bonus: weight distribution ----
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(w_norm, bins=30, color="#FF9800", edgecolor="k", alpha=0.85)
    ax3.set_xlabel("Weight  w_i", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title(f"Centre Weight Distribution  ({core.num_centers} centres)", fontsize=13)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    path_out2 = os.path.join(OUTPUT_DIR, "high_dim_weights.png")
    fig2.savefig(path_out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"    -> saved {path_out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _set_seed()

    print("\n" + "+" + "=" * 63 + "+")
    print("|  High-Dimensional Concept Core (64-D) - PyTorch              |")
    print("+" + "=" * 63 + "+")

    # Create anchors
    A, B, C = make_anchors()

    # Build core
    core = HighDimConceptCore()
    print(f"\n  Parameters: {core.p}")

    # Scenario A: Valley formation
    traj_AC, traj_BC = scenario_a(core, A, B, C)

    # Scenario B: Marble inference
    marble_path, marble_dist = scenario_b(core, A, B, C)

    # Scenario C: Weathering resilience
    scenario_c()

    # Visualisation
    visualise_landscape(core, A, B, C, traj_AC, traj_BC, marble_path, marble_dist)

    # Summary
    print("\n" + "=" * 65)
    print("ALL HIGH-DIM EXPERIMENTS COMPLETE")
    print("=" * 65)
    print(f"  Output directory: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.startswith("high_dim"):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    - {f}  ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
