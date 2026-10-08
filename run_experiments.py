"""
run_experiments.py – Automated verification of the Concept Core toy model.
==========================================================================

Runs three scenarios that validate the key properties of the self-organising
potential-field architecture:

  A) Y-shaped valley formation  (abstraction / common concept acquisition)
  B) Marble convergence test    (inference / attractor pull-in)
  C) Weathering resilience test (noise removal / forgetting)

Produces publication-quality PNG images and a summary GIF animation.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend for CI / script use
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for projection='3d'
import numpy as np

from concept_core import ConceptCore, PhysicsParams

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Landmark coordinates (x, y) on a 100×100 grid
POINT_A = (20.0, 75.0)   # "Dog" stimulus origin
POINT_B = (80.0, 75.0)   # "Cat" stimulus origin
POINT_C = (50.0, 25.0)   # "Quadruped" – common concept target

LANDMARKS: Dict[str, Tuple[float, float]] = {
    "A (Dog)": POINT_A,
    "B (Cat)": POINT_B,
    "C (Quadruped)": POINT_C,
}

N_EROSION_CYCLES = 60   # number of stimulus repetitions
WEATHERING_PER_CYCLE = 2 # weathering steps per erosion cycle


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_landmarks(
    ax: plt.Axes,
    landmarks: Dict[str, Tuple[float, float]],
    fontsize: int = 11,
) -> None:
    """Overlay labelled landmark markers on an axes."""
    for name, (lx, ly) in landmarks.items():
        ax.plot(
            lx, ly, "s",
            color="white", markersize=11,
            markeredgecolor="black", markeredgewidth=2, zorder=6,
        )
        ax.annotate(
            name, (lx, ly),
            textcoords="offset points", xytext=(8, 8),
            fontsize=fontsize, fontweight="bold", color="white",
            path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        )


def plot_contour(
    core: ConceptCore,
    title: str,
    filename: str,
    marble_paths: Optional[List[np.ndarray]] = None,
    landmarks: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    """Save a 2-D filled-contour plot of the potential field."""
    fig, ax = plt.subplots(figsize=(9, 8))

    gs = core.params.grid_size
    X, Y = np.meshgrid(np.arange(gs), np.arange(gs))

    cf = ax.contourf(X, Y, core.H, levels=40, cmap="terrain")
    ax.contour(X, Y, core.H, levels=40, colors="k", linewidths=0.25, alpha=0.4)
    plt.colorbar(cf, ax=ax, label="Potential  H(x, y)")

    # Marble trajectories
    if marble_paths:
        palette = ["#FF1744", "#D500F9", "#FF9100", "#00E5FF"]
        for i, traj in enumerate(marble_paths):
            c = palette[i % len(palette)]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=2.2, alpha=0.85)
            ax.plot(traj[0, 0], traj[0, 1], "o", color=c, ms=10, zorder=5,
                    label=f"Start {i+1}")
            ax.plot(traj[-1, 0], traj[-1, 1], "*", color=c, ms=16, zorder=5,
                    markeredgecolor="black", markeredgewidth=0.8,
                    label=f"End {i+1}")

    if landmarks:
        _draw_landmarks(ax, landmarks)

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    if marble_paths:
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {path}")


def plot_surface_3d(
    core: ConceptCore,
    title: str,
    filename: str,
) -> None:
    """Save a 3-D surface plot of the potential field."""
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    gs = core.params.grid_size
    step = max(1, gs // 60)
    X, Y = np.meshgrid(np.arange(0, gs, step), np.arange(0, gs, step))
    Z = core.H[::step, ::step]

    ax.plot_surface(X, Y, Z, cmap="terrain", alpha=0.92,
                    edgecolors="k", linewidth=0.08)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("H")
    ax.view_init(elev=40, azim=230)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {path}")


def save_animation(
    frames: List[dict],
    filename: str,
    fps: int = 4,
) -> None:
    """Create a GIF from a list of frame snapshots."""
    fig, ax = plt.subplots(figsize=(9, 8))

    def _update(idx: int) -> None:
        ax.clear()
        d = frames[idx]
        H = d["H"]
        gs = H.shape[0]
        X, Y = np.meshgrid(np.arange(gs), np.arange(gs))
        ax.contourf(X, Y, H, levels=40, cmap="terrain")
        ax.contour(X, Y, H, levels=40, colors="k", linewidths=0.2, alpha=0.3)
        if "landmarks" in d:
            _draw_landmarks(ax, d["landmarks"], fontsize=10)
        ax.set_title(d.get("title", f"Step {idx}"), fontsize=13)
        ax.set_aspect("equal")

    anim = animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000 // fps)
    path = os.path.join(OUTPUT_DIR, filename)
    anim.save(path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"    → saved {path}")


# ---------------------------------------------------------------------------
# Scenario A – Y-shaped valley formation (abstraction)
# ---------------------------------------------------------------------------

def scenario_a() -> ConceptCore:
    """Erode two converging channels A→C and B→C, forming a Y-shaped valley."""
    print("\n" + "=" * 65)
    print("SCENARIO A: Y-shaped Valley Formation  (Abstraction)")
    print("=" * 65)
    print(f"  A = {POINT_A}  (Dog stimulus)")
    print(f"  B = {POINT_B}  (Cat stimulus)")
    print(f"  C = {POINT_C}  (Quadruped – shared concept)")
    print(f"  Erosion cycles: {N_EROSION_CYCLES}")

    core = ConceptCore()
    anim_frames: list[dict] = []

    # Capture initial state
    anim_frames.append({
        "H": core.H.copy(),
        "landmarks": LANDMARKS,
        "title": "Step 0 – Flat Terrain",
    })

    for cycle in range(1, N_EROSION_CYCLES + 1):
        core.erode([POINT_A, POINT_C])  # Dog → Quadruped
        core.erode([POINT_B, POINT_C])  # Cat → Quadruped
        for _ in range(WEATHERING_PER_CYCLE):
            core.weathering()

        # Snapshot every 10 cycles for the animation
        if cycle % 10 == 0 or cycle == 1:
            anim_frames.append({
                "H": core.H.copy(),
                "landmarks": LANDMARKS,
                "title": f"After {cycle} erosion cycles",
            })

    # --- Validation ---------------------------------------------------------
    depth_C = core.get_depth_at(*POINT_C)
    depth_bg = core.get_depth_at(10, 10)  # background (untouched)
    depth_mid_AC = core.get_depth_at(35, 50)  # midpoint of A→C channel
    depth_mid_BC = core.get_depth_at(65, 50)  # midpoint of B→C channel

    print(f"\n  Depth at C (shared valley)   : {depth_C:.4f}")
    print(f"  Depth at mid(A→C) channel    : {depth_mid_AC:.4f}")
    print(f"  Depth at mid(B→C) channel    : {depth_mid_BC:.4f}")
    print(f"  Depth at background (10,10)  : {depth_bg:.6f}")

    passed = depth_C < depth_bg - 0.5 and depth_C < -0.3
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Valley formation check: {status}")
    print(f"    (C is {abs(depth_C - depth_bg):.3f} deeper than background)")

    # --- Plots --------------------------------------------------------------
    print("\n  Generating visualisations …")
    plot_contour(core, "Scenario A – Y-shaped Valley (contour)",
                 "scenario_a_contour.png", landmarks=LANDMARKS)
    plot_surface_3d(core, "Scenario A – Y-shaped Valley (3D surface)",
                    "scenario_a_3d.png")
    save_animation(anim_frames, "scenario_a_evolution.gif")

    return core


# ---------------------------------------------------------------------------
# Scenario B – Marble inference test (attractor pull-in)
# ---------------------------------------------------------------------------

def scenario_b(core: ConceptCore) -> None:
    """Drop marbles near the Y-junction; verify they converge toward C."""
    print("\n" + "=" * 65)
    print("SCENARIO B: Marble Inference Test  (Attractor Pull-in)")
    print("=" * 65)

    # --- Test 1: Drop from A-channel (slightly off center) ----------------
    start1 = (30.0, 60.0)
    print(f"  Marble 1 start : {start1}  (inside A-channel)")
    final1, path1 = core.roll_marble(start1)
    path1_arr = np.array(path1)
    dist1 = np.hypot(final1[0] - POINT_C[0], final1[1] - POINT_C[1])
    print(f"  Final position : ({final1[0]:.2f}, {final1[1]:.2f})")
    print(f"  Distance to C  : {dist1:.2f}")
    print(f"  Steps taken    : {len(path1)}")

    # --- Test 2: Drop from inside the merged channel (above C) -----------
    start2 = (50.0, 40.0)
    print(f"\n  Marble 2 start : {start2}  (inside merged channel)")
    final2, path2 = core.roll_marble(start2)
    path2_arr = np.array(path2)
    dist2 = np.hypot(final2[0] - POINT_C[0], final2[1] - POINT_C[1])
    print(f"  Final position : ({final2[0]:.2f}, {final2[1]:.2f})")
    print(f"  Distance to C  : {dist2:.2f}")
    print(f"  Steps taken    : {len(path2)}")

    # --- Test 3: Drop from B-channel (slightly off center) ---------------
    start3 = (70.0, 60.0)
    print(f"\n  Marble 3 start : {start3}  (inside B-channel)")
    final3, path3 = core.roll_marble(start3)
    path3_arr = np.array(path3)
    dist3 = np.hypot(final3[0] - POINT_C[0], final3[1] - POINT_C[1])
    print(f"  Final position : ({final3[0]:.2f}, {final3[1]:.2f})")
    print(f"  Distance to C  : {dist3:.2f}")
    print(f"  Steps taken    : {len(path3)}")

    # --- Validation -------------------------------------------------------
    passed = dist1 < 12.0 and dist2 < 12.0 and dist3 < 12.0
    status = "PASS" if passed else "FAIL"
    print(f"\n  Convergence check (all dist < 12): {status}")
    print(f"    marble 1 dist={dist1:.1f}, marble 2 dist={dist2:.1f}, marble 3 dist={dist3:.1f}")

    # --- Plots --------------------------------------------------------------
    print("\n  Generating visualisations ...")
    extended_landmarks = {**LANDMARKS, "Start 1": start1, "Start 2": start2, "Start 3": start3}
    plot_contour(
        core,
        "Scenario B - Marble Trajectories",
        "scenario_b_marble.png",
        marble_paths=[path1_arr, path2_arr, path3_arr],
        landmarks=extended_landmarks,
    )


# ---------------------------------------------------------------------------
# Scenario C – Weathering resilience (noise removal)
# ---------------------------------------------------------------------------

def scenario_c() -> None:
    """Apply a single noise impulse and verify weathering erases it."""
    print("\n" + "=" * 65)
    print("SCENARIO C: Weathering Resilience  (Noise Removal)")
    print("=" * 65)

    core = ConceptCore()

    # Random noise point
    rng = np.random.default_rng(42)
    D = (rng.uniform(20, 80), rng.uniform(20, 80))
    print(f"  Noise impulse at D = ({D[0]:.1f}, {D[1]:.1f})")

    # Single erosion burst (noise)
    core.erode([D, (D[0] + 0.1, D[1] + 0.1)])  # tiny segment = point impulse
    depth_before = core.get_depth_at(*D)
    print(f"  Depth at D after impulse  : {depth_before:.4f}")

    # Snapshot before weathering
    H_before = core.H.copy()

    # Apply many weathering steps
    N_WEATHER = 500
    for _ in range(N_WEATHER):
        core.weathering()

    depth_after = core.get_depth_at(*D)
    remaining_fraction = abs(depth_after / depth_before) if depth_before != 0 else 0.0
    print(f"  Depth at D after {N_WEATHER} weathering steps: {depth_after:.6f}")
    print(f"  Remaining fraction: {remaining_fraction:.4%}")

    passed = remaining_fraction < 0.10  # less than 10% remains
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  Noise erasure check (< 10% remaining): {status}")

    # --- Side-by-side comparison plot ---------------------------------------
    print("\n  Generating visualisations …")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, H, label in [
        (axes[0], H_before, f"Before Weathering\nDepth at D = {depth_before:.4f}"),
        (axes[1], core.H, f"After {N_WEATHER} Weathering Steps\nDepth at D = {depth_after:.6f}"),
    ]:
        gs = core.params.grid_size
        X, Y = np.meshgrid(np.arange(gs), np.arange(gs))
        cf = ax.contourf(X, Y, H, levels=40, cmap="terrain")
        ax.contour(X, Y, H, levels=40, colors="k", linewidths=0.2, alpha=0.3)
        plt.colorbar(cf, ax=ax, label="H(x, y)")
        ax.plot(D[0], D[1], "x", color="red", markersize=14, markeredgewidth=3, zorder=5)
        ax.annotate("D (noise)", (D[0], D[1]), textcoords="offset points",
                    xytext=(8, 8), fontsize=11, fontweight="bold", color="red",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")

    fig.suptitle("Scenario C – Weathering Resilience", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "scenario_c_weathering.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {path}")


# ---------------------------------------------------------------------------
# Cross-section plot (bonus): depth profile across the Y-junction
# ---------------------------------------------------------------------------

def plot_cross_section(core: ConceptCore) -> None:
    """Plot a 1-D cross-section through the Y-junction at various y-levels."""
    print("\n  Generating cross-section plot …")
    fig, ax = plt.subplots(figsize=(10, 5))

    xs = np.arange(core.params.grid_size)
    for y_level, color, label in [
        (75, "#E91E63", f"y = 75  (near A & B)"),
        (50, "#FF9800", f"y = 50  (mid-channel)"),
        (25, "#2196F3", f"y = 25  (near C)"),
    ]:
        ax.plot(xs, core.H[y_level, :], color=color, lw=2, label=label)

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Potential  H(x, y)", fontsize=12)
    ax.set_title("Cross-section: Depth Profile at Different Y Levels", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "cross_section.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "+" + "=" * 63 + "+")
    print("|  Concept Core - Self-Organising Potential Field Toy Model      |")
    print("+" + "=" * 63 + "+")

    # Scenario A
    core_trained = scenario_a()

    # Scenario B (uses trained terrain from A)
    scenario_b(core_trained)

    # Cross-section analysis
    plot_cross_section(core_trained)

    # Scenario C (independent)
    scenario_c()

    # Summary
    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 65)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    • {f}  ({size_kb:.0f} KB)")
    print()


if __name__ == "__main__":
    main()
