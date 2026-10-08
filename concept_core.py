"""
Concept Core: Self-Organizing Potential Field AI Architecture
=============================================================

A 2D physics simulation toy model that models thought and memory as
continuous potential-field phenomena. Concepts form as stable valleys
(attractors) through erosion (learning), while weathering (forgetting)
provides regularization by smoothing unused shallow depressions.

Key metaphors:
  - Potential field H(x,y)  → Concept landscape (energy-based model)
  - Erosion along trajectory → Learning / attractor formation (Hebbian)
  - Weathering / smoothing   → Forgetting / regularization / noise removal
  - Marble rolling downhill  → Inference / convergence to nearest concept
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ---------------------------------------------------------------------------
# Tunable physical parameters
# ---------------------------------------------------------------------------

@dataclass
class PhysicsParams:
    """All tunable parameters for the Concept Core simulation.

    Attributes:
        grid_size:          Side length of the square grid.
        erosion_rate:       η – depth of erosion per stimulus pass (learning rate).
        weathering_rate:    λ – strength of per-step smoothing (forgetting coeff).
        friction:           γ – multiplicative velocity damping each tick.
        gravity:            Scaling factor for the gradient-descent force.
        erosion_radius:     σ of Gaussian spread around eroded trajectory (pixels).
        dt:                 Integration time-step for marble dynamics.
        velocity_threshold: Speed below which the marble is considered at rest.
        max_marble_steps:   Safety cap on simulation ticks for marble rolling.
    """
    grid_size: int = 100
    erosion_rate: float = 0.05         # η
    weathering_rate: float = 0.005     # λ
    friction: float = 0.85             # γ
    gravity: float = 5.0
    erosion_radius: float = 3.0
    dt: float = 0.5
    velocity_threshold: float = 0.001
    max_marble_steps: int = 2000


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

class ConceptCore:
    """2-D potential field that self-organises through erosion and weathering.

    The field ``H`` starts flat (all zeros). Repeated ``erode()`` calls carve
    valleys along stimulus trajectories, ``weathering()`` gently smooths the
    surface, and ``roll_marble()`` performs gradient-descent inference.
    """

    def __init__(self, params: Optional[PhysicsParams] = None) -> None:
        self.params = params or PhysicsParams()
        p = self.params
        self.H: np.ndarray = np.zeros((p.grid_size, p.grid_size), dtype=np.float64)
        self.step_count: int = 0

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _interpolate_trajectory(
        trajectory: List[Tuple[float, float]],
        density: float = 0.5,
    ) -> np.ndarray:
        """Densely resample a piecewise-linear trajectory.

        Returns an (N, 2) array of (x, y) points spaced ≈ ``density`` apart.
        """
        points: list[Tuple[float, float]] = []
        for i in range(len(trajectory) - 1):
            x0, y0 = trajectory[i]
            x1, y1 = trajectory[i + 1]
            seg_len = np.hypot(x1 - x0, y1 - y0)
            n_pts = max(int(seg_len / density), 2)
            xs = np.linspace(x0, x1, n_pts, endpoint=(i == len(trajectory) - 2))
            ys = np.linspace(y0, y1, n_pts, endpoint=(i == len(trajectory) - 2))
            points.extend(zip(xs, ys))
        return np.asarray(points)

    # -- core operations -----------------------------------------------------

    def erode(self, trajectory: List[Tuple[float, float]]) -> None:
        """Erode the potential field along *trajectory* (learning).

        Each call lowers ``H`` by at most ``erosion_rate`` along the path,
        with a Gaussian cross-section of width ``erosion_radius``.
        Erosion deepens toward the end of the trajectory (flow accumulation),
        creating a natural downhill slope.

        Args:
            trajectory: Ordered list of (x, y) waypoints.
        """
        p = self.params
        dense = self._interpolate_trajectory(trajectory)
        n_pts = len(dense)

        # Build a soft mask by placing weighted impulses at each sample.
        # Weight increases linearly from 0.3 at the start to 1.0 at the end,
        # simulating greater erosion where water accumulates downstream.
        mask = np.zeros_like(self.H)
        for idx, (x, y) in enumerate(dense):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < p.grid_size and 0 <= xi < p.grid_size:
                weight = 0.3 + 0.7 * (idx / max(n_pts - 1, 1))
                mask[yi, xi] += weight

        # Blur to get a smooth channel profile.
        mask = gaussian_filter(mask, sigma=p.erosion_radius)

        # Normalise so peak erosion equals erosion_rate.
        peak = mask.max()
        if peak > 0:
            mask /= peak

        self.H -= p.erosion_rate * mask
        self.step_count += 1

    def weathering(self) -> None:
        """Apply one step of weathering (forgetting / regularisation).

        Blends the field toward a Gaussian-smoothed version of itself and
        gently pulls the entire surface toward zero.
        """
        p = self.params
        smoothed = gaussian_filter(self.H, sigma=2.0)
        # Diffusion toward smoothed version
        self.H += p.weathering_rate * (smoothed - self.H)
        # Global decay toward zero (erases noise over time)
        self.H *= 1.0 - p.weathering_rate

    def roll_marble(
        self,
        start_pos: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], List[Tuple[float, float]]]:
        """Drop a marble and let it roll to the nearest attractor.

        Simulates Newtonian dynamics with gradient force, inertia, and
        friction damping until the marble velocity drops below threshold.

        Args:
            start_pos: Initial (x, y) coordinates.

        Returns:
            ``(final_pos, trajectory)`` – final resting (x, y) and full path.
        """
        p = self.params
        x, y = float(start_pos[0]), float(start_pos[1])
        vx, vy = 0.0, 0.0
        path: list[Tuple[float, float]] = [(x, y)]

        # Pre-compute gradient fields (row=dH/dy, col=dH/dx)
        grad_y, grad_x = np.gradient(self.H)

        for _ in range(p.max_marble_steps):
            # Bilinear interpolation of gradient at continuous position
            coords = np.array([[y], [x]])
            gx = float(map_coordinates(grad_x, coords, order=1, mode="nearest")[0])
            gy = float(map_coordinates(grad_y, coords, order=1, mode="nearest")[0])

            # Acceleration = –gravity · ∇H  (roll downhill)
            ax = -p.gravity * gx
            ay = -p.gravity * gy

            # Velocity update: damping + force
            vx = p.friction * vx + ax * p.dt
            vy = p.friction * vy + ay * p.dt

            # Position update
            x += vx * p.dt
            y += vy * p.dt

            # Clamp to grid boundaries
            x = float(np.clip(x, 0, p.grid_size - 1))
            y = float(np.clip(y, 0, p.grid_size - 1))

            path.append((x, y))

            if np.hypot(vx, vy) < p.velocity_threshold:
                break

        return (x, y), path

    def get_depth_at(self, x: float, y: float) -> float:
        """Return the interpolated potential at a continuous (x, y)."""
        coords = np.array([[y], [x]])
        return float(map_coordinates(self.H, coords, order=1, mode="nearest")[0])

    def reset(self) -> None:
        """Reset the field to flat zero."""
        self.H[:] = 0.0
        self.step_count = 0

    def copy(self) -> "ConceptCore":
        """Return a deep copy of this core (for branching experiments)."""
        return copy.deepcopy(self)
