"""
High-Dimensional Concept Core (64-D)
======================================

Scales the 2D potential-field toy model to 64 dimensions using a parametric
Gaussian kernel density representation (dynamic memory bank) and PyTorch
autograd for gradient-based marble inference.

Potential field:
    H(x) = - sum_i  w_i * exp( -||x - c_i||^2 / (2 * sigma^2) )

where c_i are learnt valley centres, w_i their accumulated depths, and
sigma controls the concept bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class HDParams:
    """Tunable parameters for the high-dimensional Concept Core.

    Attributes:
        dim:              Dimensionality of the concept space.
        sigma:            Bandwidth of each Gaussian kernel (concept radius).
        decay_rate:       Multiplicative weight decay per weathering step.
        prune_thresh:     Minimum weight below which a centre is deleted.
        depth_gain:       Default erosion depth per stimulus pass.
        marble_steps:     Maximum simulation ticks for marble rolling.
        marble_lr:        Effective step-size for marble dynamics (gravity * dt).
        marble_momentum:  Velocity damping factor (friction).
        marble_vel_thresh: Speed below which the marble is considered at rest.
    """
    dim: int = 64
    sigma: float = 1.0               # sigma_attr (attraction radius)
    use_ridge: bool = False          # Whether to use Mexican Hat (DoG) repulsion
    sigma_rep: float = 1.5           # Repulsion radius (should be > sigma)
    alpha_rep: float = 0.3           # Repulsion strength coefficient
    decay_rate: float = 0.99
    prune_thresh: float = 0.01
    depth_gain: float = 0.1
    marble_steps: int = 300
    marble_lr: float = 0.08
    marble_momentum: float = 0.85
    marble_vel_thresh: float = 1e-4
    marble_max_v: float = 0.2


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class HighDimConceptCore(nn.Module):
    """64-D concept landscape with parametric Gaussian potential.

    Valleys (attractors) are represented as weighted Gaussian kernels stored
    in a dynamic memory bank.  ``erode()`` deepens / creates kernels,
    ``weather()`` decays and prunes them, and ``roll_marble()`` performs
    autograd-based inference.
    """

    def __init__(self, params: Optional[HDParams] = None) -> None:
        super().__init__()
        self.p = params or HDParams()
        self._centers: Optional[torch.Tensor] = None   # (M, dim)
        self._weights: Optional[torch.Tensor] = None    # (M,)

    # -- properties ----------------------------------------------------------

    @property
    def num_centers(self) -> int:
        return 0 if self._centers is None else self._centers.shape[0]

    @property
    def centers(self) -> torch.Tensor:
        if self._centers is None:
            return torch.empty(0, self.p.dim)
        return self._centers

    @property
    def weights(self) -> torch.Tensor:
        if self._weights is None:
            return torch.empty(0)
        return self._weights

    # -- potential -----------------------------------------------------------

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the potential field at positions *x*.

        Args:
            x: (B, dim) batch of query points.

        Returns:
            H: (B,) potential values (negative in valleys, positive on ridges).
        """
        if self.num_centers == 0:
            return torch.zeros(x.shape[0], dtype=x.dtype)

        # (B, M, D)
        diffs = x.unsqueeze(1) - self._centers.unsqueeze(0)
        dist_sq = (diffs ** 2).sum(dim=-1)                     # (B, M)
        
        # Attraction kernel
        K = torch.exp(-dist_sq / (2.0 * self.p.sigma ** 2))
        
        # Repulsion kernel (Mexican Hat / Difference of Gaussians)
        if self.p.use_ridge:
            K_rep = torch.exp(-dist_sq / (2.0 * self.p.sigma_rep ** 2))
            K = K - self.p.alpha_rep * K_rep

        H = -(K * self._weights.unsqueeze(0)).sum(dim=-1)    # (B,)
        return H

    # -- erosion (learning) --------------------------------------------------

    def erode(
        self,
        trajectory: torch.Tensor,
        depth_gain: Optional[float] = None,
    ) -> None:
        """Carve valleys along *trajectory* (learning / Hebbian reinforcement).

        Points within ``sigma * 0.5`` of an existing centre merge
        (weight is added); otherwise a new centre is spawned.
        Erosion depth increases linearly toward the trajectory end
        (flow accumulation).

        Args:
            trajectory: (T, dim) ordered waypoints of the stimulus.
            depth_gain: Override for per-pass erosion depth.
        """
        if depth_gain is None:
            depth_gain = self.p.depth_gain

        T = trajectory.shape[0]
        merge_thresh = self.p.sigma * 0.5

        for idx in range(T):
            point = trajectory[idx: idx + 1]  # (1, D)
            # Flow-accumulation weight: deeper downstream
            w = depth_gain * (0.3 + 0.7 * idx / max(T - 1, 1))

            if self.num_centers == 0:
                self._centers = point.clone().detach()
                self._weights = torch.tensor([w])
                continue

            dists = torch.cdist(point, self._centers).squeeze(0)  # (M,)
            min_dist, min_idx = dists.min(dim=0)

            if min_dist.item() < merge_thresh:
                self._weights = self._weights.clone()
                self._weights[min_idx.item()] += w
            else:
                self._centers = torch.cat(
                    [self._centers, point.clone().detach()], dim=0
                )
                self._weights = torch.cat(
                    [self._weights, torch.tensor([w])]
                )

    # -- autonomous learning -------------------------------------------------

    def rain_and_erode(self, drops: torch.Tensor, steps: int = 40, depth_gain: float = 0.02) -> torch.Tensor:
        """Autonomous unsupervised concept discovery (self-organising flow).
        
        A batch of observational 'raindrops' falls on the landscape.
        In each step, they carve the surface (erode) and then roll downhill 
        into the valleys they collectively created. Dense data regions 
        autonomously form deep attractors.
        
        Args:
            drops: (B, dim) initial drop coordinates.
            steps: Number of simulation steps.
            depth_gain: Erosion depth per drop per step.
            
        Returns:
            final_drops: (B, dim) final resting coordinates of the drops.
        """
        B = drops.shape[0]
        x = drops.clone().detach().float()
        v = torch.zeros_like(x)
        merge_thresh = self.p.sigma * 0.5
        
        for step in range(steps):
            # 1. Erode at current positions
            for i in range(B):
                point = x[i:i+1]
                w = depth_gain
                
                if self.num_centers == 0:
                    self._centers = point.clone().detach()
                    self._weights = torch.tensor([w])
                    continue
                    
                dists = torch.cdist(point, self._centers).squeeze(0)
                min_dist, min_idx = dists.min(dim=0)
                
                if min_dist.item() < merge_thresh:
                    self._weights = self._weights.clone()
                    self._weights[min_idx.item()] += w
                else:
                    self._centers = torch.cat([self._centers, point.clone().detach()], dim=0)
                    self._weights = torch.cat([self._weights, torch.tensor([w])])
            
            # 2. Weathering to clean up shallow noise trails
            if step % 3 == 0:
                self.weather()
                
            # 3. Roll downhill (Physics)
            x_var = x.clone().detach().requires_grad_(True)
            H = self.potential(x_var)
            H.sum().backward()
            grad = x_var.grad.detach()
            
            v = self.p.marble_momentum * v - self.p.marble_lr * grad
            
            # Terminal velocity (clip)
            v_norm = v.norm(dim=-1, keepdim=True)
            mask = v_norm > self.p.marble_max_v
            v = torch.where(mask, v * (self.p.marble_max_v / v_norm.clamp(min=1e-9)), v)
            
            x = x + v
            # Spherical constraint: project back to unit sphere
            x = x / x.norm(dim=-1, keepdim=True)
            
        return x.detach()

    # -- weathering (forgetting) ---------------------------------------------

    def weather(self) -> int:
        """Decay all weights and prune negligible centres.

        Returns:
            Number of centres pruned this step.
        """
        if self.num_centers == 0:
            return 0

        n_before = self.num_centers
        self._weights = self._weights * self.p.decay_rate

        mask = self._weights >= self.p.prune_thresh
        self._centers = self._centers[mask].clone()
        self._weights = self._weights[mask].clone()

        if self.num_centers == 0:
            self._centers = None
            self._weights = None

        return n_before - self.num_centers

    # -- marble inference ----------------------------------------------------

    def roll_marble(
        self,
        start_pos: torch.Tensor,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """Roll a marble from *start_pos* following -nabla H.

        Uses PyTorch autograd to compute the potential gradient and
        applies Newtonian dynamics with inertia and friction damping.

        Args:
            start_pos: (dim,) or (1, dim) initial position vector.

        Returns:
            ``(final_pos, trajectory, n_steps)``
        """
        if steps is None:
            steps = self.p.marble_steps
        if lr is None:
            lr = self.p.marble_lr
        if momentum is None:
            momentum = self.p.marble_momentum

        if start_pos.dim() == 1:
            start_pos = start_pos.unsqueeze(0)

        x = start_pos.clone().detach().float()   # (1, D)
        v = torch.zeros_like(x)                   # (1, D)
        path: list[torch.Tensor] = [x.squeeze(0).clone()]

        n_steps = steps
        for step_i in range(steps):
            x_var = x.clone().detach().requires_grad_(True)
            H = self.potential(x_var)
            H.sum().backward()
            grad = x_var.grad.detach()

            # Velocity update: momentum damping + gradient force
            v = momentum * v - lr * grad
            
            # Apply terminal velocity (prevent overshoot in deep valleys)
            v_norm = v.norm()
            if v_norm > self.p.marble_max_v:
                v = v * (self.p.marble_max_v / v_norm)

            x = x + v
            # Spherical constraint: project back to unit sphere
            x = x / x.norm(dim=-1, keepdim=True)

            path.append(x.squeeze(0).clone().detach())

            if v.norm().item() < self.p.marble_vel_thresh:
                n_steps = step_i + 1
                break

        return x.squeeze(0).detach(), path, n_steps

    # -- utilities -----------------------------------------------------------

    @staticmethod
    def interpolate(
        start: torch.Tensor,
        end: torch.Tensor,
        n_points: int = 30,
    ) -> torch.Tensor:
        """Generate a linear interpolation trajectory (T, dim)."""
        t = torch.linspace(0.0, 1.0, n_points).unsqueeze(1)  # (T, 1)
        return start.unsqueeze(0) * (1.0 - t) + end.unsqueeze(0) * t

    def weight_stats(self) -> dict:
        """Return summary statistics of the current weight distribution."""
        if self.num_centers == 0:
            return {"n": 0, "min": 0, "max": 0, "mean": 0, "total": 0}
        w = self._weights
        return {
            "n": self.num_centers,
            "min": w.min().item(),
            "max": w.max().item(),
            "mean": w.mean().item(),
            "total": w.sum().item(),
        }
