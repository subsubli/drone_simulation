"""Path lookahead matching shape_dataset and policy_infer defaults.

Requires a planned closed path in traversal order with roughly uniform spacing.
Do not substitute recorded drone positions for the planned reference path.
"""
import numpy as np


class PathLookahead:
    """Nearest waypoint followed by a fixed number of forward waypoints.

    Reverse waypoint order for the opposite direction. At self-intersections,
    nearest-point selection is ambiguous, just as in the original tracker;
    figure-eight paths need an additional phase/progress constraint.
    """

    def __init__(self, path, lookahead_dist=0.3):
        self.path = np.array(path, dtype=np.float64, copy=True)
        if self.path.ndim != 2 or self.path.shape[1] != 3 or len(self.path) < 2:
            raise ValueError('path must have shape (N, 3), N >= 2')
        if not np.isfinite(self.path).all():
            raise ValueError('path must be finite')
        if not np.isfinite(lookahead_dist) or lookahead_dist <= 0:
            raise ValueError('lookahead_dist must be finite and positive')
        perimeter = np.linalg.norm(np.roll(self.path, -1, axis=0)-self.path, axis=1).sum()
        if perimeter <= 1e-12:
            raise ValueError('path must have nonzero length')
        self.lookahead_steps = max(1, round(lookahead_dist/(perimeter/len(self.path))))

    def compute(self, drone_pos):
        """Return (nearest-path error, lookahead vector, closest index).

        Vectors are world-frame meters. X500 obs[:3] is time-indexed target
        error, so do not replace it with nearest-path error for existing models.
        """
        pos = np.asarray(drone_pos, dtype=np.float64)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError('drone_pos must be a finite (3,) vector')
        closest = int(np.argmin(np.linalg.norm(self.path-pos, axis=1)))
        ahead = (closest+self.lookahead_steps) % len(self.path)
        return self.path[closest]-pos, self.path[ahead]-pos, closest
