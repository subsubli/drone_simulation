"""Signed geometric progress on a periodic path, independent of reference-clock time."""
import numpy as np


class PathProgress:
    def __init__(self, at, period, initial_position, segments=2048):
        self.n = segments
        self.points = np.array([at(period*i/segments)[0] for i in range(segments)])
        self.edges = np.roll(self.points, -1, axis=0)-self.points
        self.length2 = np.sum(self.edges**2, axis=1)
        self.min_length = float(np.sqrt(self.length2.min()))
        self.phase = 0.  # rollouts start at at(0), including figure-eight crossings
        self.previous = np.asarray(initial_position).copy()
        self.rejected = 0

    def update(self, position):
        position = np.asarray(position)
        distance = float(np.linalg.norm(position-self.previous))
        self.previous = position.copy()
        if distance < 1e-10:
            return  # a stationary aircraft must never advance with the reference clock
        radius = min(self.n//8, max(8, int(np.ceil(2*distance/self.min_length))+2))
        indices = np.arange(int(np.floor(self.phase))-radius, int(np.floor(self.phase))+radius+1)
        wrapped = indices % self.n
        edges = self.edges[wrapped]
        u = np.clip(np.sum((position-self.points[wrapped])*edges, axis=1)/self.length2[wrapped], 0, 1)
        projected = self.points[wrapped]+u[:, None]*edges
        errors = np.sum((projected-position)**2, axis=1)
        candidates = np.flatnonzero(errors <= errors.min()+1e-14)
        best = candidates[np.argmin(np.abs(indices[candidates]+u[candidates]-self.phase))]
        # Do not creep along the search boundary when far from the tracked branch.
        if best in (0, len(indices)-1):
            self.rejected += 1
            return
        self.phase = float(indices[best]+u[best])

    def metrics(self, elapsed, budget, period):
        laps = self.phase/self.n
        return dict(net_laps=laps, target_laps_elapsed=elapsed/period,
                    target_laps_budget=budget/period,
                    progress_ratio_elapsed=laps/(elapsed/period) if elapsed else 0.,
                    progress_ratio_budget=laps/(budget/period),
                    progress_projection_rejections=self.rejected)
