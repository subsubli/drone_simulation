"""Standalone policy inference — get a target-velocity command from a trained policy given
(a waypoint path + the drone's current state). No simulator, no training data needed; just
the 3 files in a run dir: final.pt + config.json + obs_normalization.npz.

Usage from another script:

    from policy_infer import PolicyController
    ctl = PolicyController("<run_dir>", path=my_waypoints)   # my_waypoints: (N,3) array
    # every control step (e.g. 100 Hz):
    target_vel = ctl.get_velocity(drone_pos, quat, vel, angvel)   # -> (3,) m/s command
    # feed target_vel to your low-level velocity controller.
    ctl.reset()   # call at the start of each new run (clears slew/prev-action memory)

The policy maps a PATH-RELATIVE state to a velocity, so you can even swap the path at runtime
(ctl.set_path(new_waypoints)) — unlike pure-pursuit it doesn't need the whole plan in advance.
"""
import json
import os

import numpy as np
import torch

from evaluate_trained_policy import load_policy, load_normalization


class PolicyController:
    def __init__(self, run_dir, path, lookahead_dist=0.3,
                 slew_max_accel=2.0, control_freq_hz=100):
        """run_dir: folder with final.pt + config.json + obs_normalization.npz.
        path: (N,3) waypoints of the closed path to follow.
        lookahead_dist: meters ahead the look-ahead point sits (match the training default 0.3).
        slew_max_accel: m/s^2 slew cap re-applied to the policy output (REQUIRED for stability;
            None = off/raw). control_freq_hz: how often you'll call get_velocity()."""
        cfg = json.load(open(os.path.join(run_dir, 'config.json')))
        self.include_lookahead = bool(cfg.get('include_lookahead'))
        self.include_prev_action = bool(cfg.get('include_prev_action'))
        self.mean, self.std, action_bound = load_normalization(run_dir)
        self.policy = load_policy(run_dir, max_action=action_bound)
        self.max_delta_v = None if slew_max_accel is None else slew_max_accel / control_freq_hz
        self.set_path(path, lookahead_dist)
        self.reset()

    def set_path(self, path, lookahead_dist=0.3):
        """Set/replace the waypoint path (N,3). Recomputes the look-ahead step count from the
        path's own point spacing so lookahead_dist stays a physical distance in meters."""
        self.path = np.asarray(path, dtype=np.float64)
        n = len(self.path)
        perim = float(np.sum(np.linalg.norm(np.roll(self.path, -1, axis=0) - self.path, axis=1)))
        self.lookahead_steps = max(1, round(lookahead_dist / (perim / n)))

    def reset(self):
        """Clear slew / prev-action memory. Call before each new traversal."""
        self.prev_cmd = np.zeros(3)
        self.prev_raw = np.zeros(3)

    def closest_index(self, drone_pos):
        """Index of the nearest path point (useful for progress/coverage tracking)."""
        return int(np.argmin(np.linalg.norm(self.path - np.asarray(drone_pos), axis=1)))

    def get_velocity(self, drone_pos, quat, vel, angvel):
        """Map the current drone state to a target-velocity command.

        drone_pos: (3,) world position.  quat: (4,) attitude quaternion in PyBullet order
        (x, y, z, w).  vel: (3,) world linear velocity.  angvel: (3,) angular velocity.
        Returns: (3,) target_vel in m/s (slew-limited if slew_max_accel was set)."""
        p = np.asarray(drone_pos, dtype=np.float64)
        i = int(np.argmin(np.linalg.norm(self.path - p, axis=1)))          # nearest path point
        pos_err = self.path[i] - p                                          # perpendicular offset
        look = self.path[(i + self.lookahead_steps) % len(self.path)] - p   # look-ahead vector

        #### Build the observation in the EXACT order the policy trained on:
        #### [pos_err(3), quat(4), vel(3), angvel(3)] then look-ahead, then prev-action.
        obs = np.concatenate([pos_err, np.asarray(quat), np.asarray(vel),
                              np.asarray(angvel)]).astype(np.float32)
        if self.include_lookahead:
            obs = np.concatenate([obs, look.astype(np.float32)])
        if self.include_prev_action:
            obs = np.concatenate([obs, self.prev_raw.astype(np.float32)])
        obs = ((obs - self.mean) / self.std).astype(np.float32)             # normalize (REQUIRED)

        with torch.no_grad():
            raw = self.policy.act(torch.from_numpy(obs), deterministic=True).numpy()
        self.prev_raw = raw.copy()

        cmd = raw
        if self.max_delta_v is not None:                                    # slew cap (REQUIRED)
            d = raw - self.prev_cmd
            n = float(np.linalg.norm(d))
            if n > self.max_delta_v:
                cmd = self.prev_cmd + d * (self.max_delta_v / n)
        self.prev_cmd = cmd
        return cmd


if __name__ == '__main__':
    #### Tiny demo: build one circle path, feed a single made-up drone state, print the command.
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else sys.exit(
        "usage: python policy_infer.py <run_dir>")
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    path = np.stack([2.0 * np.cos(th), 2.0 * np.sin(th), np.ones_like(th)], axis=1)  # r=2 circle
    ctl = PolicyController(run, path=path)
    print(f"loaded: include_lookahead={ctl.include_lookahead}  lookahead_steps={ctl.lookahead_steps}")

    #### Drone sitting on the path at (2,0,1), stationary, level attitude (quat x,y,z,w = 0,0,0,1).
    drone_pos = np.array([2.0, 0.0, 1.0]); quat = np.array([0., 0., 0., 1.])
    vel = np.zeros(3); angvel = np.zeros(3)

    #### RAW policy output (slew off) = what the policy actually commands: full-speed tangent.
    raw_ctl = PolicyController(run, path=path, slew_max_accel=None)
    vr = raw_ctl.get_velocity(drone_pos, quat, vel, angvel)
    print(f"raw target_vel      = {vr.round(3)}   |v| = {np.linalg.norm(vr):.3f} m/s  (policy's real command)")

    #### With slew ON, the command ramps up from rest at max_accel/freq per step (here 0.02 m/s
    #### per step) -- so it starts small and converges to the raw command over ~70 steps. Shown
    #### for a few steps of the SAME state to make the ramp visible.
    print("slew-limited ramp (same state repeated):")
    for step in range(0, 40, 8):
        for _ in range(8 if step else 1):
            v = ctl.get_velocity(drone_pos, quat, vel, angvel)
        print(f"  after {step+1:>2} calls: |target_vel| = {np.linalg.norm(v):.3f} m/s")
    print("=> policy steers full-speed along the circle; slew just enforces smooth ramp-up.")
