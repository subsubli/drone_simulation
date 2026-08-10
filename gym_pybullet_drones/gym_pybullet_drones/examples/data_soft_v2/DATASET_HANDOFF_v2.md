# Soft dataset v2 — velocity-only, **attitude D-gain restored to default (att_d_gain_scale = 1.0)**

`merged1.5M_soft_v2.csv.gz` — 1,502,262 rows / 100 Hz / pure-pursuit + DSLPID velocity-only expert.
Drop-in successor to `data_soft/merged1.5M_soft.csv.gz` (v1). **Only one setting changed vs v1.**

## What changed vs v1 (and why)

| | v1 (`data_soft`) | **v2 (`data_soft_v2`)** |
|---|---|---|
| `--att_d_gain_scale` | 0.3 | **1.0 (default)** |
| everything else | — | **identical** (same shapes, seeds, kicks, speeds, workspace) |

v1 lowered the attitude D-gain to 0.3 to damp a cruise-time ±11° / ~1 Hz roll-pitch oscillation. Side effect (see EXPERIMENT_LOG.md §24–27): at D=0.3 the velocity-only controller is **underdamped for large transients**, so a perturbation kick tilts the drone past 90° into an unrecoverable inverted freeze — **9.2% of v1 episodes are frozen-inverted crashes, and a real kick recovers ~0% of the time.** Restoring D=1.0 makes the controller **recover from kicks** (0.3m *and* 1.5m kicks return to <0.1m), at a modest cruise-precision cost on corners (~2×). This is the deployability fix for the open real-hardware prereq (attitude-gain retune for velocity-only mode).

## Reproduction (identical to v1 except the gain)

```
python collect_shape_dataset.py \
  --target_steps 1500000 \
  --shapes triangle square pentagon circle \
  --att_d_gain_scale 1.0 \
  --perturb_prob 0.1 --perturb_count 2 --perturb_magnitude 0.3 \
  --direction both \
  --output_folder data_soft_v2
python merge_shape_dataset.py --input_folder data_soft_v2/shape_dataset --output_file data_soft_v2/merged.csv
gzip -c data_soft_v2/merged.csv > data_soft_v2/merged1.5M_soft_v2.csv.gz
```

Seeds are deterministic (`seed = seed_start + episode_idx`), so v2 uses the **same seeds / paths / kick timing as v1** — the only difference is the controller gain. (Verified: v1 and v2 circle seeds match exactly.) Per-episode `shape_dataset/` and any `*.tar.gz` are local-only (`.gitignore`d, regenerable); the merged `.csv.gz` here is the shared artifact (Git LFS).

## Schema

Same as v1: `episode_id, step, tx-x, ty-y, tz-z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, lx, ly, lz, ax, ay, az, reward, done`.
State = path-relative pos_err (tx-x…) + quat + vel + angvel + look-ahead (lx…); action = target velocity (ax,ay,az). The logged action causally explains the transition (velocity-only, position term off).

## Downstream result (EXPERIMENT_LOG.md §27)

Trained with the standard soft recipe (init 300k IQL + DAgger×2), **evaluated at D=1.0** (its deployment gain), 50 seeds × both dirs + untrained star:

| shape | traverse | dist mean | p99 / max |
|---|---|---|---|
| triangle | 100/100 | 0.110 | 0.431 / 0.488 |
| square | 100/100 | 0.124 | 0.443 / 0.507 |
| pentagon | 100/100 | 0.117 | 0.421 / 0.494 |
| circle | 100/100 | 0.016 | 0.065 / 0.141 |
| star (untrained) | 100/100 | 0.148 | 0.433 / 0.518 |
| **TOTAL** | **500/500** | | tight tails, zero blow-ups |

vs v1 (D=0.3): equal-or-better completion (star 98→100/100), clean tails, at a small precision cost (circle 7→16mm). **Use v2 when disturbance recovery / real-hardware deployment matters; use v1 for maximum cruise precision in an undisturbed sim.**
