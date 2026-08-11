# Hard dataset v2 — heavy-kick recovery data, **attitude D-gain restored to 1.0**

`merged1.5M_hard_v2.csv.gz` — 1,502,262 rows / 100 Hz / pure-pursuit + DSLPID velocity-only expert.
The hard-kick collection (`--perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5`) re-run at **att_d_gain_scale = 1.0**. Successor to the original hard dataset (`data/merged1.5M.csv.gz`), which used the same kicks at att_d_gain_scale 0.3.

## Why this dataset exists — the recovery-coverage fix

The original hard dataset put 6×1.5m kicks in every episode to create "off-path → recover" samples, but at att_d_gain_scale 0.3 the velocity-only controller is underdamped for large transients (EXPERIMENT_LOG.md §27): a hard kick tilts the drone past 90° into an unrecoverable inverted freeze. Result: **~60% of rows are inverted crashes, not recoveries** — the "recovery" data is mostly a drone stuck upside-down 3–5m off-path. Restoring **D=1.0 makes the drone actually recover** from those same kicks. Measured (same seeds, only the gain changed):

| | original hard (D=0.3) | **hard v2 (D=1.0)** | soft v2 (gentle kicks) |
|---|---|---|---|
| flipped (tilt>90°) | ~60% | **1.1%** | 0.3% |
| off-path (\|pos_err\|>0.2m) | ~79% (mostly frozen) | **32% (genuine recovery)** | 0.3% |

So hard v2 is **32% genuine off-path→recovery transitions with almost no crashes** — the richest recovery-coverage dataset in the project. soft v2 has too little off-path (0.3%, the gentle kicks barely perturb at D=1.0); the original hard has lots but frozen/dead. hard v2 is the one that actually fills the coverage hole (§16/§19) at the data level.

## Reproduction (identical to original hard except the gain)

```
python collect_shape_dataset.py \
  --target_steps 1500000 --shapes triangle square pentagon circle \
  --att_d_gain_scale 1.0 \
  --perturb_prob 1.0 --perturb_count 6 --perturb_magnitude 1.5 \
  --direction both --output_folder data_hard_v2
python merge_shape_dataset.py --input_folder data_hard_v2/shape_dataset --output_file data_hard_v2/merged.csv
gzip -c data_hard_v2/merged.csv > data_hard_v2/merged1.5M_hard_v2.csv.gz
```

Seeds are deterministic (`seed = seed_start + episode_idx`), so hard v2 uses the **same seeds / paths / kick timing** as the original hard dataset — only the controller gain differs. Per-episode `shape_dataset/` and `*.tar.gz` are local-only (`.gitignore`d, regenerable); the merged `.csv.gz` here is the shared artifact (Git LFS).

## Schema

Same 22 columns as soft/hard: `episode_id, step, tx-x, ty-y, tz-z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz, lx, ly, lz, ax, ay, az, reward, done`. State = path-relative pos_err + quat + vel + angvel + look-ahead; action = target velocity. Velocity-only so the logged action causally explains the transition.

## Downstream

Trained with the standard recipe (init 300k IQL + DAgger×2) at D=1.0; INIT-only and FINAL results in EXPERIMENT_LOG.md §29. The key question this dataset answers: does having genuine recovery in the *initial* data reduce the pre-DAgger (init-only) divergence that all other datasets (soft/diffusion/GAN) suffer?
