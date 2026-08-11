# Trajectory Diffusion generator (`trajectory_diffusion.py`)

A DDPM that generates whole H=64-step drone trajectory windows `[state(16) | action(3)]` per step, as
per-episode CSVs consumable by the same `merge_shape_dataset.py` / `drone_dataset` / `eval_aug.py`
pipeline as the real data. Used to augment the offline-RL (IQL) shape-tracing dataset. Class-conditional
on **on-path vs off-path** windows (0.2m `|pos_err|` threshold) so the two physically-distinct modes are
learned and sampled separately.

## Papers this is built on

| Component | Paper | ref |
|---|---|---|
| **Trajectory diffusion framework** ("Diffuser-lite": diffuse whole `[s\|a]` windows with a 1D temporal conv net) | Janner, Du, Tenenbaum, Levine — *Planning with Diffusion for Flexible Behavior Synthesis* (ICML 2022) | arXiv:2205.09991 |
| **DDPM** (ε-prediction training, ancestral sampling, EMA of weights) | Ho, Jain, Abbeel — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) | arXiv:2006.11239 |
| **Classifier-free guidance** (cond-dropout → null class at train, `cfg_weight` blend at sampling) | Ho, Salimans — *Classifier-Free Diffusion Guidance* (2022) | arXiv:2207.12598 |
| **Conditional trajectory generation for decision-making** (conditioning trajectory diffusion, the on/off-path class idea's lineage) | Ajay, Du, Gupta, Tenenbaum, Jaakkola, Agrawal — *Is Conditional Generative Modeling all you need for Decision-Making?* (Decision Diffuser, ICLR 2023) | arXiv:2211.15657 |
| **Min-SNR loss weighting** (`--min-snr-gamma`, weight ε-MSE by min(SNR,γ)/SNR) | Hang, Gu, Wang, Wu, Chen, Bao, Guo — *Efficient Diffusion Training via Min-SNR Weighting Strategy* (ICCV 2023) | arXiv:2303.09556 |
| **DDIM sampler** (`--sampler ddim`, deterministic/subsampled reverse) | Song, Meng, Ermon — *Denoising Diffusion Implicit Models* (ICLR 2021) | arXiv:2010.02502 |

## Architecture

`TemporalDenoiser`: 1D conv over the H time-axis (sees sequence structure, not a flattened blob); the
timestep embedding and the class embedding are added the same broadcast way; `ch=128`, `depth=6`
ResBlocks. ε-prediction with T=1000 (DDPM) or subsampled DDIM. Classifier-free guidance via a NULL
class. `n_classes=0` reproduces byte-identical unconditional behaviour so pre-class checkpoints load
unchanged.

## Project-specific additions (NOT from a paper — from this project's experiments)

These came out of the experiment log (EXPERIMENT_LOG.md §13/§14/§17/§23), not any single paper:

- **asinh(pos_err / 0.05) normalization** — linear near 0, log-compresses the recovery tail; fixed the
  pos_err jitter that plain standardization left (§13).
- **On/off-path class conditioning** (`--class-cond`, `--offpath-batch-frac`, `--gen-offpath-frac`) —
  split at `|pos_err|>0.2m` (matches the dataset's off-path fraction), with class-balanced batch
  oversampling because the natural off-path ratio starves the branch (§17).
- **pos_err ↔ vel consistency loss** (`--lambda-cons`) — penalizes physical inconsistency between the
  pos_err channels and the velocity channels.
- **x0 pos_err penalty** (`--lambda-x0pe`) — direct penalty on the predicted x0 pos_err to tighten the
  bulk (§23; a data-level win that did not reach the policy, §23b).

## Usage

```bash
# train the class-conditional generator (the "cc" recipe used for gen_traj_cc / _v2)
python trajectory_diffusion.py --csv <merged.csv> \
  --steps 50000 --pe-asinh 0.05 --lambda-cons 0.1 \
  --class-cond --offpath-batch-frac 0.5 --cond-dropout 0.1 --cfg-weight 1.5 \
  --gen-batch 200 --n-gen 500 --out gen_traj_cc

# resample a pool from a saved model (no retraining); --gen-offpath-frac -1 = natural ratio
python trajectory_diffusion.py --load gen_traj_cc/model.pt \
  --n-gen 24000 --gen-offpath-frac -1.0 --cfg-weight 1.5 --gen-batch 200 --out gen_pool_cc
```

See `pe_dist.py` for the Table-14-style `|pos_err|` distribution of a generated pool, and
EXPERIMENT_LOG.md §15/§18/§23/§28 for the downstream augmentation results.
