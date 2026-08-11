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

## Architecture (`TemporalDenoiser`, ε-prediction UNet-free 1D conv net)

Feature dim `feat = 19` (state 16 + action 3), window `H = 64`, width `ch = 128`, `depth = 6`.
Forward `(x:(B,H,feat), t:(B,), c:(B,) | None) → ε̂:(B,H,feat)`:

1. `x.transpose(1,2)` → `(B, feat, H)`, then **input conv** `Conv1d(feat→ch, k=5, pad=2)` → `(B, ch, H)`.
2. **Timestep embedding**: `temb = Linear(1→ch)` applied to `t/1000`, broadcast-added over time → `+ temb[:,:,None]`.
3. **Class embedding** (conditional only): `cemb = Embedding(n_classes→ch)`, `+ cemb(c)[:,:,None]` — added the *same* broadcast way as `temb`. `n_classes=0` ⇒ byte-identical unconditional (pre-class checkpoints load unchanged).
4. **Residual stack**: `depth=6` blocks, each `Conv1d(ch→ch, k=5, pad=2·d, dilation=d) → GroupNorm(8, ch) → SiLU`, applied residually (`h = h + blk(h)`). `d=1` (undilated, default; RF ≈ 33 < 64) or `d = 1,2,4,8,16` with `--dilated` (RF > 64).
5. **Output conv** `Conv1d(ch→feat, k=5, pad=2)` → `transpose` → `(B, H, feat)`.

No down/up-sampling — constant `H` throughout (a flat 1D-conv denoiser, the "lite" in Diffuser-lite).

**Diffusion process**: `T=1000`, `betas = linspace(1e-4, 0.02, T)`, `ᾱ = cumprod(1−betas)`; ε-prediction
MSE loss (optionally min-SNR-weighted). Weights kept as an **EMA shadow** (`--ema 0.999`); sampling uses
the EMA weights. **Sampling**: DDPM ancestral (inject noise each step) or DDIM (`--sampler ddim`,
subsampled, `--eta` stochasticity); classifier-free guidance blends `ε = ε_uncond + w·(ε_cond − ε_uncond)`
(`--cfg-weight`). Cosine LR with warmup.

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

---

## Alternative: transition-level diffusion (`transition_diffusion.py`, SynthER-style)

A separate, simpler generator that diffuses **single transitions** instead of whole trajectory windows.
It learns the joint `(s, a, r, s')` distribution and samples synthetic transitions, written as **2-row
mini-episodes** (row0 = `s,a,r,done=False`; row1 = `s',0,0,done=True`) so the same
`merge_shape_dataset.py` / `drone_dataset.py` pipeline reads each back as one `(s,a,r,s')` transition.

| Component | Paper | ref |
|---|---|---|
| **Transition-level diffusion for RL** (diffuse `(s,a,r,s')`, upsample the replay buffer) | Lu, Ball, Parker-Holder, Osborne, Roberts — *Synthetic Experience Replay* (SynthER, NeurIPS 2023) | arXiv:2303.06614 |
| **DDPM** (same ε/DDPM core as above) | Ho, Jain, Abbeel (NeurIPS 2020) | arXiv:2006.11239 |

**Architecture**: a plain **MLP denoiser** (not the 1D-conv net) on the flat `DIM = 36` transition vector
`[s(16) | a(3) | r(1) | s'(16)]` — `Linear(36+1 → 512) → SiLU → Linear(512→512) → SiLU → Linear(512→512)
→ SiLU → Linear(512→36)`, the `+1` being the timestep `t/1000`. DDPM T=1000.

```bash
python transition_diffusion.py --csv <merged.csv> --steps 30000 --n-gen 200000 --per-file 20000 --out gen_diffusion
```

**Why the trajectory generator (top of this file) is the one actually used**: transition-level sampling
has no temporal structure, so it can emit `(s,a,r,s')` tuples whose action doesn't causally explain the
state change (cf. the memory note "logged action must causally explain the transition"). The trajectory
diffusion generates whole coherent episodes (action |mean| ≈ 13.9, not 0), which is why it — not this
SynthER-style variant — was carried through the augmentation experiments (§15/§18/§28).
