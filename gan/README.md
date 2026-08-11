# Trajectory GAN generator (`trajectory_gan.py`)

A conditional GAN drop-in alternative to the diffusion generator: same data pipeline (`load_windows`,
asinh normalization, per-episode CSV output), so `build_mix.py` / `merge_shape_dataset.py` /
`eval_aug.py` / `pe_dist.py` consume its pool unchanged. Maps latent `z` (+ on/off-path class) → an
H=64-step `[state|action]` window via a 1D temporal-conv backbone; the discriminator is a conv +
global-pool critic with a projection term for conditioning. Default recipe is **R3GAN** (RpGAN + R1 + R2).

## Papers this is built on

| Component | Paper | ref |
|---|---|---|
| **R3GAN recipe** (the default: relativistic-pairing loss + R1 + R2, a modern minimal GAN baseline — "R3" = RpGAN + R1 + R2) | Huang, Gokaslan, Kuleshov, Tompkin — *The GAN is dead; long live the GAN! A Modern GAN Baseline* (NeurIPS 2024) | arXiv:2501.05441 |
| **RpGAN — relativistic pairing loss** (`--loss rp`; only D(real)−D(fake) matters) | Jolicoeur-Martineau — *The relativistic discriminator: a key element missing from standard GAN* (ICLR 2019) | arXiv:1807.00734 |
| **R1 gradient penalty** (`--r1-gamma`, penalize \|∇_real D\|²) | Mescheder, Geiger, Nowozin — *Which Training Methods for GANs do actually Converge?* (ICML 2018) | arXiv:1801.04406 |
| **R1 popularized / EMA-G / modern GAN practice** | Karras et al. — *Analyzing and Improving the Image Quality of StyleGAN* (StyleGAN2, CVPR 2020) | arXiv:1912.04958 |
| **Spectral normalization** (`--d-reg sn`, the alternative Lipschitz control) | Miyato, Kataoka, Koyama, Yoshida — *Spectral Normalization for GANs* (ICLR 2018) | arXiv:1802.05957 |
| **Projection discriminator** (class conditioning term ⟨embed(c), pooled_features⟩) | Miyato, Koyama — *cGANs with Projection Discriminator* (ICLR 2018) | arXiv:1802.05637 |
| **WGAN-GP** (the `--d-reg gp` alternative, gradient penalty) | Gulrajani, Ahmed, Arjovsky, Dumoulin, Courville — *Improved Training of Wasserstein GANs* (NeurIPS 2017) | arXiv:1704.00028 |

## Architecture

`feat = 19`, `H = 64`, `latent = 128`, `ch = 128`, `depth = 6`. Shared backbone **`ResBlocks`**: `depth`
blocks of `Conv1d(ch→ch, k=5, pad=2·d, dilation=d) → GroupNorm(8, ch) → SiLU`, applied residually
(`h = h + blk(h)`) — the *same* 1D temporal-conv stack as the diffusion denoiser (`sn=True` spectral-
normalizes its convs, D-side only). ~**G 1.56M / D 0.51M** params.

- **Generator** `z:(B,latent) [+ c] → (B,H,feat)`: `fc = Linear(latent → ch·H)` → `view (B, ch, H)`;
  class `Embedding(n_classes→ch)` broadcast-added `+ cemb(c)[:,:,None]` (exactly like the diffusion
  timestep embedding); `ResBlocks` (no spectral norm); `out = Conv1d(ch→feat, k=5, pad=2)` → transpose.
- **Discriminator** `x:(B,H,feat) [+ c] → (B,1)`: `inp = Conv1d(feat→ch, k=5, pad=2)` → `ResBlocks` →
  **global average pool over time** `h.mean(dim=2) → (B, ch)` → `head = Linear(ch→1)`; **Miyato
  projection** conditioning adds `+ (cproj(c) · h).sum(dim=1)` where `cproj = Embedding(n_classes→ch)`.
  GroupNorm (not BatchNorm) so the WGAN-GP penalty stays well-defined; every layer spectral-normalized
  under `--d-reg sn`.
- **Loss**: RpGAN — `lossD = softplus(D(fake) − D(real)).mean()`, `lossG = softplus(D(real_g) − D(fake)).mean()`
  (relativistic: only the real−fake gap matters). R1 on real + R2 on fake, `lossD += 0.5·γ·|∇D|²` each
  (`--r1-gamma --r2-gamma`, R3GAN). (`--loss hinge` + `--d-reg sn/gp` are the alternatives.) TTUR /
  dynamic-D balance hacks exist but are unnecessary once RpGAN is used.
- **Best-checkpoint**: GANs are non-monotonic, so every `--eval-every` steps the **EMA generator**
  (`--ema 0.999`) is scored (on-path pos_err median + pe_jerk) and only the best is saved to `model.pt`.
  Cosine LR down to `--lr-min-frac` of the base LR.

## Project-specific additions (NOT from a paper — from this project's experiments)

From EXPERIMENT_LOG.md §20/§20b (not any single paper):

- **asinh(pos_err / 0.05) normalization**, **on/off-path class conditioning**, **pos_err↔vel consistency
  loss** — shared with the diffusion generator (see `../diffusion/README.md`).
- **`--lambda-smooth`** — a direct 2nd-difference penalty on the generated pos_err channels, targeting
  `pe_jerk` (roughness) itself. This is the key lever that pushed the GAN through its roughness floor
  (λ=10 → pe_jerk ≈ diffusion, §20b); it is not a standard GAN term.

## Usage

```bash
# train the class-conditional smoothness-penalized GAN (the recipe used for gen_gan_cc / _v2)
python trajectory_gan.py --csv <merged.csv> \
  --steps 50000 --pe-asinh 0.05 --lambda-cons 0.1 --lambda-smooth 10.0 --lr-min-frac 0.1 \
  --r1-gamma 1.0 --r2-gamma 1.0 --class-cond --offpath-batch-frac 0.5 --out gen_gan_cc

# resample a pool from the best checkpoint (no retraining)
python trajectory_gan.py --load gen_gan_cc/model.pt \
  --n-gen 24000 --gen-offpath-frac -1.0 --out gen_pool_gan
```

> ⚠️ **`--offpath-batch-frac 0.5` on low-off-path data (e.g. soft v2, 0.3% off-path):** the
> class-conditional **off-path branch collapses**. It trains on only ~160 windows there, so even forcing
> `--gen-offpath-frac 1.0` at generation produces no real excursions (0% > 0.2m; the GAN outputs a
> razor-thin ~0.13m band) — see **EXPERIMENT_LOG.md §28**. `--offpath-batch-frac` oversamples but cannot
> manufacture diversity from ~160 windows (cf. §20c: even 3228 collapse). This is expected, not a bug:
> on such data the pool is on-path-only and recovery coverage must come from elsewhere (a heavy-kick
> dataset like hard v2 §29, and DAgger). The knob matters on high-off-path data (v1: 7%; hard: ~32%).

See EXPERIMENT_LOG.md §20/§21/§28 for the downstream augmentation results (headline: the
smoothness-penalized GAN's ultra-precise, cleaner-than-real data yields the most precise augmented
policy in the project).
