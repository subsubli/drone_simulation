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

- **Generator**: `Linear(latent → ch·H)` → reshape `(B, ch, H)` + broadcast-added class embedding →
  ResBlocks → `Conv1d(ch, feat)`. latent 128, ch 128, depth 6.
- **Discriminator**: conv + ResBlocks + global-avg-pool → `Linear(ch, 1)` head + Miyato projection
  `(cproj(c) · pooled).sum`. GroupNorm (not BatchNorm). Every layer spectral-normalized under `--d-reg sn`.
- **Loss**: RpGAN `softplus(D(fake) − D(real))` (D) / `softplus(D(real_g) − D(fake))` (G); R1 on real +
  R2 on fake (`0.5·γ·|∇D|²`). TTUR/dynamic-D hacks exist but are unnecessary once RpGAN is used.
- **Best-checkpoint**: GANs are non-monotonic, so every `--eval-every` steps the EMA generator is scored
  (on-path median + pe_jerk) and the best is saved to `model.pt`.

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

See EXPERIMENT_LOG.md §20/§21/§28 for the downstream augmentation results (headline: the
smoothness-penalized GAN's ultra-precise, cleaner-than-real data yields the most precise augmented
policy in the project).
