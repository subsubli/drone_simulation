#!/bin/zsh
# Autonomous: retrain diffusion cc + GAN cc on soft v2, gen 24k pools, build 6 mixes (3 ratios x 2 gens),
# run each through the D=1.0 pipeline (init + DAgger x2 + INIT/FINAL eval). Fully unattended.
source ~/miniconda3/etc/profile.d/conda.sh
EX=~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
DIFF=~/drone_simulation/diffusion
GAN=~/drone_simulation/gan
V2CSV=$EX/data_soft_v2/merged.csv
stamp(){ date '+%m-%d %H:%M:%S'; }
say(){ echo "[genv2 $(stamp)] $*"; }

say "START"

# ---------- Phase 1: train diffusion cc on v2 ----------
cd $DIFF; conda activate iql
say "train diffusion cc v2 ..."
KMP_DUPLICATE_LIB_OK=TRUE python trajectory_diffusion.py --csv $V2CSV \
  --steps 50000 --pe-asinh 0.05 --lambda-cons 0.1 \
  --class-cond --offpath-batch-frac 0.5 --cond-dropout 0.1 --cfg-weight 1.5 \
  --gen-batch 200 --n-gen 500 --out gen_traj_cc_v2 > $DIFF/train_cc_v2.log 2>&1
say "diffusion cc v2 trained -> gen_traj_cc_v2/model.pt"

# ---------- Phase 2: resample 24k diffusion pool ----------
say "gen diffusion pool (24k) ..."
KMP_DUPLICATE_LIB_OK=TRUE python trajectory_diffusion.py --load gen_traj_cc_v2/model.pt \
  --n-gen 24000 --gen-offpath-frac -1.0 --cfg-weight 1.5 --gen-batch 200 \
  --out gen_pool_cc_v2 > $DIFF/pool_cc_v2.log 2>&1
say "diffusion pool: $(ls $DIFF/gen_pool_cc_v2/shape_dataset/*.csv 2>/dev/null | wc -l | tr -d ' ') windows"

# ---------- Phase 3: train GAN cc on v2 ----------
cd $GAN
say "train GAN cc v2 (lambda_smooth=10) ..."
KMP_DUPLICATE_LIB_OK=TRUE python trajectory_gan.py --csv $V2CSV \
  --steps 50000 --pe-asinh 0.05 --lambda-cons 0.1 --lambda-smooth 10.0 --lr-min-frac 0.1 \
  --r1-gamma 1.0 --r2-gamma 1.0 \
  --class-cond --offpath-batch-frac 0.5 --out gen_gan_cc_v2 > $GAN/train_gan_v2.log 2>&1
say "GAN cc v2 trained -> gen_gan_cc_v2/model.pt"

# ---------- Phase 4: resample 24k GAN pool ----------
say "gen GAN pool (24k) ..."
KMP_DUPLICATE_LIB_OK=TRUE python trajectory_gan.py --load gen_gan_cc_v2/model.pt \
  --n-gen 24000 --gen-offpath-frac -1.0 --out gen_pool_gan_v2 > $GAN/pool_gan_v2.log 2>&1
say "GAN pool: $(ls $GAN/gen_pool_gan_v2/shape_dataset/*.csv 2>/dev/null | wc -l | tr -d ' ') windows"

# ---------- Phase 5: build 6 mixes + run pipelines (2 lanes) ----------
cd $EX; conda activate drones
CCP=../../../diffusion/gen_pool_cc_v2/shape_dataset
GANP=../../../gan/gen_pool_gan_v2/shape_dataset
build(){ # name real gen diffdir
  python build_mix.py --real-rows $2 --diff-rows $3 --real-dir data_soft_v2/shape_dataset --diff-dir $4 --out $1
  python merge_shape_dataset.py --input_folder $1/shape_dataset --output_file $1/merged.csv
}
say "building 6 mixes ..."
build mix_v2_r1.0_cc0.5   1000000  500000 $CCP
build mix_v2_r0.5_cc1.0    500000 1000000 $CCP
build mix_v2_cc1.5              0 1500000 $CCP
build mix_v2_r1.0_gan0.5  1000000  500000 $GANP
build mix_v2_r0.5_gan1.0   500000 1000000 $GANP
build mix_v2_gan1.5             0 1500000 $GANP
say "mixes built; launching 2 lanes"

laneA(){ for m in mix_v2_r1.0_cc0.5 mix_v2_r0.5_cc1.0 mix_v2_cc1.5; do
  say "laneA pipeline $m"; zsh $EX/run_mix_v2_pipeline.sh $m >> $EX/$m/pipeline.log 2>&1; say "laneA done $m"; done }
laneB(){ for m in mix_v2_r1.0_gan0.5 mix_v2_r0.5_gan1.0 mix_v2_gan1.5; do
  say "laneB pipeline $m"; zsh $EX/run_mix_v2_pipeline.sh $m >> $EX/$m/pipeline.log 2>&1; say "laneB done $m"; done }
laneA & A=$!
laneB & B=$!
wait $A; wait $B
say "ALL 6 PIPELINES DONE"
say "GENV2 COMPLETE"
