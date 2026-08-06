#!/bin/zsh
# Full soft-recipe pipeline for ONE mixed dataset (matches Table 1 soft: initial train + DAgger x2).
# Assumes <MIX>/shape_dataset/ exists (build_mix.py). Fixed DAgger protocol for comparability:
# pass1 = 4 shapes seeds 0-59, pass2 = square+triangle seeds 60-119, both directions.
# Each train's run dir is captured from main.py's own "Log dir:" print (no ls -td race -> parallel-safe).
# Usage:  zsh run_aug_pipeline.sh mix_r1.0_d0.5
set -e
source ~/miniconda3/etc/profile.d/conda.sh
EX=~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
IQL=~/drone_simulation/IQL-PyTorch-main
MIX=$1
KICKS="--slew-max-accel 2.0 --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 --direction both"
TRAIN="--n-steps 300000 --hidden-dim 256 --beta 3.0 --include-lookahead --reward-clip-min -1.0 --eval-period 100000"
stamp() { date '+%H:%M:%S'; }
run_train() {   # $1=csv  $2=tag ; echoes the exact run dir main.py wrote
  ( cd $IQL && conda activate iql && python main.py --csv-file "$1" --log-dir $IQL/runs $=TRAIN ) \
     > $EX/$MIX/train_$2.log 2>&1
  grep -oE 'Log dir: .*' $EX/$MIX/train_$2.log | tail -1 | sed 's/^Log dir: //'
}

echo "===== [$MIX] START $(stamp) ====="
INIT=$(run_train $EX/$MIX/merged.csv init);            echo "[$MIX] init=$INIT $(stamp)"

conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$INIT" \
  --shapes triangle square pentagon circle --seed-start 0 --n-seeds 60 $=KICKS --output_folder $MIX/dagger1
echo "[$MIX] dagger1 done $(stamp)"
python merge_shape_dataset.py --input_folder $MIX/shape_dataset $MIX/dagger1/shape_dataset --output_file $MIX/d1_merged.csv
R1=$(run_train $EX/$MIX/d1_merged.csv d1);             echo "[$MIX] retrain1=$R1 $(stamp)"

conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$R1" \
  --shapes square triangle --seed-start 60 --n-seeds 60 $=KICKS --output_folder $MIX/dagger2
echo "[$MIX] dagger2 done $(stamp)"
python merge_shape_dataset.py --input_folder $MIX/shape_dataset $MIX/dagger1/shape_dataset $MIX/dagger2/shape_dataset --output_file $MIX/d2_merged.csv
FINAL=$(run_train $EX/$MIX/d2_merged.csv d2);          echo "$FINAL" > $EX/$MIX/final_run.txt
echo "===== [$MIX] FINAL=$FINAL  DONE $(stamp) ====="
