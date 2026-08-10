#!/bin/zsh
# soft v2 (att_d_gain_scale=1.0) full pipeline: init train -> eval INIT -> DAgger x2 -> eval FINAL.
# All rollouts (DAgger collection + eval) at D=1.0 to match the v2 collection gain.
# Mirrors run_aug_pipeline.sh but for the single all-real v2 dataset, and runs both evals.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
EX=~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
IQL=~/drone_simulation/IQL-PyTorch-main
DIR=$EX/data_soft_v2
ATT=1.0
KICKS="--slew-max-accel 2.0 --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 --direction both --att_d_gain_scale $ATT"
TRAIN="--n-steps 300000 --hidden-dim 256 --beta 3.0 --include-lookahead --reward-clip-min -1.0 --eval-period 100000"
stamp() { date '+%H:%M:%S'; }
run_train() {  # $1=csv $2=tag ; echoes the run dir main.py wrote
  ( cd $IQL && conda activate iql && python main.py --csv-file "$1" --log-dir $IQL/runs $=TRAIN ) \
     > $DIR/train_$2.log 2>&1
  grep -oE 'Log dir: .*' $DIR/train_$2.log | tail -1 | sed 's/^Log dir: //'
}

echo "===== [soft_v2] START $(stamp) ====="

# --- init train on all-real v2 ---
INIT=$(run_train $DIR/merged.csv init); echo "[v2] init=$INIT $(stamp)"

# --- INIT-only eval (D=1.0): stuck verdict + 10-seed aug (x/20) ---
conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python eval_stuck.py --run-dir "$INIT" --att-d-gain-scale $ATT \
  --seeds 500 501 502 --label "v2_INIT_stuck" > $DIR/eval_INIT_stuck.txt 2>/dev/null || true
KMP_DUPLICATE_LIB_OK=TRUE python eval_aug.py --run-dir "$INIT" --att-d-gain-scale $ATT \
  --seeds $(seq 500 509) --shapes triangle square pentagon circle \
  --label "v2_INIT" > $DIR/eval_INIT_aug.txt 2>/dev/null || true
echo "[v2] INIT eval done $(stamp)"

# --- DAgger 1 (D=1.0) ---
KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$INIT" \
  --shapes triangle square pentagon circle --seed-start 0 --n-seeds 60 $=KICKS --output_folder $DIR/dagger1
python merge_shape_dataset.py --input_folder $DIR/shape_dataset $DIR/dagger1/shape_dataset --output_file $DIR/d1_merged.csv
R1=$(run_train $DIR/d1_merged.csv d1); echo "[v2] retrain1=$R1 $(stamp)"

# --- DAgger 2 (D=1.0) ---
conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$R1" \
  --shapes square triangle --seed-start 60 --n-seeds 60 $=KICKS --output_folder $DIR/dagger2
python merge_shape_dataset.py --input_folder $DIR/shape_dataset $DIR/dagger1/shape_dataset $DIR/dagger2/shape_dataset --output_file $DIR/d2_merged.csv
FINAL=$(run_train $DIR/d2_merged.csv d2); echo "$FINAL" > $DIR/final_run.txt; echo "[v2] FINAL=$FINAL $(stamp)"

# --- FINAL eval (D=1.0): 50-seed both dirs + untrained star (§21 format) ---
conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python eval_aug.py --run-dir "$FINAL" --att-d-gain-scale $ATT \
  --seeds $(seq 500 549) --shapes triangle square pentagon circle star \
  --label "v2_FINAL" > $DIR/eval_FINAL_aug.txt 2>/dev/null || true
echo "===== [soft_v2] ALL DONE $(stamp) FINAL=$FINAL ====="
