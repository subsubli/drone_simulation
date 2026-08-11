#!/bin/zsh
# One mix at D=1.0: init train -> INIT eval -> DAgger x2 -> FINAL eval. $1 = mix dir name (under EX).
# Expects $EX/$1/shape_dataset (built by build_mix) and $EX/$1/merged.csv (merged) to exist.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
EX=~/drone_simulation/gym_pybullet_drones/gym_pybullet_drones/examples
IQL=~/drone_simulation/IQL-PyTorch-main
DIR=$EX/$1
ATT=1.0
KICKS="--slew-max-accel 2.0 --perturb_prob 1.0 --perturb_count 8 --perturb_magnitude 1.5 --direction both --att_d_gain_scale $ATT"
TRAIN="--n-steps 300000 --hidden-dim 256 --beta 3.0 --include-lookahead --reward-clip-min -1.0 --eval-period 100000"
stamp() { date '+%H:%M:%S'; }
run_train() { ( cd $IQL && conda activate iql && python main.py --csv-file "$1" --log-dir $IQL/runs $=TRAIN ) > $DIR/train_$2.log 2>&1
  grep -oE 'Log dir: .*' $DIR/train_$2.log | tail -1 | sed 's/^Log dir: //'; }

echo "===== [$1] START $(stamp) ====="
INIT=$(run_train $DIR/merged.csv init); echo "[$1] init=$INIT $(stamp)"
conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python eval_stuck.py --run-dir "$INIT" --att-d-gain-scale $ATT --seeds 500 501 502 --label "${1}_INIT_stuck" > $DIR/eval_INIT_stuck.txt 2>/dev/null || true
KMP_DUPLICATE_LIB_OK=TRUE python eval_aug.py --run-dir "$INIT" --att-d-gain-scale $ATT --seeds $(seq 500 509) --shapes triangle square pentagon circle --label "${1}_INIT" > $DIR/eval_INIT_aug.txt 2>/dev/null || true
echo "[$1] INIT eval done $(stamp)"

KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$INIT" --shapes triangle square pentagon circle --seed-start 0 --n-seeds 60 $=KICKS --output_folder $DIR/dagger1
python merge_shape_dataset.py --input_folder $DIR/shape_dataset $DIR/dagger1/shape_dataset --output_file $DIR/d1_merged.csv
R1=$(run_train $DIR/d1_merged.csv d1); echo "[$1] retrain1=$R1 $(stamp)"

conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python collect_dagger.py --run-dir "$R1" --shapes square triangle --seed-start 60 --n-seeds 60 $=KICKS --output_folder $DIR/dagger2
python merge_shape_dataset.py --input_folder $DIR/shape_dataset $DIR/dagger1/shape_dataset $DIR/dagger2/shape_dataset --output_file $DIR/d2_merged.csv
FINAL=$(run_train $DIR/d2_merged.csv d2); echo "$FINAL" > $DIR/final_run.txt; echo "[$1] FINAL=$FINAL $(stamp)"

conda activate drones; cd $EX
KMP_DUPLICATE_LIB_OK=TRUE python eval_aug.py --run-dir "$FINAL" --att-d-gain-scale $ATT --seeds $(seq 500 549) --shapes triangle square pentagon circle star --label "${1}_FINAL" > $DIR/eval_FINAL_aug.txt 2>/dev/null || true
echo "===== [$1] ALL DONE $(stamp) FINAL=$FINAL ====="
