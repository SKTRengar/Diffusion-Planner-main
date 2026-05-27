export CUDA_VISIBLE_DEVICES=0,1,2,6
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/home/skt/Code/nuplan-devkit"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/home/skt/Code/nuplan-devkit/nuplan/dataset"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/home/skt/Code/nuplan-devkit/nuplan/dataset/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="/home/skt/Code/nuplan-devkit/nuplan/exp" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "trainval"       (dataset/splits/trainval)
#   - "mini"           (small subset for quick testing)
#   - "val14"
#   - "test14-random"
#   - "test14-hard"
SPLIT="test14-hard"  # e.g., "trainval", "mini", "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_reactive_agents"  # e.g., "closed_loop_nonreactive_agents"

# Number of repeated runs (same hydra args except experiment_uid timestamp)
NUM_RUNS=50

# Aggregated metrics CSV (written under Diffusion-Planner-main)
RESULT_CSV="/home/skt/Code/Diffusion-Planner-main/sim_multi_diffusion_planner_10runs_metrics.csv"
###################################


BRANCH_NAME=test14-hard
ARGS_FILE="/home/skt/Code/Diffusion-Planner/checkpoints/args.json"
CKPT_FILE="/home/skt/Code/Diffusion-Planner-main/training_log/diffusion-planner-training/2026-04-22-11:57:12/model_epoch_460_trainloss_0.0488.pth"
SCENARIO_BUILDER="nuplan"

echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner
PLANNER_NAME=${PLANNER}_${FILENAME_WITHOUT_EXTENSION}
DATA_ROOT="/home/skt/Code/nuplan-devkit/nuplan/dataset/data/cache/test"

# Optional prefix for experiment_uid (leave unset to omit)
# export EXP_NAME="my_exp"

write_csv_header() {
    echo "run_index,experiment_uid,总得分,drivable_area_compliance,driving_direction_compliance,ego_is_comfortable,ego_is_making_progress,ego_progress_along_expert_route,no_ego_at_fault_collisions,speed_limit_compliance,time_to_collision_within_bound" >"$RESULT_CSV"
}

append_metrics_row() {
    local run_idx="$1"
    local exp_uid="$2"
    local agg_dir="${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${exp_uid}/aggregator_metric"
    python3 <<PY
import csv
import glob
import os
import sys

import pandas as pd

run_idx = int("${run_idx}")
exp_uid = """${exp_uid}"""
agg_dir = """${agg_dir}"""
csv_path = """${RESULT_CSV}"""

pattern = os.path.join(agg_dir, "*weighted_average_metrics*.parquet")
paths = glob.glob(pattern)
if not paths:
    print(f"ERROR: no parquet under {pattern}", file=sys.stderr)
    sys.exit(1)
parquet_path = max(paths, key=os.path.getmtime)
df = pd.read_parquet(parquet_path)
row = df[df["scenario"].astype(str) == "final_score"]
if row.empty:
    row = df.loc[[df["num_scenarios"].idxmax()]]
r = row.iloc[0]
cols = [
    "score",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "ego_is_comfortable",
    "ego_is_making_progress",
    "ego_progress_along_expert_route",
    "no_ego_at_fault_collisions",
    "speed_limit_compliance",
    "time_to_collision_within_bound",
]
vals = [r[c] for c in cols]

def fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return f"{float(v):.4f}"

with open(csv_path, "a", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [run_idx, exp_uid, fmt(vals[0])]
        + [fmt(v) for v in vals[1:]]
    )
print(f"Run {run_idx}: wrote row from {parquet_path}")
PY
}

write_csv_header

for ((i=1; i<=NUM_RUNS; i++)); do
    RUN_TS=$(date "+%Y-%m-%d-%H-%M-%S")
    # Match nuplan layout: output_dir = ${NUPLAN_EXP_ROOT}/exp/simulation/${CHALLENGE}/${experiment_uid}
    if [[ -n "${EXP_NAME:-}" ]]; then
        EXPERIMENT_UID="${EXP_NAME}/${PLANNER_NAME}/${BRANCH_NAME}/${PLANNER_NAME}_${BRANCH_NAME}${RUN_TS}"
    else
        EXPERIMENT_UID="${PLANNER_NAME}/${BRANCH_NAME}/${PLANNER_NAME}_${BRANCH_NAME}${RUN_TS}"
    fi

    echo "========== Run $i / $NUM_RUNS | experiment_uid=$EXPERIMENT_UID =========="

    python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
        +simulation="$CHALLENGE" \
        planner="$PLANNER" \
        planner.diffusion_planner.config.args_file="$ARGS_FILE" \
        planner.diffusion_planner.ckpt_path="$CKPT_FILE" \
        scenario_builder="$SCENARIO_BUILDER" \
        scenario_builder.data_root="$DATA_ROOT" \
        scenario_filter="$SPLIT" \
        experiment_uid="$EXPERIMENT_UID" \
        verbose=true \
        worker=ray_distributed \
        worker.threads_per_node=128 \
        distributed_mode='SINGLE_NODE' \
        number_of_gpus_allocated_per_simulation=0.15 \
        enable_simulation_progress_bar=true \
        hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"

    append_metrics_row "$i" "$EXPERIMENT_UID"
done

echo "All runs finished. Metrics CSV: $RESULT_CSV"
