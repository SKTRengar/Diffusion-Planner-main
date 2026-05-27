export CUDA_VISIBLE_DEVICES=2,3,4,5
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/skt/Code/Diffusion-Planner-main:$PYTHONPATH"
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
SPLIT="test14-hard-20"  # e.g., "trainval", "mini", "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_reactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################


BRANCH_NAME=test14-hard-20
CKPT_FILE="/home/skt/Code/Diffusion-Planner-main/training_log/Bezier_degree_4_len16_2026-05-27-01:59:09/model_epoch_500_trainloss_16.1550.pth"
# Use args.json from the same training run as the checkpoint (required for Bezier + normalizers)
ARGS_FILE="/home/skt/Code/Diffusion-Planner-main/checkpoints/args_16_bezier4.json"
if [[ ! -f "${ARGS_FILE}" ]]; then
    echo "ERROR: args.json not found next to checkpoint: ${ARGS_FILE}"
    exit 1
fi
SCENARIO_BUILDER="nuplan"

echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner
PLANNER_NAME=${PLANNER}_${FILENAME_WITHOUT_EXTENSION}
DATA_ROOT="/home/skt/Code/nuplan-devkit/nuplan/dataset/data/cache/test"
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_builder.data_root=$DATA_ROOT \
    scenario_filter=$SPLIT \
    experiment_uid=$EXP_NAME/$PLANNER_NAME/$BRANCH_NAME/${PLANNER_NAME}_${BRANCH_NAME}$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=32 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"