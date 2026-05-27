export CUDA_VISIBLE_DEVICES=2,3,4,5

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/skt/anaconda3/envs/nuplan/bin/python"

# Training data
TRAIN_SET_PATH="/home/skt/Code/DiffusionPlanner_dataprocess/preprocess_data"
TRAIN_SET_LIST_PATH="/home/skt/Code/Diffusion-Planner-main/cache_list.json"
# Random subset per training run (<=0 means use full list)
TRAIN_SUBSET_SIZE=50000

# Pretrained encoder + staged freeze (0-based: epochs 0..399 frozen, 400+ joint train)
ENCODER_FREEZE_EPOCHS=400
CHECKPOINT_DIR="/home/skt/Code/Diffusion-Planner-main/checkpoints"
ENCODER_INIT_PATH="${CHECKPOINT_DIR}/model.pth"
NORMALIZATION_FILE="/home/skt/Code/Diffusion-Planner-main/normalization.json"

# Optional: resume Bezier training from an existing run directory (uses latest.pth inside).
# Leave empty to start a new run (may sweep BEZIER_DEGREE values).
RESUME_MODEL_PATH=""
# Example:
# RESUME_MODEL_PATH="/home/skt/Code/Diffusion-Planner-main/training_log/diffusion-planner-training/2026-05-20-12:00:00"
###################################

cd /home/skt/Code/Diffusion-Planner-main || exit 1

COMMON_TRAIN_ARGS=(
    --train_set "${TRAIN_SET_PATH}"
    --train_set_list "${TRAIN_SET_LIST_PATH}"
    --normalization_file_path "${NORMALIZATION_FILE}"
    --encoder_init_path "${ENCODER_INIT_PATH}"
    --use_bezier True
    --freeze_encoder True
    --encoder_freeze_epochs "${ENCODER_FREEZE_EPOCHS}"
    --train_subset_size "${TRAIN_SUBSET_SIZE}"
)

run_distributed_train() {
    "${RUN_PYTHON_PATH}" -m torch.distributed.run --nnodes 1 --nproc-per-node 4 --standalone train_predictor.py \
        "${COMMON_TRAIN_ARGS[@]}" \
        "$@"
}

if [[ -n "${RESUME_MODEL_PATH}" ]]; then
    # Resume: continue in the existing run folder from latest.pth (skip BEZIER_DEGREE loop).
    RESUME_RUN_DIR="${RESUME_MODEL_PATH}"
    if [[ -f "${RESUME_RUN_DIR}" ]]; then
        RESUME_RUN_DIR="$(dirname "${RESUME_RUN_DIR}")"
    fi
    LATEST_CKPT="${RESUME_RUN_DIR}/latest.pth"
    RUN_ARGS_JSON="${RESUME_RUN_DIR}/args.json"

    if [[ ! -f "${LATEST_CKPT}" ]]; then
        echo "ERROR: latest.pth not found: ${LATEST_CKPT}"
        exit 1
    fi
    if [[ ! -f "${RUN_ARGS_JSON}" ]]; then
        echo "ERROR: args.json not found: ${RUN_ARGS_JSON}"
        exit 1
    fi

    BEZIER_DEGREE="$("${RUN_PYTHON_PATH}" -c "import json; print(json.load(open('${RUN_ARGS_JSON}'))['bezier_degree'])")"

    echo "=========================================="
    echo "Resume training (skip degree sweep)"
    echo "  run dir : ${RESUME_RUN_DIR}"
    echo "  latest  : ${LATEST_CKPT}"
    echo "  args    : ${RUN_ARGS_JSON}"
    echo "  degree  : ${BEZIER_DEGREE}"
    echo "=========================================="

    run_distributed_train \
        --encoder_args_path "${RUN_ARGS_JSON}" \
        --bezier_degree "${BEZIER_DEGREE}" \
        --resume_model_path "${RESUME_RUN_DIR}"
else
    for BEZIER_DEGREE in 5; do
        ENCODER_ARGS_PATH="${CHECKPOINT_DIR}/args_${BEZIER_DEGREE}.json"

        echo "=========================================="
        echo "New training: args_${BEZIER_DEGREE}.json, bezier_degree=${BEZIER_DEGREE}"
        echo "=========================================="

        run_distributed_train \
            --encoder_args_path "${ENCODER_ARGS_PATH}" \
            --bezier_degree "${BEZIER_DEGREE}"
    done
fi
