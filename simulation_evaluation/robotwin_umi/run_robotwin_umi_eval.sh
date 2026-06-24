#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/post_training/lerobot}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
POLICY_PATH="${POLICY_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/simulation_evaluation/robotwin_umi/outputs/$(date +%Y%m%d_%H%M%S)_vista_robotwin_umi}"
GPU_LIST="${GPU_LIST:-0}"
TASK_FILE="${TASK_FILE:-${SCRIPT_DIR}/tasks/smoke_tasks.txt}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_umi_new}"
TEST_NUM="${TEST_NUM:-1}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-false}"
SKIP_GET_OBS_WITHIN_REPLAN="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
FAIL_FAST="${FAIL_FAST:-false}"
RUN_ID="${RUN_ID:-vista_robotwin_umi_$(date +%Y%m%d_%H%M%S)}"
VISTA_RPC_BASE_PORT="${VISTA_RPC_BASE_PORT:-18150}"
IMAGE_CHANNEL_ORDER="${IMAGE_CHANNEL_ORDER:-rgb}"
ENABLE_FISHEYE="${ENABLE_FISHEYE:-true}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
INSTRUCTION_SOURCE="${INSTRUCTION_SOURCE:-generated_template}"
CKPT_SETTING="${CKPT_SETTING:-pretrained_model}"
RENDER_FREQ="${RENDER_FREQ:-0}"
CLIENT_PYTHON="${CLIENT_PYTHON:-python3}"
POLICY_PYTHON="${POLICY_PYTHON:-python3}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-${OUTPUT_ROOT}/summary}"

if [[ -z "${ROBOTWIN_ROOT}" ]]; then
  echo "missing ROBOTWIN_ROOT; set it to your RoboTwin checkout" >&2
  exit 2
fi
if [[ -z "${POLICY_PATH}" ]]; then
  echo "missing POLICY_PATH; set it to a VISTA pretrained_model checkpoint directory" >&2
  exit 2
fi
if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "missing policy config: ${POLICY_PATH}/config.json" >&2
  exit 2
fi
if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "missing RoboTwin root: ${ROBOTWIN_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${TASK_FILE}" ]]; then
  echo "missing task file: ${TASK_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/rpc_shards" "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/rpc_results"
mapfile -t TASKS < <(sed 's/#.*$//' "${TASK_FILE}" | awk 'NF {print $1}')
GPU_LIST="${GPU_LIST//,/ }"
read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "TASK_FILE has no tasks: ${TASK_FILE}" >&2
  exit 2
fi
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 2
fi

SHARD_DIR="${OUTPUT_ROOT}/rpc_shards/${RUN_ID}"
RESULT_ROOT="${OUTPUT_ROOT}/rpc_results/${RUN_ID}"
mkdir -p "${SHARD_DIR}" "${RESULT_ROOT}"
printf '%s\n' "${TASKS[@]}" > "${SHARD_DIR}/all_tasks.txt"
for gpu in "${GPUS[@]}"; do
  : > "${SHARD_DIR}/gpu${gpu}.txt"
done
idx=0
for task in "${TASKS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  printf '%s\n' "${task}" >> "${SHARD_DIR}/gpu${gpu}.txt"
  idx=$((idx + 1))
done

cat > "${RESULT_ROOT}/run_meta.txt" <<EOF
run_id=${RUN_ID}
project_root=${PROJECT_ROOT}
lerobot_root=${LEROBOT_ROOT}
robotwin_root=${ROBOTWIN_ROOT}
policy_path=${POLICY_PATH}
task_file=${TASK_FILE}
task_config=${TASK_CONFIG}
test_num=${TEST_NUM}
gpu_list=${GPUS[*]}
result_root=${RESULT_ROOT}
skip_get_obs_within_replan=${SKIP_GET_OBS_WITHIN_REPLAN}
instruction_source=${INSTRUCTION_SOURCE}
instruction_type=${INSTRUCTION_TYPE}
vista_rpc_base_port=${VISTA_RPC_BASE_PORT}
EOF

printf 'PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
printf 'ROBOTWIN_ROOT=%s\n' "${ROBOTWIN_ROOT}"
printf 'POLICY_PATH=%s\n' "${POLICY_PATH}"
printf 'OUTPUT_ROOT=%s\n' "${OUTPUT_ROOT}"
printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
printf 'TASK_FILE=%s\n' "${TASK_FILE}"
printf 'TEST_NUM=%s\n' "${TEST_NUM}"
printf 'GPU_LIST=%s\n' "${GPUS[*]}"

pids=()
for gpu in "${GPUS[@]}"; do
  shard_file="${SHARD_DIR}/gpu${gpu}.txt"
  log_file="${OUTPUT_ROOT}/logs/${RUN_ID}_gpu${gpu}.log"
  (
    PROJECT_ROOT="${PROJECT_ROOT}" \
    LEROBOT_ROOT="${LEROBOT_ROOT}" \
    ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" \
    CLIENT_PYTHON="${CLIENT_PYTHON}" \
    POLICY_PYTHON="${POLICY_PYTHON}" \
    TEST_NUM="${TEST_NUM}" \
    TASK_CONFIG="${TASK_CONFIG}" \
    CKPT_SETTING="${CKPT_SETTING}" \
    INSTRUCTION_TYPE="${INSTRUCTION_TYPE}" \
    INSTRUCTION_SOURCE="${INSTRUCTION_SOURCE}" \
    EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG}" \
    RENDER_FREQ="${RENDER_FREQ}" \
    SKIP_GET_OBS_WITHIN_REPLAN="${SKIP_GET_OBS_WITHIN_REPLAN}" \
    FAIL_FAST="${FAIL_FAST}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    VISTA_RPC_BASE_PORT="${VISTA_RPC_BASE_PORT}" \
    IMAGE_CHANNEL_ORDER="${IMAGE_CHANNEL_ORDER}" \
    ENABLE_FISHEYE="${ENABLE_FISHEYE}" \
    bash "${SCRIPT_DIR}/run_robotwin_umi_eval_shard.sh" "${shard_file}" "${gpu}" "${POLICY_PATH}" "${RUN_ID}"
  ) 2>&1 | tee "${log_file}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

"${CLIENT_PYTHON}" "${SCRIPT_DIR}/summarize_robotwin_umi.py" \
  --result-root "${RESULT_ROOT}" \
  --expected-tasks "${TASK_FILE}" \
  --out-prefix "${SUMMARY_PREFIX}"

echo "summary: ${SUMMARY_PREFIX}.tsv"
exit "${rc}"
