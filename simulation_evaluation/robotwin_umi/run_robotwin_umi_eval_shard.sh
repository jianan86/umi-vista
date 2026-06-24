#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "Usage: $0 <shard_file> <gpu_id> <policy_path> <run_id>" >&2
  exit 2
fi

SHARD_FILE="$1"
GPU_ID="$2"
POLICY_PATH="$3"
RUN_ID="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${PROJECT_ROOT}/post_training/lerobot}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:?missing ROBOTWIN_ROOT}"
CLIENT_PYTHON="${CLIENT_PYTHON:-python3}"
POLICY_PYTHON="${POLICY_PYTHON:-python3}"
TEST_NUM="${TEST_NUM:-1}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean_umi_new}"
SEED="${SEED:-1}"
CKPT_SETTING="${CKPT_SETTING:-pretrained_model}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
INSTRUCTION_SOURCE="${INSTRUCTION_SOURCE:-generated_template}"
EVAL_VIDEO_LOG="${EVAL_VIDEO_LOG:-false}"
RENDER_FREQ="${RENDER_FREQ:-0}"
SKIP_GET_OBS_WITHIN_REPLAN="${SKIP_GET_OBS_WITHIN_REPLAN:-true}"
FAIL_FAST="${FAIL_FAST:-false}"
RESULT_ROOT="${RESULT_ROOT:?missing RESULT_ROOT}"
LANE_STATUS_DIR="${LANE_STATUS_DIR:-${RESULT_ROOT}/lane_status}"
DEBUG_IMAGE_DIR="${DEBUG_IMAGE_DIR:-${RESULT_ROOT}/debug_images}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${RESULT_ROOT}/server_logs}"
VISTA_RPC_BASE_PORT="${VISTA_RPC_BASE_PORT:-18150}"
IMAGE_CHANNEL_ORDER="${IMAGE_CHANNEL_ORDER:-rgb}"
ENABLE_FISHEYE="${ENABLE_FISHEYE:-true}"

mkdir -p "${RESULT_ROOT}" "${LANE_STATUS_DIR}" "${DEBUG_IMAGE_DIR}" "${SERVER_LOG_DIR}"
STATUS_TSV="${LANE_STATUS_DIR}/gpu${GPU_ID}.tsv"
printf 'timestamp\ttask\tstatus\texit_code\tseconds\n' > "${STATUS_TSV}"

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

exit_code=0
while IFS= read -r TASK_NAME || [[ -n "${TASK_NAME}" ]]; do
  TASK_NAME="${TASK_NAME%%#*}"
  TASK_NAME="$(printf '%s' "${TASK_NAME}" | xargs)"
  [[ -z "${TASK_NAME}" ]] && continue

  start_s=$(date +%s)
  port=$((VISTA_RPC_BASE_PORT + GPU_ID))
  server_log="${SERVER_LOG_DIR}/${TASK_NAME}_gpu${GPU_ID}_server.log"
  echo "[robotwin-umi-shard] task_start $(date) ${TASK_NAME} port=${port}"

  SERVER_PID=""
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${SCRIPT_DIR}:${LEROBOT_ROOT}/src:${LEROBOT_ROOT}/third_party/pi_transformers/src:${PYTHONPATH:-}" \
  "${POLICY_PYTHON}" "${SCRIPT_DIR}/vista_policy_server.py" \
    --host 127.0.0.1 \
    --port "${port}" \
    --policy-path "${POLICY_PATH}" \
    --task-name "${TASK_NAME}" \
    --project-root "${PROJECT_ROOT}" \
    --lerobot-root "${LEROBOT_ROOT}" \
    --debug-image-dir "${DEBUG_IMAGE_DIR}" \
    --image-channel-order "${IMAGE_CHANNEL_ORDER}" \
    --enable-fisheye "${ENABLE_FISHEYE}" \
    > "${server_log}" 2>&1 &
  SERVER_PID=$!

  ready=0
  for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    if grep -q "listening 127.0.0.1:${port}" "${server_log}" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 2
  done

  if [[ "${ready}" -ne 1 ]]; then
    echo "[robotwin-umi-shard] server failed to become ready for ${TASK_NAME}; log=${server_log}" >&2
    tail -80 "${server_log}" >&2 || true
    rc=1
  else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="${SCRIPT_DIR}:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/description/utils:${PYTHONPATH:-}" \
    "${CLIENT_PYTHON}" "${SCRIPT_DIR}/eval_robotwin_umi_rpc_client.py" \
      --robotwin-root "${ROBOTWIN_ROOT}" \
      --policy-path "${POLICY_PATH}" \
      --task-name "${TASK_NAME}" \
      --task-config "${TASK_CONFIG}" \
      --seed "${SEED}" \
      --test-num "${TEST_NUM}" \
      --policy-name "vista_rpc" \
      --tag "vista_robotwin_umi_rpc" \
      --ckpt-setting "${CKPT_SETTING}" \
      --instruction-type "${INSTRUCTION_TYPE}" \
      --instruction-source "${INSTRUCTION_SOURCE}" \
      --eval-result-root "${RESULT_ROOT}" \
      --eval-video-log "${EVAL_VIDEO_LOG}" \
      --render-freq "${RENDER_FREQ}" \
      --skip-get-obs-within-replan "${SKIP_GET_OBS_WITHIN_REPLAN}" \
      --rpc-host 127.0.0.1 \
      --rpc-port "${port}"
    rc=$?
  fi

  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    ROBOTWIN_UMI_RPC_PORT="${port}" \
    PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
    "${CLIENT_PYTHON}" - <<'PYSHUTDOWN' >/dev/null 2>&1 || true
import os
from rpc_common import RPCClient
with RPCClient("127.0.0.1", int(os.environ["ROBOTWIN_UMI_RPC_PORT"]), timeout=5) as client:
    client.call("shutdown")
PYSHUTDOWN
  fi
  cleanup_server

  end_s=$(date +%s)
  seconds=$((end_s - start_s))
  if [[ "${rc}" -eq 0 ]]; then
    status=ok
  else
    status=failed
    exit_code="${rc}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${TASK_NAME}" "${status}" "${rc}" "${seconds}" >> "${STATUS_TSV}"
  echo "[robotwin-umi-shard] task_done $(date) ${TASK_NAME} status=${status} rc=${rc} seconds=${seconds}"

  if [[ "${rc}" -ne 0 && "${FAIL_FAST}" == "true" ]]; then
    exit "${rc}"
  fi
done < "${SHARD_FILE}"

exit "${exit_code}"
