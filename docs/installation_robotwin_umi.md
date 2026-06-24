# RoboTwin-UMI Evaluation Installation

This guide adds RoboTwin-UMI evaluation on top of the VISTA post-training installation. Follow `docs/installation_finetuning.md` first so the VISTA LeRobot package and local Transformers fork are installed.

## Install RoboTwin

RoboTwin is an external simulator dependency. Install it from the official RoboTwin project and keep it outside this repository. The official installation guide is:

```text
https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

The recommended RoboTwin environment in the official guide is Linux with Python 3.10 and CUDA 12.1. Follow the guide for system packages such as Vulkan support and ffmpeg, then run the RoboTwin setup scripts from the RoboTwin checkout:

```bash
cd /path/to/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
```

The downloaded RoboTwin assets can be large. Do not copy them into `umi-vista`; keep them under the external RoboTwin checkout or another shared data directory.

## Prepare VISTA Runtime

Use the VISTA post-training environment and prepare the local Transformers fork:

```bash
export VISTA_ROOT=/path/to/umi-vista
export LEROBOT_ROOT=${VISTA_ROOT}/post_training/lerobot
cd ${LEROBOT_ROOT}
bash third_party/prepare_pi_transformers.sh
pip install --no-build-isolation -e ".[pi]"
```

Set runtime paths for evaluation:

```bash
export ROBOTWIN_ROOT=/path/to/RoboTwin
export POLICY_PATH=/path/to/vista_robotwin_umi_checkpoint/pretrained_model
export OUTPUT_ROOT=/path/to/vista_outputs/robotwin_umi_eval
```

`POLICY_PATH` must point to a VISTA checkpoint directory containing `config.json`. Public checkpoints are distributed through the VISTA Hugging Face collection.

## Smoke Check

Run one episode on the default smoke task:

```bash
cd ${VISTA_ROOT}

ROBOTWIN_ROOT=${ROBOTWIN_ROOT} \
POLICY_PATH=${POLICY_PATH} \
OUTPUT_ROOT=${OUTPUT_ROOT}/smoke \
GPU_LIST=0 \
TEST_NUM=1 \
bash simulation_evaluation/robotwin_umi/run_robotwin_umi_eval.sh
```

A successful run writes `summary.tsv`, `summary.json`, logs, and per-task `_result.json` files under `OUTPUT_ROOT`.

## Full 50-Task Evaluation

After the smoke check passes, run the full task list explicitly:

```bash
cd ${VISTA_ROOT}

ROBOTWIN_ROOT=${ROBOTWIN_ROOT} \
POLICY_PATH=${POLICY_PATH} \
OUTPUT_ROOT=${OUTPUT_ROOT}/full50 \
GPU_LIST="0 1 2 3" \
TEST_NUM=100 \
TASK_FILE=simulation_evaluation/robotwin_umi/tasks/full_50_tasks.txt \
EVAL_VIDEO_LOG=false \
SKIP_GET_OBS_WITHIN_REPLAN=true \
bash simulation_evaluation/robotwin_umi/run_robotwin_umi_eval.sh
```

`TEST_NUM` is episodes per task. For benchmark-style runs, keep `EVAL_VIDEO_LOG=false` to avoid large video outputs.

## Output Files

The runner creates a timestamped output directory unless `OUTPUT_ROOT` is set. The important files are:

```text
summary.tsv
summary.json
rpc_results/<run_id>/**/_result.json
rpc_results/<run_id>/lane_status/*.tsv
rpc_results/<run_id>/run_meta.txt
logs/*.log
```

Generated output directories are ignored by git. Keep benchmark artifacts outside the repository when possible.
