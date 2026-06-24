# RoboTwin-UMI Evaluation

This directory contains the VISTA-only RoboTwin-UMI evaluation adapter. It does not vendor RoboTwin, RoboTwin assets, checkpoints, videos, debug images, or benchmark outputs. Install RoboTwin separately, then point this runner to your RoboTwin checkout and VISTA checkpoint.

## Requirements

- A working VISTA post-training installation from `post_training/lerobot`.
- A RoboTwin checkout with assets downloaded by the official RoboTwin scripts.
- A VISTA RoboTwin-UMI checkpoint directory containing `config.json`.

Follow the official RoboTwin installation guide for simulator dependencies and assets:

```text
https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

See `docs/installation_robotwin_umi.md` for a VISTA-specific setup checklist.

## Smoke Evaluation

The default task file contains one small eval-through task. This is intended to verify that the policy server, RoboTwin client, and summarizer can run end to end.

```bash
cd ${VISTA_ROOT}

ROBOTWIN_ROOT=/path/to/RoboTwin \
POLICY_PATH=/path/to/vista_robotwin_umi_checkpoint/pretrained_model \
OUTPUT_ROOT=/path/to/vista_outputs/robotwin_umi_smoke \
GPU_LIST=0 \
TEST_NUM=1 \
bash simulation_evaluation/robotwin_umi/run_robotwin_umi_eval.sh
```

The runner writes logs, per-task result JSON files, and summaries under `OUTPUT_ROOT`:

```text
summary.tsv
summary.json
rpc_results/<run_id>/
rpc_shards/<run_id>/
logs/
```

## Full 50-Task Evaluation

Use the bundled full task list only when your RoboTwin environment and checkpoint are ready.

```bash
ROBOTWIN_ROOT=/path/to/RoboTwin \
POLICY_PATH=/path/to/vista_robotwin_umi_checkpoint/pretrained_model \
OUTPUT_ROOT=/path/to/vista_outputs/robotwin_umi_full50 \
GPU_LIST="0 1 2 3" \
TEST_NUM=100 \
TASK_FILE=simulation_evaluation/robotwin_umi/tasks/full_50_tasks.txt \
bash simulation_evaluation/robotwin_umi/run_robotwin_umi_eval.sh
```

## Important Options

- `ROBOTWIN_ROOT`: path to the external RoboTwin checkout.
- `POLICY_PATH`: path to a VISTA `pretrained_model` checkpoint.
- `GPU_LIST`: one or more GPU ids, separated by spaces or commas.
- `TASK_FILE`: one task name per line.
- `TEST_NUM`: episodes per task.
- `TASK_CONFIG`: defaults to `demo_clean_umi_new`.
- `SKIP_GET_OBS_WITHIN_REPLAN`: defaults to `true`. The client still refreshes observation before every action chunk.
- `EVAL_VIDEO_LOG`: defaults to `false`; keep it disabled for benchmark runs unless you need videos.

## What Is Not Included

This repository intentionally excludes RoboTwin assets, downloaded objects, generated logs, videos, debug images, raw result directories, and checkpoints. VISTA checkpoints are distributed through the VISTA Hugging Face collection.
