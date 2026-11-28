#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[evaluation] no usable python interpreter found"
  exit 1
fi

RUN_DIR="${RUN_DIR:-}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$REPO_ROOT/data/eval/id/basic/G1WholebodyGroundBoxPickTeleop}"
TASK_DIR_NAME="$(basename "$EVAL_DATA_DIR")"
if [[ "$TASK_DIR_NAME" == level-* ]]; then
    TASK_DIR_NAME="$(basename "$(dirname "$EVAL_DATA_DIR")")"
fi
ENV_ID="${ENV_ID:-humano/${TASK_DIR_NAME%-v0}-v0}"
SONIC_MODE="${SONIC_MODE:-native}"
HAND_TYPE="${HAND_TYPE:-dex3}"
SIM_MODE="${SIM_MODE:-mujoco_isaac}"

if [[ -z "$RUN_DIR" ]]; then
  echo "[evaluation] set RUN_DIR to the GR00T run or checkpoint directory"
  exit 2
fi
if [[ ! -f "$EVAL_DATA_DIR/meta/info.json" ]]; then
  echo "[evaluation] LeRobot dataset metadata not found: $EVAL_DATA_DIR/meta/info.json"
  echo "[evaluation] set EVAL_DATA_DIR to a valid local LeRobot dataset root"
  exit 2
fi

case "$SONIC_MODE" in
  internal)
    # eval_task starts and manages the SONIC deploy service.
    "$PYTHON_BIN" scripts/run/launch_evaluation.py \
        --run-dir "$RUN_DIR" \
        --data-dir "$EVAL_DATA_DIR" \
        --policy gr00t_n17 \
        --env-id "$ENV_ID" \
        --sim-mode "$SIM_MODE" \
        --hand-type "$HAND_TYPE" \
        --num-episodes 4 \
        --port 22085 \
        --sonic-mode internal \
        --env-step-dt 0.1 \
        --sonic-control-dt 0.4 \
        --save-video \
        --save-exocentric-video
    ;;
  native)
    # Use the in-process native SonicController.
    "$PYTHON_BIN" scripts/run/launch_evaluation.py \
        --run-dir "$RUN_DIR" \
        --data-dir "$EVAL_DATA_DIR" \
        --policy gr00t_n17 \
        --env-id "$ENV_ID" \
        --sim-mode "$SIM_MODE" \
        --hand-type "$HAND_TYPE" \
        --num-episodes 1 \
        --episode-start 0 \
        --port 22085 \
        --sonic-mode native \
        --seed 42 \
        --deterministic \
        --fast
    ;;
  *)
    echo "[evaluation] unsupported SONIC_MODE: $SONIC_MODE (expected internal or native)"
    exit 2
    ;;
esac
