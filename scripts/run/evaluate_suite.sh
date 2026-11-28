#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <suite> [launch_evaluation options]"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
SUITE="$1"
shift
DATA_ROOT="${HUMANO_DATA_ROOT:-$REPO_ROOT/data}/eval/id"
if [[ -n "${HUMANO_EVAL_ROOT:-}" ]]; then
  DATA_ROOT="$HUMANO_EVAL_ROOT"
elif [[ -n "${HUMANO_DATA_ROOT:-}" && ! -d "$DATA_ROOT/$SUITE" && -d "$HUMANO_DATA_ROOT/public/eval/$SUITE" ]]; then
  DATA_ROOT="$HUMANO_DATA_ROOT/public/eval"
fi
SUITE_CONFIG="$REPO_ROOT/configs/suites/$SUITE.json"

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "[evaluate_suite] set RUN_DIR to the policy run or checkpoint directory"
  exit 2
fi
if [[ ! -f "$SUITE_CONFIG" ]]; then
  echo "[evaluate_suite] suite config not found: $SUITE_CONFIG"
  exit 2
fi
if [[ ! -d "$DATA_ROOT/$SUITE" ]]; then
  echo "[evaluate_suite] suite data not found: $DATA_ROOT/$SUITE; install the task data or set HUMANO_EVAL_ROOT"
  exit 2
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/launch_evaluation.py" \
  --run-dir "$RUN_DIR" \
  --policy "${POLICY:-act}" \
  --suite-config "$SUITE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --sonic-mode "${SONIC_MODE:-native}" \
  "$@"
