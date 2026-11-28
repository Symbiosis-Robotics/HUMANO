#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]] || ! "$VENV_PYTHON" -m pre_commit --version; then
  echo "Install the project's dev dependencies in .venv first; see docs/install.md." >&2
  exit 1
fi

HOOK_URL="$(git ls-remote --get-url https://github.com/pre-commit/pre-commit-hooks)"
if [[ "$HOOK_URL" == git://* ]]; then
  echo "Git rewrites the hook's HTTPS URL to git://. Review the URL rules before installing:" >&2
  echo "git config --show-origin --get-regexp 'url\..*\.insteadof'" >&2
  exit 1
fi

export PRE_COMMIT_HOME="$ROOT_DIR/.cache/pre-commit"
export PIP_CACHE_DIR="$PRE_COMMIT_HOME/pip"
export TMPDIR="$PRE_COMMIT_HOME/tmp"
mkdir -p "$TMPDIR"
"$VENV_PYTHON" -m pre_commit install
"$VENV_PYTHON" - <<'PY'
import os
import subprocess
from pathlib import Path

hook_path = Path(subprocess.check_output(
    ["git", "rev-parse", "--git-path", "hooks/pre-commit"], text=True,
).strip())
cache_path = Path(os.environ["PRE_COMMIT_HOME"]).relative_to(Path.cwd()).as_posix()
header = f'''HUMANO_REPO_ROOT="$(git rev-parse --show-toplevel)" || exit 1
export PRE_COMMIT_HOME="$HUMANO_REPO_ROOT/{cache_path}"
export PIP_CACHE_DIR="$PRE_COMMIT_HOME/pip"
export TMPDIR="$PRE_COMMIT_HOME/tmp"
mkdir -p "$TMPDIR" || exit 1
'''
lines = hook_path.read_text().splitlines(keepends=True)
lines = [
    'INSTALL_PYTHON="$HUMANO_REPO_ROOT/.venv/bin/python"\n'
    if line.startswith("INSTALL_PYTHON=") else line
    for line in lines
]
hook_path.write_text(lines[0] + header + "".join(lines[1:]))
PY
"$VENV_PYTHON" -m pre_commit install-hooks
echo "HUMANO commit hooks ready; cache: $PRE_COMMIT_HOME"
