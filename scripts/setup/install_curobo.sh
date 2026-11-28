#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${VIRTUAL_ENV:-$ROOT_DIR/.venv}}"

export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export GIT_LFS_SKIP_SMUDGE=1
export MAX_JOBS="${MAX_JOBS:-4}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO:-0.7.0}"

VENV_PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "[install_curobo] missing venv python at $VENV_PYTHON"
  echo "[install_curobo] create the project venv or set UV_PROJECT_ENVIRONMENT" >&2
  exit 1
fi

export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"

if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  echo "[install_curobo] using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
else
  "$VENV_PYTHON" - <<'PYTHON'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU visible to PyTorch. Set TORCH_CUDA_ARCH_LIST for an offline build.")
print("Building for GPUs visible to PyTorch:", [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])
PYTHON
fi
echo "[install_curobo] using Python environment at: $VENV_PYTHON"

uv sync --locked --no-build-isolation-package nvidia-curobo

echo "[install_curobo] verifying curobo CUDA extensions"
"$VENV_PYTHON" - <<'PY'
import sys
import torch

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

try:
    import curobo.curobolib.lbfgs_step_cu
    import curobo.curobolib.kinematics_fused_cu
    import curobo.curobolib.line_search_cu
    import curobo.curobolib.tensor_step_cu
    import curobo.curobolib.geom_cu
    print("All CUDA extensions loaded successfully")
except ImportError as exc:
    print(f"CUDA extension import failed: {exc}")
    sys.exit(1)
PY

echo "[install_curobo] done"
