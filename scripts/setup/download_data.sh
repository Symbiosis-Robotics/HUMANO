#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
echo "📥 Downloading data from Hugging Face..."
uv run python scripts/setup/download_data.py --local-dir data/resources "$@"
echo "✅ All data downloaded successfully."
