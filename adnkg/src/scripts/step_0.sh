#!/usr/bin/env bash
# NOTE: Intentionally NOT using `set -e` so that a failure in one experiment
# does NOT abort the remaining comparison runs.
set -uo pipefail

# Find repository root directory relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment configuration if it exists
if [ -f "${REPO_ROOT}/.env" ]; then
    echo "Loading environment from ${REPO_ROOT}/.env"
    set -a
    . "${REPO_ROOT}/.env"
    set +a
fi

# Default parameters derived from environment or defaults
LLM_PROVIDER="${LLM_PROVIDER:-ollama}"
LLM_MODEL="${LLM_MODEL:-qwen3.5:35b}"
GRAPH_STORAGE="NetworkXStorage" # Default to file-based NetworkXStorage for lightweight execution
# DATAPATH="${REPO_ROOT}/../dataset/VHis/data"
# NAME="Vhis-fl-9b"
# DATAPATH="${REPO_ROOT}/../dataset/LiHua-World/LiHua-World"
# NAME="LiW-fl-9b"
DATAPATH="${REPO_ROOT}/../dataset-v1/.fined/wiki-text"
NAME="VNH3045-fl-noschema-9b"

echo "=========================================================="
echo "Running Step 0 Indexing WITH Fuzzy Logic BUT WITHOUT Schema (--fl -mp)"
echo "=========================================================="
python "${REPO_ROOT}/reproduce/Step_0_index.py" \
    --llm-provider "${LLM_PROVIDER}" \
    --llm-model "${LLM_MODEL}" \
    --graph-storage "${GRAPH_STORAGE}" \
    --datapath "${DATAPATH}" \
    --workingdir "${REPO_ROOT}/../output/${NAME}" \
    --fl \
    -mp



