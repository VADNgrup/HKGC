#!/usr/bin/env bash
set -euo pipefail

# Find repository root directory relative to the script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Inputs / Defaults
DIR1="${1:-${REPO_ROOT}/../output/VNH3045-minirag-9b}"
DIR2="${2:-${REPO_ROOT}/../output/VNH3045-fl-9b-r3}"
OUTPUT_DIR="${3:-./compare_results}"
# The rest of arguments are nodes
shift 3 || true
if [ $# -eq 0 ]; then
    NODES=("Việt Minh")
else
    NODES=("$@")
fi

TITLE1="${TITLE1:-MiniRAG }"
TITLE2="${TITLE2:-NDAKG }"

echo "=========================================================="
echo "Comparing Knowledge Graphs (Side-by-Side Visual Image)"
echo "Folder 1: ${DIR1}"
echo "Folder 2: ${DIR2}"
echo "Nodes: ${NODES[*]}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "=========================================================="

python3 "${REPO_ROOT}/graph-visuals/compare_kg_visual.py" \
    --dir1 "${DIR1}" \
    --dir2 "${DIR2}" \
    --output_dir "${OUTPUT_DIR}" \
    --nodes "${NODES[@]}" \
    --title1 "${TITLE1}" \
    --title2 "${TITLE2}" \
    --radius 1

echo ""
echo "=========================================================="
echo "Comparison images saved to directory: ${OUTPUT_DIR}"
echo "=========================================================="

