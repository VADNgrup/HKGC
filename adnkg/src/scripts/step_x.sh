#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SRC_ROOT}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/entity-dedup}"
THRESHOLD="${THRESHOLD:-0.80}"
MIN_SHARED_NEIGHBORS="${MIN_SHARED_NEIGHBORS:-2}"

echo "=========================================================="
echo "Direct entity deduplication evaluation"
echo "Run ID: ${RUN_ID}"
echo "=========================================================="

"${PYTHON}" "${SRC_ROOT}/reproduce/prepare_entity_dedup_gold.py"
"${PYTHON}" "${SRC_ROOT}/reproduce/run_entity_dedup_eval.py" \
    --run-id "${RUN_ID}" \
    --output-dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}" \
    --min-shared-neighbors "${MIN_SHARED_NEIGHBORS}"

echo "=========================================================="
echo "Evaluation artifacts:"
echo "  MiniRAG KG:      ${OUTPUT_DIR}/${RUN_ID}/minirag/graph_chunk_entity_relation.graphml"
echo "  MiniRAG entities:${OUTPUT_DIR}/${RUN_ID}/minirag/entities.json"
echo "  NDAKG KG:        ${OUTPUT_DIR}/${RUN_ID}/ndakg/graph_chunk_entity_relation.graphml"
echo "  NDAKG entities:  ${OUTPUT_DIR}/${RUN_ID}/ndakg/entities.json"
echo "  Comparison:      ${OUTPUT_DIR}/${RUN_ID}/table.md"
echo "  LaTeX table:     ${OUTPUT_DIR}/${RUN_ID}/table.tex"
echo "=========================================================="
