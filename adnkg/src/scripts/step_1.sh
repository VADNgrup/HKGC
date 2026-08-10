#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
	set -a
	# shellcheck disable=SC1090
	. "${REPO_ROOT}/.env"
	set +a
fi

# python3 "${REPO_ROOT}/reproduce/Step_0_index.py"

python3 "${REPO_ROOT}/reproduce/Step_1_QA.py" -q 'Who is Tran Quoc Tuan?'