#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR=${VENV_DIR:-${ROOT}/.venv}
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  printf 'Run ./install.sh first.\n' >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
export INSPECTION_DATA_DIR=${INSPECTION_DATA_DIR:-${ROOT}/data}
cd "${ROOT}"
exec "${VENV_DIR}/bin/python" "${ROOT}/server.py" "$@"