#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-${ROOT}/.venv}
WHEELHOUSE=${WHEELHOUSE:-}

"${PYTHON_BIN}" - <<'PY'
import platform
import sys
if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
    raise SystemExit("Supported targets: Linux x86_64 and aarch64.")
if not (3, 10) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit("Use Python 3.10, 3.11, 3.12, or the tested Debian 13 Python 3.13 environment.")
PY

if ! "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"; then
  printf 'Install Python venv support first: sudo apt-get install python3-venv\n' >&2
  exit 1
fi
PIP_ARGS=(--disable-pip-version-check)
PIP=("${VENV_DIR}/bin/python" "${ROOT}/tools/pip.pyz")
if [[ -n "${WHEELHOUSE}" ]]; then
  PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
fi
"${PIP[@]}" install "${PIP_ARGS[@]}" --only-binary=:all: -r "${ROOT}/requirements.txt"
"${PIP[@]}" install "${PIP_ARGS[@]}" --only-binary=:all: --no-deps --ignore-requires-python rapidocr_onnxruntime==1.4.4

cd "${ROOT}"
"${VENV_DIR}/bin/python" - <<'PY'
import hashlib
from pathlib import Path
from engine import BARE_PHOTO, POST_PHOTO, InspectionEngine
for path in [BARE_PHOTO, POST_PHOTO]:
    if not path.is_file():
        raise SystemExit(f"Missing bundled reference: {path}")
model = Path("models/en_PP-OCRv3_rec_infer.onnx")
expected = "ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea"
if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != expected:
    raise SystemExit("The bundled English OCR model is missing or corrupted. Download a complete checkout.")
engine = InspectionEngine()
engine.load_ocr(threads=1)
if engine.ocr is None or engine.latin_ocr is None:
    raise SystemExit(f"OCR initialization failed: {engine.ocr_error}")
print("Reference assets and both OCR recognizers are ready.")
PY

if ! command -v paplay >/dev/null 2>&1; then
  printf 'Audio is optional. For voice feedback install: sudo apt-get install pulseaudio-utils\n'
fi
printf '\nReady. Run ./run.sh for a USB camera, or ./run.sh --image-only for bundled samples.\n'