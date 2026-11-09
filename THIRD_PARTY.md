# Third-party Notices and Asset Sources

The root MIT license applies to original application code contributed to this repository. It does not replace the licenses of bundled third-party components or installed dependencies.

## Bundled Components

| Component | Source | License / notice |
| --- | --- | --- |
| Lucide 0.468.0 browser bundle | https://unpkg.com/lucide@0.468.0/dist/umd/lucide.min.js | ISC; includes Feather-derived portions. See [licenses/lucide-ISC.txt](licenses/lucide-ISC.txt). |
| English PP-OCRv3 recognition model | https://huggingface.co/SWHL/RapidOCR/resolve/main/PP-OCRv3/en_PP-OCRv3_rec_infer.onnx | The hosting model card declares Apache-2.0. The model is converted from PaddleOCR. See [licenses/PaddleOCR-Apache-2.0.txt](licenses/PaddleOCR-Apache-2.0.txt). |
| Standalone pip bootstrap | https://bootstrap.pypa.io/pip/pip.pyz | pip / get-pip are MIT-licensed. Their license texts are in `licenses/`; vendor license files are also included inside the unmodified zipapp. |

The English model's SHA256 is `ef7abd8bd3629ae57ea2c28b425c1bd258a871b93fd2fe7c433946ade9b5d9ea` and is checked by the installer. The pip bootstrap's SHA256 is `e5ef64ef63ff89a746014ad81658b76c1940385fdcdbc4fcbfa6e3ef1887352d`.

## Runtime Dependencies

The installer obtains pinned packages from PyPI, not a private package server. Their own distributions contain their applicable notices:

- RapidOCR `rapidocr_onnxruntime` 1.4.4: Apache-2.0. [Upstream](https://github.com/RapidAI/RapidOCR), [license copy](licenses/RapidOCR-Apache-2.0.txt).
- ONNX Runtime: MIT. [License](licenses/onnxruntime-MIT.txt) and [third-party notices](licenses/onnxruntime-ThirdPartyNotices.txt).
- OpenCV: Apache-2.0, with its distribution's third-party notices.
- NumPy: BSD-style license and bundled-library notices.
- Pillow: HPND-style license; PyYAML: MIT; six: MIT.
- Shapely: BSD-3-Clause and applicable GEOS notices; pyclipper: MIT / Clipper license notices.
- tqdm, pip vendors, and transitive dependencies retain their respective licenses.

RapidOCR's primary detection and recognition models are included in its wheel. Runtime OCR does not require contacting a model service. The standalone English fallback model is included in this repository rather than downloaded at launch.

## Project Assets

`assets/reference/`, `testdata/`, and the example `calibration/` profile contain contributor-supplied photographs and crops of the demo M2 board. They demonstrate the inspection geometry and are not an accuracy-certified dataset. The example profile includes normal samples only; no physical missing/damaged training samples are represented as tested defect coverage.

`static/audio/qualified.wav` is a fixed Mandarin notification saying “合格” (Qualified), generated for this demo and loudness-adjusted. `rejected.wav` is a generated short tone. These are prerecorded local assets, not a runtime cloud TTS integration or a recording of an operator.

Product names and marks visible in board photographs identify the hardware being demonstrated; their appearance does not grant trademark rights or imply a separate endorsement.