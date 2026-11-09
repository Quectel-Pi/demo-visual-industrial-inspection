# M2 Visual Assembly Inspection

An on-device visual inspection demo for the front-side assembly of a Quectel Pi M2 board. It recognizes the M2 marking, inspects **12 interfaces and four copper standoffs**, reinspects individual checkpoints, announces the result, and exports per-board and batch reports.

The repository includes application source, the browser interface, reference photographs, a sample calibration profile, an English OCR fallback model, audio prompts, and regression tests. No sibling repository or private deployment folder is required.

[中文说明](README_zh.md) · [Operation and deployment](docs/DEPLOYMENT.md) · [Third-party notices](THIRD_PARTY.md)

## Quick Start

Supported targets: Linux **aarch64 / x86_64**, Python **3.10–3.13**. The target-device baseline is Debian 13 on M2, a USB camera, a display, and optional speaker output. Installation requires access to PyPI; inspection works offline after installation.

```bash
git clone https://github.com/Quectel-Pi/demo-visual-industrial-inspection.git
cd demo-visual-industrial-inspection
./install.sh
./run.sh
```

Open **http://127.0.0.1:8765/** in a browser **on the machine running the application**. `./run.sh` discovers a single USB capture node under `/dev/v4l/by-id/`; it does not guess an ISP camera node. If multiple USB cameras are connected, select one explicitly:

```bash
./run.sh --camera /dev/v4l/by-id/usb-YOUR_CAMERA-video-index0
```

To try the included images without opening a camera:

```bash
./run.sh --image-only
```

Image-only mode is a manual debugging workflow, not a live production inspection. On Debian/Ubuntu, install missing prerequisites with your administrator:

```bash
sudo apt-get update
sudo apt-get install python3 python3-venv libglib2.0-0 libgl1
# Optional device audio and dedicated kiosk browser:
sudo apt-get install pulseaudio-utils firefox-esr
```

On Ubuntu, use the distribution's Firefox package instead of `firefox-esr` if needed. Run the application as the desktop user, and ensure that user can access the USB camera (typically the `video` group). Do not use `sudo pip`.

RapidOCR 1.4.4 declares Python `<3.13`, although the demo has been tested on the M2 Debian 13 / Python 3.13 environment. The installer deliberately installs that one pinned package with `--ignore-requires-python --no-deps` after installing its explicit dependencies. This is a tested compatibility exception, not an upstream support claim.

## Inspection Workflow

1. Put a complete board in view. Actual OCR must recognize **M2**; no fixed model or synthetic success is inserted.
2. The system inspects all 16 checkpoints. It finalizes and announces **Qualified** as soon as every checkpoint is confirmed; otherwise the maximum wait is **20 seconds**.
3. A failed cycle produces a short beep. An unconfirmed checkpoint is not automatically classified as missing.
4. Remove the board for about three seconds before the next cycle. A board that remains in view is not repeatedly counted.
5. Click an unconfirmed checkpoint to request three fresh reinspection frames. Only that checkpoint is reset; other results remain intact.

The default UI and voice prompt are Chinese. This repository's documentation describes the controls in English as well. Wiring inspection and electrical interface tests are not implemented.

## Calibration and Evidence

The bundled profile is an example of a known complete M2 board, **not a universal calibration for every camera or station**. Copying it into a fresh data directory allows immediate evaluation. For a different station, position a verified complete board in view and use the UI's station-calibration controls before relying on results.

Normal-reference matching uses a 65% threshold plus image-quality and temporal checks. Upper USB and VRTC use two normal observations within three seconds, counted independently. Other checkpoints require three consecutive confirmations. Calibrated defect evidence takes priority. The camera must still see enough of the board for reliable localization.

Runtime state is isolated under `data/` and is not committed:

```text
data/
	calibration/        # Initial copy of the bundled profile; operator changes live here
	output/             # Original image, annotated image, JSON; HTML tables served on request
	production.sqlite3  # Completed cycles and checkpoint reinspection links
```

Override with `INSPECTION_DATA_DIR=/path/to/data` or `--data-dir /path/to/data`. Existing data and calibration are never overwritten by the installer. Back up this directory before changing stations.

The batch report includes board sequence numbers, initial and post-reinspection outcomes, known missing/damaged components, unconfirmed checkpoints, interruption flags, and first-pass / post-reinspection yields. Reinspection does not increment the board count. These are inspection-cycle counts, not hardware-serial-number tracking.

## Checkpoints

| ID | Inspection target |
| --- | --- |
| A01 | 40-pin GPIO header |
| A02 | SD / SIM socket |
| A03 | HDMI port |
| A04 | CAM0 ribbon connector |
| A05 | Audio jack |
| A06 / A07 | Upper / lower USB assembly |
| A08 | Ethernet port |
| A09 | USB Type-C power port |
| A10 | PoE header |
| A11 | PCIe ribbon connector |
| A12 | VRTC connector |
| B01–B04 | Four copper standoffs |

## Verification

```bash
.venv/bin/python -m unittest -q test_engine test_decisions test_live
.venv/bin/python verify_device.py --help
```

The tests cover model gating, geometry, sample decisions, defect precedence, early approval, timeouts, reinspection freshness, report export, persistent yield counts, and audio deduplication. Test success is **not** industrial defect-accuracy certification. Fine missing/bent pins and actual electrical functionality require a separate validation plan and real defective samples.

## Security and Licensing

The server binds to loopback by default. It is an unauthenticated local demo; do not expose it directly to the Internet. Camera images, OCR readings, and reports are processed locally. No credentials, original operator ledger, personal Word documents, or diagnostic history are part of this repository.

Original application code is under the [MIT license](LICENSE). Third-party libraries and models retain their own licenses; see [THIRD_PARTY.md](THIRD_PARTY.md).

## Practical Operations Summary

This section summarizes the validated deployment flow used during real project delivery.

1. Build a standalone release folder from this repository and exclude `.git` and Python caches.
2. Generate `SHA256SUMS.txt` for package integrity checks.
3. Compress the release folder into a single `tar.gz` file for transfer.
4. On target devices, use `install.sh` for online installation.
5. For offline devices, prepare `wheelhouse/` in advance and install with `tools/pip.pyz`.
6. Start with `run.sh`, then verify camera nodes by `v4l2-ctl --list-devices` and `ls /dev/video*`.
7. Use image-only mode for no-camera smoke checks, then validate with real camera and board.
8. Run regression tests before release:

```bash
.venv/bin/python -m unittest -q test_engine test_decisions test_live
```

Validated behavior points from field integration:

- All 16 checkpoints can be visible on 1280x800 and 1280x720 layouts after UI compaction.
- A06 (upper USB assembly) and A12 (VRTC) use independent "2 hits in 3 seconds" confirmation windows.
- The cycle finalizes early when all 16 checkpoints are confirmed; 20 seconds is only the upper timeout.
- Reinspection updates only selected checkpoints and does not increase the board count.
- Runtime data is isolated in `data/` and can be relocated with `--data-dir` or `INSPECTION_DATA_DIR`.