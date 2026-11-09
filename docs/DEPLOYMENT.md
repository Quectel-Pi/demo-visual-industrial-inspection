# Deployment and Operation

## Installation

Use the complete repository, not a single Python file. Keep `assets/`, `models/`, `calibration/`, and `static/` next to the source files. Installation does not need a GitHub login.

```bash
./install.sh
./run.sh
```

`PYTHON_BIN=/path/to/python3` selects the installer interpreter. `VENV_DIR=/path/to/venv` can place the environment elsewhere; use the same value when running. The application does not modify system Python packages.

To install without Internet access, prepare wheels for the target Python version and CPU architecture on a connected machine, then provide that directory:

```bash
WHEELHOUSE=/path/to/target-wheelhouse ./install.sh
```

The wheelhouse must include every pinned package and transitive dependency plus `rapidocr_onnxruntime==1.4.4`. Do not copy an x86 virtual environment to ARM. The included English OCR model and the models packaged in the RapidOCR wheel allow runtime recognition without model downloads.

## Camera and Display

The tested capture format is 640x480 YUYV. Higher advertised camera resolutions do not imply the same capture rate. Inspect actual formats using your distribution's V4L2 tools. No fixed camera serial number is required.

```bash
./run.sh --camera /dev/v4l/by-id/usb-YOUR_CAMERA-video-index0 --camera-width 640 --camera-height 480
./run.sh --image-only --port 8766
```

The browser must open the URL on the device. Changing `--host` to an external address removes the default loopback isolation; there is no authentication in this demonstration server.

The optional `launch_device.sh` opens a dedicated Firefox kiosk profile on an existing GNOME desktop. It is a convenience launcher, not a requirement for using the web interface. Do not start a second camera server against a camera already held by another process.

## Calibration and Updates

On first start, the bundled calibration is copied into the chosen data directory. Later starts do not replace it. Keep the current data directory when updating code. Back it up before replacing a camera, board variant, or station geometry.

Only use physically verified complete boards for pass references. Capture missing/damaged references from real corresponding specimens. Do not label blur, reflections, unknown changes, or an unconfirmed result as a missing component.

The included station profile is a starting point for evaluation. Reliable use at a new station requires calibration and independent validation. Board motion must not mix a stale overlay with a fresh preview: switch to Inspection Frame for the evidence image.

## Audio

Audio requires `paplay` and a working PulseAudio/PipeWire desktop audio device. Run as the desktop user where possible. If run as root, the demo locates an available user audio socket rather than assuming a specific username. The Qualified prompt uses 125% software playback gain with waveform headroom; the failure beep stays at 100%. The UI audio switch can disable feedback. Playback is asynchronous and is deduplicated per completed cycle.

## Data and Reports

`--data-dir` controls the calibration copy, output images/JSON, and SQLite ledger. The directory must be writable. Batch export is a self-contained HTML table, not a native Excel workbook. First-pass and post-reinspection yield use completed inspection cycles as the denominator, including cycles with unconfirmed items at timeout.

Historical retained pass state is not current-frame evidence. Reports retain this distinction. A failed or interrupted board must not be hidden from the yield count. Reinspection is linked to its original cycle rather than counted as another board.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| No camera found | Connect one USB capture camera, check permissions, or specify `--camera`; use `--image-only` for samples. |
| More than one camera | Select an explicit stable path under `/dev/v4l/by-id/`. |
| OCR initialization fails | Run the installer and inspect its error. Do not substitute a fixed M2 result. |
| Stuck waiting for M2 | Keep the actual marking visible and check OCR dependencies and image quality. |
| Checkpoint stays unconfirmed | Inspect the region, score, and reason; request fresh reinspection or recalibrate that station. |
| No audio | Check the UI switch, `paplay`, the audio session, and physical output volume. |
| No next cycle | Remove the board for about three seconds; a board in view is intentionally not recounted. |
| Port already in use | Stop your previous instance or use `--port 8766`. |

## Tests

```bash
.venv/bin/python -m unittest -q test_engine test_decisions test_live
.venv/bin/python verify_device.py --base http://127.0.0.1:8765
```

Check `verify_device.py --help` for supported options. Its WebDriver/restart options are diagnostic operations and must not run during an operator's inspection. Never interpret unit tests or synthetic erasure examples as a measured field defect-detection accuracy.