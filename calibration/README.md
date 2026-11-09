# Example Station Profile

This directory is a bundled normal-reference example for the M2 demo station. It is copied into the runtime data directory on first launch and is not modified in-place by the application.

The PNG files are normal appearances for 16 checkpoints. `approved-complete-board.jpg` and `approved-standard.json` bind the station's photographed regions to the board geometry. `manifest.json` and `provenance.json` describe the example samples. No missing or damaged sample is provided here.

For another camera, distance, board variant, or station, verify all regions and calibrate with a physically inspected complete board. A matching normal reference is not evidence that an arbitrary physical defect will be detected. Do not copy your operator ledger or private diagnostic captures into this directory when contributing changes.