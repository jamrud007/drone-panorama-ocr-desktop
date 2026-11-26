# Halotec Container OCR Desktop

Windows desktop application for automated shipping container detection, drone video panorama stitching, ISO 6346 container number validation, and cloud OCR recognition.

## Technology Stack

- **Runtime:** Python 3.11 / 3.12 (64-bit Windows 10/11)
- **Desktop GUI:** PySide6 (`6.11.1`)
- **Local AI Detection:** ONNX Runtime (`onnxruntime-gpu` CUDA / DirectML / CPU), OpenCV (`4.12.0.88`), NumPy (`2.2.6`), SciPy
- **Tracking & Stitching Pipeline:** PyTorch (`2.9.1+cu126`), Torchvision, Ultralytics (`8.3.248`), `lap` (ByteTrack), Kornia (`0.9.2`), LightGlue (`ALIKED` + `LightGlue`)
- **Storage & Auth:** SQLite (`WAL` mode, single-writer in `Orchestrator`), Windows DPAPI via `ctypes` (`data/.token.dpapi`), Windows `SingleInstanceMutex` & `WindowsJobObject`
- **Cloud OCR Client:** HTTP REST (`requests`) against the OCR backend configured via `api.url` or `OCR_API_URL` (`/login`, `/refresh_token`, `/api/ocr`)
- **Packaging:** PyInstaller (`desktop.spec`, `scripts/build_exe.py`) and Inno Setup (`installer.iss`)

## Processing Pipeline

```text
MainWindow (PySide6 UI Thread)
  ├── cmd_queue / evt_queue ──> Orchestrator (Background Thread, Single SQLite Writer)
  │     ├── LEFT_RIGHT / TOP ──> StitchingAdapter -> pipeline_yolo_paronama.py
  │     │     ├── Stage 1: YOLO + ByteTrack container tracking & blur filtering
  │     │     ├── Stage 2: Ego-motion route ordering across container stacks
  │     │     ├── Stage 3: ALIKED + LightGlue + RANSAC panorama stitching
  │     │     ├── Stage 4: ONNX/YOLO text ROI detection on panorama (iso_code, container_number)
  │     │     └── Orchestrator extracts high-res 4K frame crops (panorama fallback) + 90-deg rotation retry (TOP)
  │     ├── FRONT_REAR ──> Direct OnnxDetector two-stage tracking (no stitching) + LK camera motion
  │     ├── OCR Loop ──> HalotecOcrClient POST /api/ocr (cropped text ROIs -> ISO 6346 validation)
  │     └── Video Output ──> render_overlay_video() -> outputs/<timestamp>_<video>/<video>_annotated.mp4
  └── SQLite (data/app.db) + Inspection Preview & Manual ISO Override
```

### Key Pipeline Rules

- **Camera Modes:**
  - `LEFT_RIGHT`: Side passes; runs tracking, panorama stitching, Stage 4 text detection, 4K crop selection, and OCR.
  - `TOP`: Roof passes; same stitching pipeline plus adaptive 90-degree CW/CCW crop rotation for vertical roof numbers.
  - `FRONT_REAR`: Door passes; skips stitching and runs direct motion-compensated 5-class ONNX detection on original frames.
- **ONNX Classes (`container_detection_model.onnx`, static `480x480` FP32):**
  `0: container_left_right`, `1: iso_code`, `2: container_number`, `3: container_rear_front`, `4: container_top`.
- **OCR API Contract:**
  Sends cropped ROI images (`Zero Double-Detection`) to `POST /api/ocr` with `gate` strictly as a numeric string (for example `"1"`). Non-numeric gates trigger HTTP 500 on the backend.

## Project Structure

```text
.
├── main.py                              # App entry point, SingleInstanceMutex, DPAPI check, --self-check
├── desktop.spec                         # PyInstaller bundle spec
├── installer.iss                        # Inno Setup Windows installer script
├── requirements.txt / requirements-dev.txt
├── config/config.example.json           # Default non-secret configuration template
├── src/
│   ├── config.py                        # Config dataclasses & frozen _MEIPASS path resolution
│   ├── database/db.py                   # SQLite wrapper (jobs, containers, detections, audit_logs)
│   ├── ocr_client/client.py             # Halotec OCR API HTTP client + ISO code regex extraction
│   ├── orchestrator/
│   │   ├── engine.py                    # Master job thread, 4K crop extraction, OCR loop & logging
│   │   └── annotation_motion.py         # Lucas-Kanade optical flow & video overlay renderer
│   ├── recognition/detector.py          # ONNX Runtime detector & CUDA/DirectML/CPU preflight probe
│   ├── stitching/adapter.py             # Stitching runner + WindowsJobObject subprocess fallback
│   ├── ui/                              # PySide6 dialogs & MainWindow
│   └── utils/                           # Unicode OpenCV I/O, ISO 6346 validator, DPAPI, Job Object
├── container-detection-drone-view/      # Tracking, stitching pipeline, YAML configs, trainer & MANIFEST.json
├── scripts/                             # build_exe.py, qa_verification.py, run_video_job.py
└── tests/                               # Offline unit & synthetic integration tests (pytest)
```

## Setup and Running

1. Create a Python 3.11 / 3.12 virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# NVIDIA CUDA 12.6 workstation:
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-dev.txt
```

*(For CPU-only machines without CUDA, install `onnxruntime` instead of `onnxruntime-gpu`.)*

2. Restore the ONNX model file (excluded from Git due to its `98.73 MiB` size):
   - Place `container_detection_model.onnx` at `container-detection-drone-view/weight/container_detection_model.onnx`.
   - Verify checksum matches `container-detection-drone-view/weight/MANIFEST.json` (`SHA-256: BCAA5D96715268A32905227CE05CC328898E3DA957FCEFB7399019D8D70E795F`):

```powershell
Get-FileHash container-detection-drone-view\weight\container_detection_model.onnx -Algorithm SHA256
```

3. Prepare configuration:
   - Copy `config/config.example.json` to `config/config.json` and set `api.url` to your internal OCR service URL (or set `OCR_API_URL` / `OCR_API_GATE` environment variables).
   - Log in via the desktop UI on first launch to save the encrypted JWT token in `data/.token.dpapi`.

4. Common commands:

```powershell
# Launch desktop app
python main.py

# Run headless CLI job
python scripts/run_video_job.py --video "path\to\video.mp4" --mode LEFT_RIGHT --name "TestJob"

# Run offline tests & QA verification
pytest
python scripts/qa_verification.py

# Build standalone Windows executable (dist/HalotecContainerOCR/)
python scripts/build_exe.py
```
