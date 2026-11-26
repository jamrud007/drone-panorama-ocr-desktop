# Halotec Container OCR Desktop

Windows desktop application for automated shipping container detection, drone video panorama stitching, ISO 6346 container number validation, and cloud OCR recognition.

## Technology Stack

- **Runtime & OS:** Python 3.11 / 3.12 (64-bit Windows 10/11)
- **Desktop GUI:** PySide6 (`6.11.1`)
- **Local AI Detection:** ONNX Runtime (`onnxruntime-gpu` CUDA / DirectML / CPU fallback), OpenCV (`4.12.0.88`), NumPy (`2.2.6`), SciPy (`1.16.3`)
- **Tracking & Stitching Pipeline:** PyTorch (`2.9.1+cu126`), Torchvision (`0.24.1`), Ultralytics (`8.3.248`), `lap` (ByteTrack), Kornia (`0.9.2`), LightGlue (`ALIKED` + `LightGlue`)
- **Persistence & Security:** SQLite (`WAL` mode, single-writer in `Orchestrator`), Windows DPAPI via `ctypes` (`data/.token.dpapi`), Windows `SingleInstanceMutex` & `WindowsJobObject`
- **Cloud OCR Client:** HTTP REST (`requests`) against the configured OCR backend (`api.url` or `OCR_API_URL`) supporting `/login`, `/refresh_token`, and `/api/ocr`
- **Packaging:** PyInstaller (`desktop.spec`, `scripts/build_exe.py`) and Inno Setup (`installer.iss`)

## Processing Pipeline

```text
MainWindow (PySide6 UI Thread)
  ├── cmd_queue / evt_queue ──> Orchestrator Thread (Single SQLite Writer)
  │     ├── LEFT_RIGHT / TOP ──> StitchingAdapter -> pipeline_yolo_paronama.py
  │     │     ├── Stage 1: YOLO + ByteTrack container tracking & Laplacian blur filter
  │     │     ├── Stage 2: Camera ego-motion route ordering across container stacks
  │     │     ├── Stage 3: ALIKED + LightGlue + RANSAC panorama stitching
  │     │     ├── Stage 4: Local text ROI detection on panorama (iso_code, container_number)
  │     │     └── 4K native crop extraction via homography (panorama fallback) + 90-deg retry (TOP)
  │     ├── FRONT_REAR ──> Direct OnnxDetector two-stage tracking (no stitching) + LK camera motion
  │     ├── OCR Loop ──> HalotecOcrClient POST /api/ocr (cropped text ROIs -> ISO 6346 check digit)
  │     └── Video Render ──> render_overlay_video() -> outputs/<timestamp>_<video>/<video>_annotated.mp4
  └── SQLite (data/app.db) + RAW/Annotated Inspection & Manual ISO Override UI
```

### Key Pipeline Behavior

- **Camera Modes:**
  - `LEFT_RIGHT`: Side passes; runs tracking, panorama stitching, Stage 4 ROI detection, 4K crop selection, OCR, and annotated video export.
  - `TOP`: Top-down roof passes; runs the same stitching pipeline plus adaptive 90-degree CW/CCW crop rotation for longitudinal roof numbers.
  - `FRONT_REAR`: Front/rear door passes; skips stitching and performs direct motion-compensated 5-class ONNX detection on sampled video frames.
- **ONNX Class Contract (`container_detection_model.onnx`, static `480x480` FP32):**
  `0: container_left_right`, `1: iso_code`, `2: container_number`, `3: container_rear_front`, `4: container_top`.
- **Zero Double-Detection & API Contract:**
  Text ROIs are cropped locally from RAW images and submitted to `POST /api/ocr` (`gate` must be a numeric string such as `"1"`; non-numeric values trigger HTTP 500 on the backend).
- **Compute Auto-Selection:**
  `stitching.device = "auto"` probes `CUDAExecutionProvider` (NVIDIA), `DmlExecutionProvider` (DirectML), and falls back to `CPUExecutionProvider`.

## Repository Structure

```text
.
├── main.py                              # GUI entry point, SingleInstanceMutex, DPAPI check, --self-check
├── desktop.spec                         # PyInstaller bundle specification
├── installer.iss                        # Inno Setup Windows installer script
├── pyproject.toml                       # Project metadata, ruff, and mypy config
├── requirements.txt                     # Core runtime dependencies
├── requirements-dev.txt                 # Dev, QA (pytest/psutil), ONNX export, and PyInstaller deps
├── .env.example                         # Optional environment variable overrides
├── config/config.example.json           # Non-secret default configuration template
├── data/.gitkeep                        # Runtime folder placeholder (local app.db & .token.dpapi gitignored)
├── outputs/.gitkeep                     # Job session output placeholder (images/, logs/, _annotated.mp4)
├── src/
│   ├── config.py                        # Typed AppConfig loader & frozen _MEIPASS path resolution
│   ├── database/db.py                   # SQLite persistence (jobs, containers, detections, audit_logs)
│   ├── ocr_client/client.py             # OCR API HTTP client, retry/refresh, ISO code regex parsing
│   ├── orchestrator/
│   │   ├── engine.py                    # Background job scheduler, 4K crop selection, OCR loop & logs
│   │   └── annotation_motion.py         # Lucas-Kanade optical flow & annotated video renderer
│   ├── recognition/detector.py          # ONNX Runtime wrapper & GPU/CPU preflight diagnostics
│   ├── stitching/adapter.py             # Stitching runner + WindowsJobObject subprocess fallback
│   ├── ui/                              # PySide6 MainWindow, LoginDialog, SettingsDialog, VideoPlayerDialog
│   └── utils/                           # Unicode image I/O, ISO 6346 check digit, DPAPI, Windows Job Object
├── container-detection-drone-view/
│   ├── pipeline_yolo_paronama.py        # Tracking, ALIKED/LightGlue stitching & Stage 4 ROI detector
│   ├── hyper_config/                    # YOLO training & custom ByteTrack YAML configs
│   ├── trainer/                         # Model training (train.py) & ONNX export (export.py)
│   └── weight/MANIFEST.json             # Checksum & schema manifest for excluded ONNX model weights
└── scripts/
    ├── build_exe.py                     # PyInstaller packaging helper
    ├── qa_verification.py               # Environment, model, API contract & build QA script
    └── run_video_job.py                 # Headless CLI runner for single video jobs
```

## Setup and Installation

1. **Create a Python 3.11 or 3.12 virtual environment (Windows 10/11 64-bit):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. **Install dependencies:**

```powershell
# For NVIDIA CUDA 12.6 workstations:
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# Runtime only:
pip install -r requirements.txt

# Or full dev / export / PyInstaller packaging dependencies:
pip install -r requirements-dev.txt
```

*(For CPU-only machines without CUDA drivers, use `onnxruntime` instead of `onnxruntime-gpu`.)*

3. **Restore the ONNX model weights (`98.73 MiB`, excluded from Git):**
   - Obtain `container_detection_model.onnx` separately from your team's asset storage / GitHub Releases.
   - Place it at `container-detection-drone-view/weight/container_detection_model.onnx`.
   - Verify SHA-256 matches `container-detection-drone-view/weight/MANIFEST.json` (`BCAA5D96715268A32905227CE05CC328898E3DA957FCEFB7399019D8D70E795F`):

```powershell
Get-FileHash container-detection-drone-view\weight\container_detection_model.onnx -Algorithm SHA256
```

4. **Configure OCR service & credentials:**
   - Copy `config/config.example.json` to `config/config.json` and set `api.url` to your internal OCR service URL (keep `api.gate` as a numeric string such as `"1"`), or pass `OCR_API_URL` / `OCR_API_GATE` environment variables.
   - Launch the desktop GUI and authenticate via Login / Settings so the JWT token is stored encrypted in `data/.token.dpapi` (never commit tokens or `config/config.json`).

## Usage and Packaging

```powershell
# Launch desktop application
python main.py

# Run GPU / ONNX preflight check (writes self_check.json)
python main.py --self-check

# Run a video job via CLI without GUI
python scripts/run_video_job.py --video "path\to\video.mp4" --mode LEFT_RIGHT --name "SampleJob"

# Run QA checks (pass --require-dist after building dist/)
python scripts/qa_verification.py

# Build standalone executable into dist/HalotecContainerOCR/
python scripts/build_exe.py
```

Optional model training & ONNX export (see `container-detection-drone-view/trainer/`):

```powershell
python container-detection-drone-view\trainer\export.py -m "path\to\best.pt" -o "container-detection-drone-view\weight\container_detection_model.onnx" --imgsz 480 480 --opset 11 --overwrite --verify-infer
```

## Notes and Limitations

- **Windows Required:** `src/utils/security.py` (DPAPI via `crypt32.dll`) and `src/utils/windows_job.py` (`kernel32.dll`) target Windows 10/11.
- **Fixed ONNX Input Shape:** The default exported ONNX model expects `[1, 3, 480, 480]` FP32 inputs; re-export with `trainer/export.py --dynamic` if experimenting with variable inference sizes.
- **Packaging Notes:** Pass `--include-local-token` to `scripts/build_exe.py` only for local single-machine testing. Use `iscc installer.iss` with Inno Setup to compile `dist/HalotecContainerOCR/` into a Windows setup executable.
