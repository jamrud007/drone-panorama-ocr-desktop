# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec for Halotec Container OCR Single Unified Windows Desktop Application.
Bundles PySide6 UI, Local ONNX / YOLO detector, Drone Video Stitching (LightGlue/ALIKED),
and Halotec OCR Cloud Client into a single self-contained application.
"""
import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

block_cipher = None
APP_ROOT = Path(os.getcwd())

_drone_dir = APP_ROOT / "container-detection-drone-view"
datas = [
    (str(APP_ROOT / "config" / "config.json"), "config"),
    (str(_drone_dir / "pipeline_yolo_paronama.py"), "container-detection-drone-view"),
    (str(_drone_dir / "hyper_config"), "container-detection-drone-view/hyper_config"),
    (str(_drone_dir / "weight" / "container_detection_model.onnx"), "container-detection-drone-view/weight"),
]

binaries = []
hiddenimports = [
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "onnxruntime",
    "cv2",
    "numpy",
    "requests",
    "sqlite3",
    "torch",
    "torchvision",
    "kornia",
    "lap",
    "tqdm",
    "PIL",
    "scipy",
    "pipeline_yolo_paronama",
    "src.orchestrator.annotation_motion",
]

# Collect ultralytics, lightglue, kornia & onnxruntime (including DirectML.dll or CUDA EP) binaries and data
for pkg in ["ultralytics", "lightglue", "kornia", "onnxruntime"]:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

# Keep torch/lib pure (matched CUDA 12.6 + cuDNN 9.10 DLLs collected by PyInstaller hook-torch);
# mixing newer nvidia-cudnn-cu12 9.28 DLLs into torch/lib causes WinError 127 on import torch.

# Keep unittest: torch/_guards.py imports unittest.mock at module import time
excludes = [
    "tkinter",
    "IPython",
    "jupyter",
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt6",
    "PySide2",
    "plotly",
    "yt_dlp",
    "pandas",
    "pyarrow",
    "websockets",
    "curl_cffi",
    "matplotlib",
    "polars",
    "_polars_runtime_32",
]

a = Analysis(
    ["main.py"],
    pathex=[str(APP_ROOT), str(APP_ROOT / "container-detection-drone-view")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HalotecContainerOCR",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # Windowed GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HalotecContainerOCR",
)
