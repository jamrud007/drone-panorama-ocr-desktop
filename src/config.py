"""
Application Configuration Loader (Option 2: Self-contained & Portable).
Resolves all paths relative to the executable (production) or repository root (development).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


def get_base_dir() -> Path:
    """Returns application root directory whether running as script or frozen PyInstaller .exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def get_bundle_dir() -> Path:
    """Returns PyInstaller bundle directory (_MEIPASS) or application root."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")).resolve()
    return get_base_dir()


BASE_DIR = get_base_dir()
BUNDLE_DIR = get_bundle_dir()


@dataclass
class ApiConfig:
    url: str = "https://your-ocr-service.example.com"
    gate: str = "1"
    timeout_seconds: int = 30
    max_retries: int = 3
    token_refresh_interval_hours: int = 12


@dataclass
class DetectionConfig:
    threshold_container: float = 0.40
    threshold_text: float = 0.25
    iou_threshold: float = 0.70
    imgsz_tracking: int = 480
    imgsz_text_detection: int = 480
    frame_sample_interval_seconds: float = 0.5
    model_path: str = "container-detection-drone-view/weight/container_detection_model.onnx"

    def resolved_model_path(self) -> Path:
        for root in [BASE_DIR, BASE_DIR / "_internal", BUNDLE_DIR, BASE_DIR.parent.parent]:
            candidate = (root / self.model_path).resolve()
            if candidate.is_file():
                return candidate
        return (BASE_DIR / self.model_path).resolve()


@dataclass
class StitchingConfig:
    pipeline_script: str = "container-detection-drone-view/pipeline_yolo_paronama.py"
    profile: str = "balanced"
    min_crops_per_track: int = 25
    device: str = "auto"

    def resolved_script_path(self) -> Path:
        for root in [BASE_DIR, BASE_DIR / "_internal", BUNDLE_DIR, BASE_DIR.parent.parent]:
            candidate = (root / self.pipeline_script).resolve()
            if candidate.is_file():
                return candidate
        return (BASE_DIR / self.pipeline_script).resolve()


@dataclass
class PathsConfig:
    output_dir: str = "outputs"
    data_dir: str = "data"
    logs_dir: str = "logs"

    def resolved_output_dir(self) -> Path:
        return (BASE_DIR / self.output_dir).resolve()

    def resolved_data_dir(self) -> Path:
        return (BASE_DIR / self.data_dir).resolve()

    def resolved_logs_dir(self) -> Path:
        return (BASE_DIR / self.logs_dir).resolve()


@dataclass
class AppConfig:
    auto_resume_interrupted: bool = True
    max_parallel_ocr_requests: int = 4
    video_resolution: str = "720P"
    api: ApiConfig = field(default_factory=ApiConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stitching: StitchingConfig = field(default_factory=StitchingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> AppConfig:
        if config_path is None:
            config_path = BASE_DIR / "config" / "config.json"

        if not config_path.exists():
            cfg = cls()
            cfg.save(config_path)
        else:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg = cls.from_dict(data)
            except Exception:
                cfg = cls()

        env_url = os.environ.get("OCR_API_URL", "").strip()
        if env_url:
            cfg.api.url = env_url
        env_gate = os.environ.get("OCR_API_GATE", "").strip()
        if env_gate:
            cfg.api.gate = env_gate
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AppConfig:
        api_data = data.get("api", {})
        det_data = data.get("detection", {})
        stitch_data = data.get("stitching", {})
        paths_data = data.get("paths", {})
        app_data = data.get("app", {})

        return cls(
            auto_resume_interrupted=app_data.get("auto_resume_interrupted", True),
            max_parallel_ocr_requests=app_data.get("max_parallel_ocr_requests", 4),
            video_resolution=str(app_data.get("video_resolution", "720P")).upper(),
            api=ApiConfig(
                url=api_data.get("url", "https://your-ocr-service.example.com"),
                gate=str(api_data.get("gate", "1")),
                timeout_seconds=int(api_data.get("timeout_seconds", 30)),
                max_retries=int(api_data.get("max_retries", 3)),
                token_refresh_interval_hours=int(api_data.get("token_refresh_interval_hours", 12)),
            ),
            detection=DetectionConfig(
                threshold_container=float(det_data.get("threshold_container", 0.40)),
                threshold_text=float(det_data.get("threshold_text", 0.25)),
                iou_threshold=float(det_data.get("iou_threshold", 0.70)),
                imgsz_tracking=int(det_data.get("imgsz_tracking", 480)),
                imgsz_text_detection=int(det_data.get("imgsz_text_detection", 480)),
                frame_sample_interval_seconds=float(det_data.get("frame_sample_interval_seconds", 0.5)),
                model_path=str(det_data.get("model_path", "container-detection-drone-view/weight/container_detection_model.onnx")),
            ),
            stitching=StitchingConfig(
                pipeline_script=str(stitch_data.get("pipeline_script", "container-detection-drone-view/pipeline_yolo_paronama.py")),
                profile=str(stitch_data.get("profile", "balanced")),
                min_crops_per_track=int(stitch_data.get("min_crops_per_track", 25)),
                device=str(stitch_data.get("device", "auto")),
            ),
            paths=PathsConfig(
                output_dir=str(paths_data.get("output_dir", "outputs")),
                data_dir=str(paths_data.get("data_dir", "data")),
                logs_dir=str(paths_data.get("logs_dir", "logs")),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api": asdict(self.api),
            "detection": asdict(self.detection),
            "stitching": asdict(self.stitching),
            "paths": asdict(self.paths),
            "app": {
                "auto_resume_interrupted": self.auto_resume_interrupted,
                "max_parallel_ocr_requests": self.max_parallel_ocr_requests,
                "video_resolution": getattr(self, "video_resolution", "720P"),
            },
        }

    def save(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = BASE_DIR / "config" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def ensure_directories(self) -> None:
        """Create required runtime folders if not existing."""
        (self.paths.resolved_output_dir() / "images").mkdir(parents=True, exist_ok=True)
        (self.paths.resolved_output_dir() / "labels").mkdir(parents=True, exist_ok=True)
        (self.paths.resolved_data_dir() / "temp_crops").mkdir(parents=True, exist_ok=True)
        self.paths.resolved_logs_dir().mkdir(parents=True, exist_ok=True)
