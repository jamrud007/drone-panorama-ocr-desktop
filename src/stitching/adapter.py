"""
Stitching Subprocess Adapter.
Executes container-detection-drone-view/pipeline_yolo_paronama.py inside a Windows Job Object.
Complies with TDD v1.9:
- Subprocess lifecycle managed by Windows Job Object (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE).
- Progress parsing for live UI tracking.
- Output directory resolution.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional

from src.config import BASE_DIR, BUNDLE_DIR, AppConfig
from src.utils.windows_job import WindowsJobObject


class StitchingProcessError(Exception):
    pass


class StitchingAdapter:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()
        self.job_object = WindowsJobObject()
        self.active_process: Optional[subprocess.Popen] = None

    def run_stitching(
        self,
        video_path: Path | str,
        camera_position: str,  # "LEFT_RIGHT", "TOP", "FRONT_REAR"
        output_dir: Optional[Path | str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        on_tracks_ready: Optional[Callable[[List[Any]], None]] = None,
        on_panorama_ready: Optional[Callable[[Any], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
    ) -> List[Path]:
        """
        Executes the stitching pipeline directly in-process (or fallback as managed subprocess).
        Returns a list of generated region JSON label files.
        """
        video = Path(video_path).resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Source video not found: {video}")

        model_path = self.config.detection.resolved_model_path()
        target_output_dir = Path(output_dir).resolve() if output_dir else self.config.paths.resolved_output_dir()
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Primary: In-Process Direct Execution (Unified Single Application)
        try:
            for candidate_root in [BASE_DIR, BUNDLE_DIR, BASE_DIR.parent.parent]:
                c_dir = candidate_root / "container-detection-drone-view"
                if c_dir.is_dir() and str(c_dir) not in sys.path:
                    sys.path.insert(0, str(c_dir))

            from pipeline_yolo_paronama import run_stitching_pipeline

            if progress_callback:
                progress_callback("Tahap 1/4: Memulai inisialisasi stitching...", 0.05)

            json_files = run_stitching_pipeline(
                source=video,
                model=model_path,
                camera_position=camera_position,
                output_dir=target_output_dir,
                profile=self.config.stitching.profile,
                conf=self.config.detection.threshold_container,
                text_detect_conf=self.config.detection.threshold_text,
                text_detect_imgsz=self.config.detection.imgsz_text_detection,
                imgsz=self.config.detection.imgsz_tracking,
                min_crops_per_track=self.config.stitching.min_crops_per_track,
                device=self.config.stitching.device,
                progress_callback=progress_callback,
                on_tracks_ready=on_tracks_ready,
                on_panorama_ready=on_panorama_ready,
                is_cancelled_callback=is_cancelled_callback,
            )

            if progress_callback:
                progress_callback("Tahap 3/4: Stitching & deteksi teks selesai.", 0.85)

            return json_files

        except ImportError:
            # Fallback to subprocess if direct import is unavailable
            script_path = self.config.stitching.resolved_script_path()
            if not script_path.is_file():
                raise FileNotFoundError(f"Stitching pipeline script not found: {script_path}")
            return self._run_stitching_subprocess(
                video=video,
                script_path=script_path,
                model_path=model_path,
                camera_position=camera_position,
                output_dir=target_output_dir,
                progress_callback=progress_callback,
                on_tracks_ready=on_tracks_ready,
                on_panorama_ready=on_panorama_ready,
            )

    def _run_stitching_subprocess(
        self,
        video: Path,
        script_path: Path,
        model_path: Path,
        camera_position: str,
        output_dir: Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        on_tracks_ready: Optional[Callable[[List[Any]], None]] = None,
        on_panorama_ready: Optional[Callable[[Any], None]] = None,
    ) -> List[Path]:
        if getattr(sys, "frozen", False):
            import shutil
            python_exe = shutil.which("python") or "python"
        else:
            python_exe = sys.executable

        total_video_frames = 0
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        except Exception:
            pass

        cmd = [
            python_exe,
            str(script_path),
            "--source",
            str(video),
            "--model",
            str(model_path),
            "--camera-position",
            camera_position,
            "--output-dir",
            str(output_dir),
            "--profile",
            self.config.stitching.profile,
            "--conf",
            str(self.config.detection.threshold_container),
            "--text-detect-conf",
            str(self.config.detection.threshold_text),
            "--text-detect-imgsz",
            str(self.config.detection.imgsz_text_detection),
            "--imgsz",
            str(self.config.detection.imgsz_tracking),
            "--min-crops-per-track",
            str(self.config.stitching.min_crops_per_track),
            "--device",
            self.config.stitching.device,
        ]

        if progress_callback:
            progress_callback(
                f"Tahap 1/4: Memulai tracking video ({total_video_frames} frames)...",
                0.05,
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        working_dir = script_path.parent.parent if script_path.is_file() else BASE_DIR

        self.active_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(working_dir),
        )

        try:
            self.job_object.assign_process(self.active_process._handle)  # type: ignore
        except Exception:
            pass

        frame_pattern = re.compile(r"frame\s+(\d+)(?:/(\d+))?:\s*(\d+)\s+track", re.IGNORECASE)
        track_pattern = re.compile(r"\[(\d+)/(\d+)\] Track (\d+):")
        det_pattern = re.compile(r"\[DETECTION\] region_(\d+):")
        import time
        start_time = time.time()

        total_tracks = 1
        current_track = 0
        import collections
        output_history = collections.deque(maxlen=30)

        assert self.active_process.stdout is not None
        for line in self.active_process.stdout:
            line_str = line.strip()
            if not line_str:
                continue
            output_history.append(line_str)

            if line_str.startswith("EVENT_TRACKS_READY:"):
                try:
                    payload = json.loads(line_str.split(":", 1)[1].strip())
                    if on_tracks_ready:
                        on_tracks_ready(payload)
                except Exception:
                    pass
                continue

            if line_str.startswith("EVENT_PANORAMA_READY:"):
                try:
                    payload = json.loads(line_str.split(":", 1)[1].strip())
                    if on_panorama_ready:
                        on_panorama_ready(payload)
                except Exception:
                    pass
                continue

            frame_match = frame_pattern.search(line_str)
            if frame_match:
                curr_frame = int(frame_match.group(1))
                if frame_match.group(2):
                    total_video_frames = int(frame_match.group(2))
                track_count = int(frame_match.group(3))
                elapsed = max(1.0, time.time() - start_time)
                rate = curr_frame / elapsed
                if total_video_frames > 0:
                    rem_frames = max(0, total_video_frames - curr_frame)
                    eta_sec = rem_frames / max(0.1, rate)
                    eta_str = f"~{int(eta_sec // 60)}m {int(eta_sec % 60):02d}s"
                    pct = 0.05 + min(0.35, (curr_frame / total_video_frames) * 0.35)
                    msg = f"Tahap 1/4: Tracking frame {curr_frame}/{total_video_frames} ({track_count} track, Est: {eta_str})"
                else:
                    pct = min(0.38, 0.05 + curr_frame * 0.0002)
                    msg = f"Tahap 1/4: Tracking frame {curr_frame} ({track_count} kontainer)"
                if progress_callback:
                    progress_callback(msg, pct)
                continue

            if "[DEVICE]" in line_str:
                if progress_callback:
                    progress_callback(line_str, 0.05)
                continue

            if "1. YOLO Tracking" in line_str or "1. Tracking" in line_str:
                if progress_callback:
                    progress_callback("Tahap 1/4: Memulai deteksi dan tracking kontainer...", 0.05)
                continue

            if "2. Tracking selesai" in line_str:
                if progress_callback:
                    progress_callback("Tahap 1/4: Tracking selesai. Menyiapkan estimasi lintasan...", 0.40)
                continue

            if "3. Stitching ALIKED" in line_str:
                if progress_callback:
                    progress_callback("Tahap 2/4: Memulai penyambungan panorama citra (ALIKED + LightGlue)...", 0.42)
                continue

            track_match = track_pattern.search(line_str)
            if track_match:
                current_track = int(track_match.group(1))
                total_tracks = max(1, int(track_match.group(2)))
                pct = 0.42 + (current_track / total_tracks) * 0.33
                if progress_callback:
                    progress_callback(f"Tahap 2/4: Stitching kontainer {current_track}/{total_tracks}", pct)
                continue

            if "4. Deteksi iso_code" in line_str:
                if progress_callback:
                    progress_callback("Tahap 3/4: Mendeteksi ROI nomor kontainer & ISO code pada citra...", 0.75)
                continue

            det_match = det_pattern.search(line_str)
            if det_match:
                reg_num = int(det_match.group(1))
                pct = min(0.85, 0.75 + reg_num * 0.012)
                if progress_callback:
                    progress_callback(f"Tahap 3/4: Deteksi ROI teks region {reg_num}", pct)
                continue

            if "region JSON" in line_str and progress_callback:
                progress_callback("Tahap 3/4: Menyelesaikan dokumen label spasial...", 0.85)

        return_code = self.active_process.wait()
        self.active_process = None

        if return_code != 0:
            last_err = "\n".join(output_history)
            raise StitchingProcessError(f"Stitching process failed with exit code {return_code}:\n{last_err}")

        if progress_callback:
            progress_callback("Tahap 3/4: Stitching & deteksi teks selesai.", 0.85)

        labels_dir = output_dir / camera_position / "labels"
        if not labels_dir.exists():
            labels_dir = output_dir / "labels"

        return sorted(labels_dir.glob(f"{video.stem}_region_*.json"))

    def terminate(self) -> None:
        """Kills the running subprocess if any."""
        if self.active_process and self.active_process.poll() is None:
            try:
                self.active_process.terminate()
                self.active_process.wait(timeout=3)
            except Exception:
                self.active_process.kill()
            finally:
                self.active_process = None
