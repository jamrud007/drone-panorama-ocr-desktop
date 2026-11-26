from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


def format_ms(ms: int) -> str:
    """Format milliseconds into a safe mm:ss or hh:mm:ss string without overflow."""
    secs = max(0, ms) // 1000
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class VideoPlayerDialog(QDialog):
    """
    Modern, Robust Native Video Player Dialog for Drone & Result Videos.
    Engineered for zero-crash, safe multi-cycle open/close lifecycle on Windows:
      - Cleanly detaches QVideoWidget and QAudioOutput before destruction.
      - Disconnects all QtMultimedia signals on cleanup to prevent dangling WMF callbacks.
      - Handles QMediaPlayer errors gracefully with fallback to default Windows player.
      - Safe Spacebar Play/Pause, Seek slider, and mm:ss time display.
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        video_path: str = "",
        window_title: str = "Pemutar Video Drone",
    ) -> None:
        super().__init__(parent)
        self.video_path = str(Path(video_path).resolve())
        self.setWindowTitle(f"{window_title} - {Path(self.video_path).name}")
        self.resize(880, 560)
        self.setMinimumSize(640, 420)
        self._is_cleaned_up = False

        # Style dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #3d3d3d;
                border-color: #0078d4;
            }
            QPushButton:pressed {
                background-color: #0078d4;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #333333;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d4;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ffffff;
                border: 1px solid #777;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #0078d4;
            }
            QLabel {
                color: #cccccc;
                font-size: 11px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # 1. Video Display Widget
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 6px;")
        layout.addWidget(self.video_widget, stretch=1)

        # 2. Progress Slider and Time Label
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(4, 0, 4, 0)
        slider_layout.setSpacing(8)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_seek)
        slider_layout.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("00:00 / 00:00", self)
        slider_layout.addWidget(self.lbl_time)
        layout.addLayout(slider_layout)

        # 3. Control Buttons Toolbar
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(4, 0, 4, 0)
        controls_layout.setSpacing(8)

        self.btn_play = QPushButton("▶ Putar", self)
        self.btn_play.setFixedWidth(85)
        self.btn_play.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ Berhenti", self)
        self.btn_stop.setFixedWidth(85)
        self.btn_stop.clicked.connect(self._stop_playback)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addSpacing(16)

        # Path label
        self.lbl_file_info = QLabel(f"File: {Path(self.video_path).name}", self)
        self.lbl_file_info.setStyleSheet("color: #888888; font-size: 11px;")
        controls_layout.addWidget(self.lbl_file_info, stretch=1)

        # Button to open with default Windows player
        self.btn_open_external = QPushButton("🔗 Buka di Media Player Windows", self)
        self.btn_open_external.clicked.connect(self._open_in_external_player)
        controls_layout.addWidget(self.btn_open_external)

        layout.addLayout(controls_layout)

        # 4. Media Player Engine (Clean init)
        self.player: Optional[QMediaPlayer] = QMediaPlayer(self)
        self.audio: Optional[QAudioOutput] = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_player_error)

        # Space shortcut to Play/Pause
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._toggle_playback)

        # Load video source safely
        if Path(self.video_path).is_file():
            self.player.setSource(QUrl.fromLocalFile(self.video_path))
            self.player.play()
        else:
            self.lbl_file_info.setText(f"File tidak ditemukan: {Path(self.video_path).name}")
            self.lbl_file_info.setStyleSheet("color: #D83B01; font-weight: bold; font-size: 11px;")

    def _toggle_playback(self) -> None:
        if self.player is None:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _stop_playback(self) -> None:
        if self.player is not None:
            self.player.stop()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setText("⏸ Jeda")
        else:
            self.btn_play.setText("▶ Putar")

    def _on_position_changed(self, position: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        dur = self.player.duration() if self.player is not None else 0
        self.lbl_time.setText(f"{format_ms(position)} / {format_ms(dur)}")

    def _on_duration_changed(self, duration: int) -> None:
        self.slider.setRange(0, duration)
        pos = self.player.position() if self.player is not None else 0
        self.lbl_time.setText(f"{format_ms(pos)} / {format_ms(duration)}")

    def _on_seek(self, position: int) -> None:
        if self.player is not None:
            self.player.setPosition(position)

    def _on_player_error(self, error: QMediaPlayer.Error, error_string: str) -> None:
        print(f"[WARN] VideoPlayer error: {error} - {error_string}")
        self.lbl_file_info.setText(f"Peringatan Codec: Gunakan tombol 'Buka di Media Player Windows' ({error_string})")
        self.lbl_file_info.setStyleSheet("color: #D83B01; font-weight: bold; font-size: 11px;")

    def _open_in_external_player(self) -> None:
        if Path(self.video_path).is_file():
            try:
                os.startfile(self.video_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Gagal membuka pemutar video eksternal:\n{e}")

    def _cleanup(self) -> None:
        """Thoroughly detaches and cleans up native WMF / DirectX video surfaces to prevent crashes on close."""
        if self._is_cleaned_up:
            return
        self._is_cleaned_up = True

        if hasattr(self, "player") and self.player is not None:
            try:
                self.player.positionChanged.disconnect()
            except Exception:
                pass
            try:
                self.player.durationChanged.disconnect()
            except Exception:
                pass
            try:
                self.player.playbackStateChanged.disconnect()
            except Exception:
                pass
            try:
                self.player.errorOccurred.disconnect()
            except Exception:
                pass

            try:
                self.player.stop()
            except Exception:
                pass
            try:
                self.player.setVideoOutput(None)
            except Exception:
                pass
            try:
                self.player.setAudioOutput(None)
            except Exception:
                pass
            try:
                self.player.setSource(QUrl())
            except Exception:
                pass
            try:
                self.player.deleteLater()
            except Exception:
                pass
            self.player = None

        if hasattr(self, "audio") and self.audio is not None:
            try:
                self.audio.deleteLater()
            except Exception:
                pass
            self.audio = None

    def reject(self) -> None:
        self._cleanup()
        super().reject()

    def accept(self) -> None:
        self._cleanup()
        super().accept()

    def closeEvent(self, event) -> None:
        self._cleanup()
        super().closeEvent(event)

    def hideEvent(self, event) -> None:
        self._cleanup()
        super().hideEvent(event)
