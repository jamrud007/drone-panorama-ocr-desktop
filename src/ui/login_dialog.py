"""
Login Dialog for Halotec Container OCR Desktop Application.
Complies with TDD v1.9 & OCR_SERVICE_DOCUMENTATION.md:
- Supports 2 login methods:
  1. Akun: Username & Password (via POST /login)
  2. Token: Direct JWT Bearer Token (validated via GET /refresh_token)
- Securely stores active token in Windows DPAPI.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.config import AppConfig
from src.ocr_client.client import HalotecOcrClient, OcrClientError
from src.utils.security import store_auth_token


class LoginDialog(QDialog):
    def __init__(self, config: Optional[AppConfig] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config or AppConfig.load()
        self.auth_token: Optional[str] = None

        self.setWindowTitle("Masuk - Halotec Container OCR")
        self.setFixedSize(480, 580)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint)

        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit, QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #0078d4;
                background-color: #333333;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background-color: #252526;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #1e1e1e;
                color: #888888;
                padding: 8px 16px;
                font-weight: bold;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #252526;
                color: #0078d4;
                border-bottom: 2px solid #0078d4;
            }
            QPushButton#btn_primary {
                background-color: #0078d4;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                padding: 10px;
            }
            QPushButton#btn_primary:hover {
                background-color: #1084d8;
            }
            QPushButton#btn_primary:pressed {
                background-color: #006cc1;
            }
            QPushButton#btn_primary:disabled {
                background-color: #444444;
                color: #888888;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(12)

        # 1. Header Section
        lbl_brand = QLabel("HALOTEC")
        font_brand = QFont()
        font_brand.setPointSize(18)
        font_brand.setBold(True)
        lbl_brand.setFont(font_brand)
        lbl_brand.setStyleSheet("color: #0078d4; letter-spacing: 2px;")
        lbl_brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_brand)

        lbl_title = QLabel("Container OCR & Drone Vision System")
        lbl_title.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: 600;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        layout.addSpacing(5)

        # 2. Tab Widget for 2 Authentication Modes
        self.tabs = QTabWidget()

        # Tab 1: Username & Password
        tab_acc = QWidget()
        lay_acc = QVBoxLayout(tab_acc)
        lay_acc.setContentsMargins(15, 15, 15, 15)
        lay_acc.setSpacing(10)

        lay_acc.addWidget(QLabel("<b>Username / Email:</b>"))
        self.edit_username = QLineEdit()
        self.edit_username.setPlaceholderText("operator@example.com")
        self.edit_username.returnPressed.connect(self._on_login_account_clicked)
        lay_acc.addWidget(self.edit_username)

        lay_acc.addWidget(QLabel("<b>Password:</b>"))
        pwd_box = QHBoxLayout()
        pwd_box.setSpacing(6)
        self.edit_password = QLineEdit()
        self.edit_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit_password.setPlaceholderText("Masukkan kata sandi")
        self.edit_password.returnPressed.connect(self._on_login_account_clicked)
        pwd_box.addWidget(self.edit_password)

        self.btn_toggle_pwd = QPushButton("👁")
        self.btn_toggle_pwd.setCheckable(True)
        self.btn_toggle_pwd.setFixedWidth(40)
        self.btn_toggle_pwd.setStyleSheet("background-color: #333333; border: 1px solid #3e3e42; border-radius: 4px; padding: 6px; color: #aaa;")
        self.btn_toggle_pwd.toggled.connect(self._toggle_password_visibility)
        pwd_box.addWidget(self.btn_toggle_pwd)
        lay_acc.addLayout(pwd_box)

        self.btn_login_acc = QPushButton("Masuk dengan Akun")
        self.btn_login_acc.setObjectName("btn_primary")
        self.btn_login_acc.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_login_acc.clicked.connect(self._on_login_account_clicked)
        lay_acc.addSpacing(8)
        lay_acc.addWidget(self.btn_login_acc)

        self.tabs.addTab(tab_acc, "Akun (Username & Password)")

        # Tab 2: Direct JWT Token
        tab_tok = QWidget()
        lay_tok = QVBoxLayout(tab_tok)
        lay_tok.setContentsMargins(15, 15, 15, 15)
        lay_tok.setSpacing(10)

        lay_tok.addWidget(QLabel("<b>Paste JWT Bearer Token:</b>"))
        self.edit_token = QTextEdit()
        self.edit_token.setPlaceholderText("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        self.edit_token.setMaximumHeight(90)
        lay_tok.addWidget(self.edit_token)

        lbl_hint = QLabel("Gunakan token dari Swagger UI (/docs pada URL server API)")
        lbl_hint.setStyleSheet("color: #888888; font-size: 10px;")
        lay_tok.addWidget(lbl_hint)

        self.btn_login_tok = QPushButton("Masuk dengan Token")
        self.btn_login_tok.setObjectName("btn_primary")
        self.btn_login_tok.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_login_tok.clicked.connect(self._on_login_token_clicked)
        lay_tok.addSpacing(8)
        lay_tok.addWidget(self.btn_login_tok)

        self.tabs.addTab(tab_tok, "Token JWT (Langsung)")
        layout.addWidget(self.tabs)

        # 3. Server URL Indicator
        self.lbl_server = QLabel(f"Server: {self.config.api.url}")
        self.lbl_server.setStyleSheet("color: #666666; font-size: 10px;")
        self.lbl_server.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_server)

        # 4. Error Message Banner
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff5555; font-size: 11px; font-weight: 500;")
        self.lbl_error.setWordWrap(True)
        self.lbl_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_error.hide()
        layout.addWidget(self.lbl_error)

        # 5. Cancel Button
        self.btn_cancel = QPushButton("Batal & Keluar")
        self.btn_cancel.setStyleSheet("background: transparent; color: #888888; border: none; font-size: 12px; padding: 4px;")
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.clicked.connect(self.reject)
        layout.addWidget(self.btn_cancel)

        # Focus
        self.edit_username.setFocus()

    def _toggle_password_visibility(self, visible: bool) -> None:
        self.edit_password.setEchoMode(
            QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        )

    def _on_login_account_clicked(self) -> None:
        username = self.edit_username.text().strip()
        password = self.edit_password.text().strip()

        if not username:
            self._show_error("Silakan masukkan username atau email!")
            self.edit_username.setFocus()
            return

        if not password:
            self._show_error("Silakan masukkan kata sandi!")
            self.edit_password.setFocus()
            return

        self._show_loading(True, "Memverifikasi Kredensial...")
        QTimer.singleShot(50, lambda: self._execute_auth_account(username, password))

    def _execute_auth_account(self, username: str, password: str) -> None:
        client = HalotecOcrClient(
            base_url=self.config.api.url,
            gate=self.config.api.gate,
            timeout=15,
        )
        try:
            token = client.login(username=username, password=password)
            self.auth_token = token
            store_auth_token(token)
            self.accept()
        except OcrClientError as e:
            self._show_loading(False)
            self._show_error(str(e))
        except Exception as e:
            self._show_loading(False)
            self._show_error(f"Gagal menghubungi server: {e}")

    def _on_login_token_clicked(self) -> None:
        token = self.edit_token.toPlainText().strip()
        if not token:
            self._show_error("Silakan paste token JWT terlebih dahulu!")
            self.edit_token.setFocus()
            return

        self._show_loading(True, "Memvalidasi Token JWT...")
        QTimer.singleShot(50, lambda: self._execute_auth_token(token))

    def _execute_auth_token(self, token: str) -> None:
        client = HalotecOcrClient(
            base_url=self.config.api.url,
            gate=self.config.api.gate,
            timeout=15,
        )

        is_valid, msg, validated_token = client.validate_token(token)
        if is_valid and validated_token:
            self.auth_token = validated_token
            store_auth_token(validated_token)
            self.accept()
        else:
            self._show_loading(False)
            self._show_error(msg)

    def _show_loading(self, loading: bool, message: str = "Memproses...") -> None:
        if loading:
            self.btn_login_acc.setEnabled(False)
            self.btn_login_tok.setEnabled(False)
            self.lbl_error.setText(message)
            self.lbl_error.setStyleSheet("color: #0078d4; font-size: 11px;")
            self.lbl_error.show()
        else:
            self.btn_login_acc.setEnabled(True)
            self.btn_login_tok.setEnabled(True)
            self.btn_login_acc.setText("Masuk dengan Akun")
            self.btn_login_tok.setText("Masuk dengan Token")

    def _show_error(self, message: str) -> None:
        self.lbl_error.setText(message)
        self.lbl_error.setStyleSheet("color: #ff5555; font-size: 11px; font-weight: 500;")
        self.lbl_error.show()
