"""
Windows DPAPI (Data Protection API) Security Manager.
Provides zero-dependency encryption tied to the Windows User session via ctypes.windll.crypt32.
Complies with TDD v1.9 Rule 4.3 (No plaintext credentials in configs/git).
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, Structure, c_void_p, wintypes
from pathlib import Path
from typing import Optional

from src.config import BASE_DIR

TOKEN_STORAGE_PATH = BASE_DIR / "data" / ".token.dpapi"


class DATA_BLOB(Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", POINTER(ctypes.c_byte)),
    ]


crypt32 = ctypes.windll.crypt32
kernel32 = ctypes.windll.kernel32

CryptProtectData = crypt32.CryptProtectData
CryptProtectData.argtypes = [
    POINTER(DATA_BLOB),
    wintypes.LPCWSTR,
    POINTER(DATA_BLOB),
    c_void_p,
    c_void_p,
    wintypes.DWORD,
    POINTER(DATA_BLOB),
]
CryptProtectData.restype = wintypes.BOOL

CryptUnprotectData = crypt32.CryptUnprotectData
CryptUnprotectData.argtypes = [
    POINTER(DATA_BLOB),
    POINTER(wintypes.LPWSTR),
    POINTER(DATA_BLOB),
    c_void_p,
    c_void_p,
    wintypes.DWORD,
    POINTER(DATA_BLOB),
]
CryptUnprotectData.restype = wintypes.BOOL

LocalFree = kernel32.LocalFree
LocalFree.argtypes = [c_void_p]
LocalFree.restype = c_void_p

CRYPTPROTECT_UI_FORBIDDEN = 0x01


def dpapi_encrypt(plaintext: str) -> bytes:
    """Encrypts plaintext string using current Windows user credentials."""
    raw_bytes = plaintext.encode("utf-8")
    blob_in = DATA_BLOB(
        cbData=len(raw_bytes),
        pbData=ctypes.cast(ctypes.create_string_buffer(raw_bytes), POINTER(ctypes.c_byte)),
    )
    blob_out = DATA_BLOB()

    success = CryptProtectData(
        ctypes.byref(blob_in),
        "Halotec_OCR_Token",
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        buffer = (ctypes.c_byte * blob_out.cbData)()
        ctypes.memmove(buffer, blob_out.pbData, blob_out.cbData)
        return bytes(buffer)
    finally:
        LocalFree(blob_out.pbData)


def dpapi_decrypt(ciphertext: bytes) -> str:
    """Decrypts ciphertext using current Windows user credentials."""
    blob_in = DATA_BLOB(
        cbData=len(ciphertext),
        pbData=ctypes.cast(ctypes.create_string_buffer(ciphertext), POINTER(ctypes.c_byte)),
    )
    blob_out = DATA_BLOB()

    success = CryptUnprotectData(
        ctypes.byref(blob_in),
        None,
        None,
        None,
        None,
        CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        buffer = (ctypes.c_byte * blob_out.cbData)()
        ctypes.memmove(buffer, blob_out.pbData, blob_out.cbData)
        return bytes(buffer).decode("utf-8")
    finally:
        LocalFree(blob_out.pbData)


def is_jwt_format(token: Optional[str]) -> bool:
    """Checks if a string has the basic 3-part base64 dot-separated JWT structure."""
    if not token or not isinstance(token, str):
        return False
    clean = token.strip()
    if clean.lower().startswith("bearer "):
        clean = clean[7:].strip()
    clean = clean.strip("'\"")
    parts = clean.split(".")
    return len(parts) == 3 and all(len(p) > 0 for p in parts)


def store_auth_token(token: str, storage_path: Path = TOKEN_STORAGE_PATH) -> None:
    """Encrypts and persists the JWT auth token securely."""
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    encrypted_bytes = dpapi_encrypt(token.strip())
    with open(storage_path, "wb") as f:
        f.write(encrypted_bytes)


def retrieve_auth_token(storage_path: Path = TOKEN_STORAGE_PATH) -> Optional[str]:
    """Retrieves and decrypts the JWT auth token. Returns None if not stored."""
    if not storage_path.exists():
        return None
    try:
        with open(storage_path, "rb") as f:
            encrypted_bytes = f.read()
        if not encrypted_bytes:
            return None
        return dpapi_decrypt(encrypted_bytes).strip()
    except Exception:
        return None


def clear_auth_token(storage_path: Path = TOKEN_STORAGE_PATH) -> None:
    """Deletes stored encrypted credentials."""
    if storage_path.exists():
        try:
            storage_path.unlink()
        except OSError:
            pass
