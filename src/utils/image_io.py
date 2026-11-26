"""
Unicode-safe image I/O utility for Windows.
Avoids cv2.imread / cv2.imwrite silent failures on Windows paths containing
spaces, non-ASCII characters, or long Unicode sequences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread_unicode(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Reads an image from a path safely on Windows, supporting Unicode characters.
    Returns np.ndarray or None if failed.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        return None

    try:
        # np.fromfile handles Unicode Windows paths seamlessly
        file_bytes = np.fromfile(str(path_obj), dtype=np.uint8)
        if file_bytes.size == 0:
            return None
        return cv2.imdecode(file_bytes, flags)
    except Exception:
        return None


def imwrite_unicode(path: Path | str, img: np.ndarray, params: Optional[list] = None) -> bool:
    """
    Writes an image to a path safely on Windows, supporting Unicode characters.
    Automatically creates parent directories.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return False

    path_obj = Path(path)
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        ext = path_obj.suffix if path_obj.suffix else ".jpg"
        success, encoded_buf = cv2.imencode(ext, img, params)
        if not success:
            return False
        encoded_buf.tofile(str(path_obj))
        return True
    except Exception:
        return False
