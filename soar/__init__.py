"""
SOAR: High-Performance Resolution-Preserving Segmentation Engine
Pure C++20 / Vulkan Compute runtime with lightweight Python wrapper.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np

__version__ = "1.0.0"

def _find_library() -> str:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "build" / "libsoar_engine.dll",
        root / "build" / "soar_engine.dll",
        root / "build" / "libsoar_engine.so",
        root / "build" / "libsoar_engine.dylib",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError(
        "Could not find native soar_engine shared library in build/. "
        "Please build the project first using CMake."
    )

_LIB_PATH = _find_library()

if sys.platform == "win32":
    lib_dir = Path(_LIB_PATH).parent.resolve()
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(lib_dir))
        except Exception:
            pass
        for path_entry in os.environ.get("PATH", "").split(os.pathsep):
            if path_entry and os.path.isdir(path_entry):
                try:
                    os.add_dll_directory(path_entry)
                except Exception:
                    pass

_lib = ctypes.CDLL(_LIB_PATH)

# Setup C-API function signatures
_lib.soar_create_model.restype = ctypes.c_void_p
_lib.soar_create_model.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

_lib.soar_destroy_model.restype = None
_lib.soar_destroy_model.argtypes = [ctypes.c_void_p]

_lib.soar_get_parameter_count.restype = ctypes.c_size_t
_lib.soar_get_parameter_count.argtypes = [ctypes.c_void_p]

_lib.soar_save_weights.restype = ctypes.c_int
_lib.soar_save_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.soar_load_weights.restype = ctypes.c_int
_lib.soar_load_weights.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.soar_train_step.restype = ctypes.c_int
_lib.soar_train_step.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]

_lib.soar_predict.restype = ctypes.c_int
_lib.soar_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_char_p,
    ctypes.c_size_t,
]

_lib.soar_encode_rle.restype = ctypes.c_int
_lib.soar_encode_rle.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_char_p,
    ctypes.c_size_t,
]

VARIANT_MAP = {
    "nano": 0, "n": 0,
    "small": 1, "s": 1,
    "medium": 2, "m": 2,
    "large": 3, "l": 3,
    "xlarge": 4, "x": 4
}

class SOAR:
    """High-level Python wrapper executing 100% on the native C++20 engine."""

    def __init__(self, in_channels: int = 1, num_classes: int = 1, variant: str = "nano"):
        v_idx = VARIANT_MAP.get(variant.lower(), 0)
        self.handle = _lib.soar_create_model(in_channels, num_classes, v_idx)
        if not self.handle:
            raise RuntimeError("Failed to create native C++ SOAR model")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.variant = variant

    def __del__(self):
        if getattr(self, "handle", None):
            _lib.soar_destroy_model(self.handle)
            self.handle = None

    @property
    def parameter_count(self) -> int:
        return _lib.soar_get_parameter_count(self.handle)

    def save_weights(self, path: str):
        res = _lib.soar_save_weights(self.handle, path.encode("utf-8"))
        if res != 0:
            raise RuntimeError(f"Failed to save weights to {path}")

    def load_weights(self, path: str):
        res = _lib.soar_load_weights(self.handle, path.encode("utf-8"))
        if res != 0:
            raise RuntimeError(f"Failed to load weights from {path}")

    def train_step(self, image: np.ndarray, mask: np.ndarray, lr: float = 1e-3) -> Dict[str, float]:
        """Execute a full C++ training step with autograd, loss, and AdamW update."""
        image = np.ascontiguousarray(image, dtype=np.float32)
        mask = np.ascontiguousarray(mask, dtype=np.float32)

        c = image.shape[0] if image.ndim == 3 else 1
        h, w = image.shape[-2], image.shape[-1]

        c_loss = ctypes.c_float(0)
        c_dice = ctypes.c_float(0)
        c_iou = ctypes.c_float(0)

        img_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        msk_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        res = _lib.soar_train_step(
            self.handle, img_ptr, msk_ptr, c, h, w, lr,
            ctypes.byref(c_loss), ctypes.byref(c_dice), ctypes.byref(c_iou)
        )
        if res != 0:
            raise RuntimeError("Native C++ train_step failed")

        return {"loss": c_loss.value, "dice": c_dice.value, "iou": c_iou.value}

    def predict(self, image: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, str]:
        """Execute native C++ inference and return (probabilities, binary_mask, rle_string)."""
        image = np.ascontiguousarray(image, dtype=np.float32)
        c = image.shape[0] if image.ndim == 3 else 1
        h, w = image.shape[-2], image.shape[-1]

        probs = np.zeros((1, h, w), dtype=np.float32)
        mask = np.zeros((h, w), dtype=np.uint8)
        rle_buf = ctypes.create_string_buffer(h * w * 2)

        img_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        probs_ptr = probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        mask_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        res = _lib.soar_predict(
            self.handle, img_ptr, c, h, w, threshold,
            probs_ptr, mask_ptr, rle_buf, len(rle_buf)
        )
        if res != 0:
            raise RuntimeError("Native C++ predict failed")

        return probs, mask, rle_buf.value.decode("utf-8")

__all__ = ["SOAR", "__version__"]
