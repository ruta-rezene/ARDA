"""
src/core/cads.py
Image loading and forensic feature extraction.

Loads an image and returns three independent input channels:
  pixels      -- float32 tensor (3,128,128) in [0,1]
  metadata    -- EXIF dict
  compression -- dict with real JPEG quantization signature

find_all_images() recursively locates every supported image under a
folder regardless of subfolder depth.
"""

import os
import io
import torch
from PIL import Image
from PIL.ExifTags import TAGS

IMG_SIZE = 128
_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def find_all_images(root_folder: str) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d != "__pycache__"]
        for fname in filenames:
            if os.path.splitext(fname.lower())[1] in _EXTS:
                found.append(os.path.join(dirpath, fname))
    return sorted(found)


def _quant_signature(pil_img) -> float:
    """
    Extract a real compression signature from the JPEG quantization table.
    Returns the mean of the luminance quantization table -- higher mean
    means heavier compression (lower quality). For non-JPEG or missing
    tables, re-encode at quality 90 to obtain a reference signature.
    """
    try:
        qt = getattr(pil_img, "quantization", None)
        if qt and 0 in qt:
            return float(sum(qt[0]) / len(qt[0]))
    except Exception:
        pass
    # No embedded table -- re-encode to derive one
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
        buf.seek(0)
        reopened = Image.open(buf)
        reopened.load()
        qt = getattr(reopened, "quantization", None)
        if qt and 0 in qt:
            return float(sum(qt[0]) / len(qt[0]))
    except Exception:
        pass
    return -1.0


def load_image_with_all_inputs(path: str) -> dict:
    pil_img = Image.open(path)
    pil_img.load()

    rgb     = pil_img.convert("RGB")
    resized = rgb.resize((IMG_SIZE, IMG_SIZE))
    # Build tensor directly from PIL bytes (no numpy->torch bridge, which
    # avoids NumPy 1.x/2.x ABI conflicts with the installed torch).
    pixels = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
    pixels = pixels.view(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1).float() / 255.0

    metadata = {}
    try:
        exif_raw = pil_img._getexif()
        if exif_raw:
            for tag_id, value in exif_raw.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                metadata[tag_name] = str(value)[:120]
    except Exception:
        pass

    compression = {
        "quant_signature": _quant_signature(pil_img),
        "format":          pil_img.format or "JPEG",
    }

    return {"pixels": pixels, "metadata": metadata, "compression": compression}


def quant_signature_from_tensor(pixels: torch.Tensor, quality: int = 90) -> float:
    """
    Re-encode a pixel tensor as JPEG at the given quality and read back the
    real quantization signature. This makes Red's compression attack genuine:
    a lower quality produces a measurably different quantization table.
    """
    # Build PIL image without .numpy() to avoid broken numpy-torch bridge
    t   = pixels.clamp(0, 1).permute(1, 2, 0)
    h, w = t.shape[0], t.shape[1]
    raw = bytes([int(round(v * 255)) for v in t.reshape(-1).tolist()])
    pil = Image.frombytes("RGB", (w, h), raw)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(max(1, min(100, quality))))
    buf.seek(0)
    reopened = Image.open(buf)
    reopened.load()
    qt = getattr(reopened, "quantization", None)
    if qt and 0 in qt:
        return float(sum(qt[0]) / len(qt[0]))
    return -1.0