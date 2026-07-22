"""
src/agents/blue.py
Agentic Blue defender.

Vision Transformer for pixel-level detection plus two forensic checks
(metadata, compression). Any channel firing -> TAMPERED.

Key design points that make the arms race real (not a 100%-catch wall):
  * Metadata check fires only on genuinely detectable forgery signals.
    Once Red forges realistic EXIF (Skill 3), metadata stops catching it,
    so the pixel ViT must carry detection -- that is where co-evolution
    actually happens.
  * Compression check compares REAL quantization signatures (mean of the
    JPEG luminance quant table) against the clean reference, with a margin.
  * Blue keeps a bounded replay buffer of recent rounds so the ViT trains
    on many images per step (mentor advice), not just the latest two.

Skill 1  Adaptive threshold from recent FP/FN rates (mean-reverting).
Skill 2  ViT training on a replay buffer each round.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from datetime import datetime as dt

IMG_SIZE   = 128
PATCH_SIZE = 8
N_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2
PATCH_DIM  = PATCH_SIZE * PATCH_SIZE * 3

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

PRETRAINED_PATH = "models/blue_pretrained.pt"
TRAINED_PATH    = "models/blue_trained.pt"


# ---- preprocessing ----------------------------------------------------------

import torch.nn.functional as F

_BLUR_K = torch.ones(3, 1, 3, 3) / 9.0


def high_pass_residual(pixels: torch.Tensor) -> torch.Tensor:
    """
    Forensic residual: image minus its local mean (a 3x3 blur). Adversarial
    perturbations and re-encode artifacts live in the high-frequency band,
    so this residual makes tampering far more separable than raw pixels --
    the same principle behind SRM features in image forensics / steganalysis.
    Amplified so the signal sits in a good range for the ViT.
    """
    blur = F.conv2d(pixels.unsqueeze(0), _BLUR_K, padding=1, groups=3)[0]
    return (pixels - blur) * 8.0


def normalize(pixels: torch.Tensor) -> torch.Tensor:
    return (pixels - MEAN[:, None, None]) / STD[:, None, None]


def apply_clahe(pixels: torch.Tensor) -> torch.Tensor:
    # Convert tensor -> uint8 numpy array WITHOUT torch's .numpy() bridge
    # (that bridge is what breaks under a NumPy version mismatch). We go
    # through a raw byte buffer, which numpy can wrap safely.
    import numpy as _np
    t    = pixels.clamp(0, 1).permute(1, 2, 0).contiguous()
    h, w = t.shape[0], t.shape[1]
    raw  = bytes(int(v * 255 + 0.5) for v in t.reshape(-1).tolist())
    img  = _np.frombuffer(raw, dtype=_np.uint8).reshape(h, w, 3)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (4, 4)).apply(lab[:, :, 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # uint8 HxWx3

    # numpy uint8 -> torch tensor, again via a byte buffer (no .numpy bridge)
    flat = torch.frombuffer(bytearray(out.tobytes()), dtype=torch.uint8)
    return flat.view(h, w, 3).permute(2, 0, 1).float() / 255.0


def preprocess(pixels: torch.Tensor) -> torch.Tensor:
    # The ViT sees the forensic high-pass residual, not the raw image.
    # Subtle perturbations invisible in raw pixels are strongly separable
    # in the high-frequency residual (SRM principle from image forensics).
    return high_pass_residual(pixels)


# ---- ViT --------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.contiguous().view(B, C, -1, PATCH_SIZE * PATCH_SIZE)
        x = x.permute(0, 2, 1, 3).contiguous().view(B, -1, PATCH_DIM)
        return self.proj(x)


class BlueViT(nn.Module):
    def __init__(self, dim=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbed(dim)
        self.cls   = nn.Parameter(torch.randn(1, 1, dim))
        self.pos   = nn.Parameter(torch.randn(1, N_PATCHES + 1, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout,
                                         batch_first=True)
        self.tf   = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x):
        B = x.shape[0]
        p = self.patch(x)
        t = torch.cat([self.cls.expand(B, -1, -1), p], 1) + self.pos
        return self.head(self.norm(self.tf(t)[:, 0]))


# ---- agent ------------------------------------------------------------------

class BlueAgent:

    THRESHOLD_MIN = 0.30
    THRESHOLD_MAX = 0.70
    STEP          = 0.02
    WINDOW        = 6
    FP_TOL        = 0.34
    FN_TOL        = 0.34
    BUFFER_MAX    = 24            # replay buffer -- many images per train step

    def __init__(self, threshold_start=0.50, metadata_stripped=False,
                 clean_quant_ref=None):
        self.threshold          = threshold_start
        self.metadata_stripped  = metadata_stripped
        self.clean_quant_ref    = clean_quant_ref     # reference signature
        self.vit                = BlueViT()
        self.optimizer          = optim.AdamW(self.vit.parameters(),
                                              lr=5e-4, weight_decay=1e-4)
        self.loss_fn            = nn.CrossEntropyLoss()
        self.buffer             = []
        self.history            = []
        self.detection_count    = 0
        self.training_steps     = 0

    # ---- perceive: three channels --------------------------------------

    def _pixel_confidence(self, pixels):
        x = preprocess(pixels)
        self.vit.eval()
        with torch.no_grad():
            p = torch.softmax(self.vit(x.unsqueeze(0)), 1)[0]
        return p[1].item()

    def _metadata_flag(self, metadata: dict) -> bool:
        # Signal 1: an editing-software tag is present (naive forgery)
        if "Software" in metadata:
            return True
        # Signal 2: freshly-stamped capture time (< 24h) -- implausible for lab data
        if "DateTime" in metadata:
            try:
                age = (dt.now() - dt.strptime(metadata["DateTime"],
                       "%Y:%m:%d %H:%M:%S")).total_seconds() / 3600
                if age < 24:
                    return True
            except ValueError:
                pass
        # Signal 3: camera claimed but implausibly few EXIF fields
        has_cam = "Make" in metadata or "Model" in metadata
        if has_cam and len(metadata) < 6:
            return True
        return False

    def _compression_flag(self, comp: dict) -> bool:
        if self.clean_quant_ref is None or self.clean_quant_ref <= 0:
            return False
        sig = comp.get("quant_signature", -1)
        if sig <= 0:
            return False
        # Heavier compression => higher quantization mean. Flag a clear jump.
        # Red can evade by re-encoding near the original quality (Skill 4),
        # so this is a genuine independent forensic signal, not an epsilon proxy.
        return (sig - self.clean_quant_ref) > (self.clean_quant_ref * 0.35)

    # ---- decide --------------------------------------------------------

    def detect(self, inputs: dict) -> dict:
        conf   = self._pixel_confidence(inputs["pixels"])
        px     = conf > self.threshold
        meta   = self._metadata_flag(inputs["metadata"])
        comp   = self._compression_flag(inputs["compression"])
        verdict = "TAMPERED" if (px or meta or comp) else "CLEAN"
        self.detection_count += 1
        return {"verdict": verdict, "pixel_flag": px,
                "pixel_confidence": round(conf, 3),
                "metadata_flag": meta, "compression_flag": comp,
                "threshold_used": round(self.threshold, 3)}

    # ---- train (replay buffer) -----------------------------------------

    def collect(self, pixels, ground_truth):
        x = preprocess(pixels)
        y = 0 if ground_truth == "CLEAN" else 1
        self.buffer.append((x, y))
        if len(self.buffer) > self.BUFFER_MAX:
            self.buffer.pop(0)

    def train_round(self, n_steps=8):
        if len(self.buffer) < 4:
            return
        imgs = torch.stack([b[0] for b in self.buffer])
        lbls = torch.tensor([b[1] for b in self.buffer], dtype=torch.long)
        self.vit.train()
        for _ in range(n_steps):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.vit(imgs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(self.vit.parameters(), 1.0)
            self.optimizer.step()
            self.training_steps += 1

    # ---- update threshold (Skill 1) ------------------------------------

    def update(self, ground_truth, verdict):
        self.history.append({"gt": ground_truth, "verdict": verdict})
        if len(self.history) < self.WINDOW:
            return
        recent = self.history[-self.WINDOW:]
        fp = sum(1 for h in recent if h["gt"] == "CLEAN" and h["verdict"] == "TAMPERED")
        fn = sum(1 for h in recent if h["gt"] == "TAMPERED" and h["verdict"] == "CLEAN")
        fpr, fnr = fp / self.WINDOW, fn / self.WINDOW
        if fpr > self.FP_TOL:
            self.threshold = min(self.THRESHOLD_MAX, self.threshold + self.STEP)
        elif fnr > self.FN_TOL:
            self.threshold = max(self.THRESHOLD_MIN, self.threshold - self.STEP)
        else:
            if self.threshold > 0.52:
                self.threshold = max(0.50, self.threshold - self.STEP * 0.3)
            elif self.threshold < 0.48:
                self.threshold = min(0.50, self.threshold + self.STEP * 0.3)

    def status(self):
        recent = self.history[-self.WINDOW:] if self.history else []
        fp = sum(1 for h in recent if h["gt"] == "CLEAN" and h["verdict"] == "TAMPERED")
        fn = sum(1 for h in recent if h["gt"] == "TAMPERED" and h["verdict"] == "CLEAN")
        return {"threshold": round(self.threshold, 3),
                "training_steps": self.training_steps,
                "recent_fp_rate": round(fp / self.WINDOW, 2) if recent else "n/a",
                "recent_fn_rate": round(fn / self.WINDOW, 2) if recent else "n/a"}

    # ---- persistence ---------------------------------------------------

    def save_weights(self, path=TRAINED_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.vit.state_dict(), path)

    def load_weights(self, path=None):
        for p in ([path] if path else []) + [PRETRAINED_PATH, TRAINED_PATH]:
            if p and os.path.exists(p):
                self.vit.load_state_dict(torch.load(p, map_location="cpu"))
                self.vit.eval()
                return p
        return None