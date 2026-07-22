"""
src/agents/red.py
Agentic Red attacker.

Attacks three INDEPENDENT channels each round:
  1. Pixel       structured perturbation of magnitude epsilon along root
                 branches (Canny-guided). The channel the ViT must learn.
  2. Metadata    forged EXIF; Red removes signals that get caught (Skill 2/3).
  3. Compression genuine JPEG re-encode. Independent of epsilon: Red chooses a
                 re-encode quality and LEARNS to raise it toward the original
                 (evading the compression check) when that check keeps
                 catching it (Skill 4).

Skills:
  Skill 1  Epsilon control. Reacts to the PIXEL channel specifically and
           probes under a persona stealth band (no flatlining at the floor).
  Skill 2  Software-tag removal after repeated metadata catches (MITRE T1070).
  Skill 3  Realistic-EXIF forgery after continued metadata catches.
  Skill 4  Compression-quality lift when the compression channel catches Red.
"""

import random
import datetime
import torch
import cv2

from src.core.cads import quant_signature_from_tensor

PERSONAS = {
    "apt":           {"evasion": 0.9, "damage": 0.1},
    "cybercriminal": {"evasion": 0.5, "damage": 0.5},
    "hacktivist":    {"evasion": 0.1, "damage": 0.9},
}

_DYNAMICS = {
    "apt":           {"up": 0.010, "down": 0.006, "band": 0.06},
    "cybercriminal": {"up": 0.014, "down": 0.004, "band": 0.13},
    "hacktivist":    {"up": 0.022, "down": 0.002, "band": 0.20},
}

_MAKES    = ["NIKON CORPORATION", "Canon", "SONY", "FUJIFILM"]
_MODELS   = ["NIKON D850", "Canon EOS 5D Mark IV", "ILCE-7RM4", "X-T4"]
_SOFTWARE = ["Adobe Photoshop 24.0", "GIMP 2.10", "ImageJ", "darktable 4.0"]


class RedAgent:

    EPSILON_MIN = 0.01
    EPSILON_MAX = 0.20

    def __init__(self, epsilon_start=0.05, persona="cybercriminal"):
        if persona not in PERSONAS:
            persona = "cybercriminal"
        self.persona = persona
        self.weights = PERSONAS[persona]
        self.dyn     = _DYNAMICS[persona]
        self.epsilon = min(epsilon_start, self.dyn["band"])
        self.history = []
        self.attack_count = 0

        self.use_software_tag    = True
        self.meta_caught_streak  = 0
        self.use_realistic_exif  = False
        self.post_sw_meta_streak = 0

        self.reencode_quality = 82

    def perceive(self, verdict):
        return verdict == "TAMPERED"

    def _pixel_attack(self, pixels):
        import numpy as _np
        t = pixels.clamp(0, 1).permute(1, 2, 0).contiguous()
        h, w = t.shape[0], t.shape[1]
        raw = bytes(int(v * 255 + 0.5) for v in t.reshape(-1).tolist())
        img = _np.frombuffer(raw, dtype=_np.uint8).reshape(h, w, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        mask = cv2.dilate(edges, _np.ones((3, 3), _np.uint8), iterations=1)
        mask_t = (torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8)
                  .view(h, w).float() / 255.0).unsqueeze(0)
        branch = (torch.rand_like(pixels) * 2 - 1) * self.epsilon
        bg     = (torch.rand_like(pixels) * 2 - 1) * (self.epsilon * 0.15)
        return torch.clamp(pixels + branch * mask_t + bg, 0.0, 1.0)

    def _forge_metadata(self):
        if self.use_realistic_exif:
            days = random.randint(30, 400)
            date = (datetime.datetime.now() -
                    datetime.timedelta(days=days)).strftime("%Y:%m:%d %H:%M:%S")
            meta = {"Make": random.choice(_MAKES), "Model": random.choice(_MODELS),
                    "DateTime": date, "DateTimeOriginal": date,
                    "DateTimeDigitized": date,
                    "ExposureTime": f"1/{random.randint(60,500)}",
                    "FNumber": str(round(random.uniform(2.0, 11.0), 1)),
                    "ISOSpeedRatings": str(random.choice([100, 200, 400, 800])),
                    "FocalLength": str(random.choice([24, 35, 50, 85])),
                    "MeteringMode": str(random.randint(1, 6)), "Flash": "16",
                    "WhiteBalance": "0", "ExposureProgram": str(random.randint(1, 4)),
                    "ColorSpace": "1", "Orientation": "1", "XResolution": "300",
                    "YResolution": "300", "ResolutionUnit": "2"}
        else:
            date = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
            meta = {"Make": random.choice(_MAKES),
                    "Model": random.choice(_MODELS), "DateTime": date}
        if self.use_software_tag:
            meta["Software"] = random.choice(_SOFTWARE)
        return meta

    def attack(self, inputs):
        tampered = self._pixel_attack(inputs["pixels"])
        metadata = self._forge_metadata()
        sig = quant_signature_from_tensor(tampered, quality=self.reencode_quality)
        compression = {"quant_signature": sig, "format": "JPEG",
                       "reencode_quality": self.reencode_quality}
        self.attack_count += 1
        return {"pixels": tampered, "metadata": metadata, "compression": compression}

    def damage_score(self, clean_conf, tampered_conf):
        shift = max(0.0, tampered_conf - clean_conf)
        return float(min(1.0, 0.5 * shift + 2.5 * self.epsilon))

    def update(self, verdict, flags_fired=None, damage=0.0):
        caught = self.perceive(verdict)
        flags  = flags_fired or []
        pixel_caught = "pixel" in flags
        meta_caught  = "metadata" in flags
        comp_caught  = "compression" in flags

        evasion = 0.0 if caught else 1.0
        reward  = self.weights["evasion"] * evasion + self.weights["damage"] * damage
        self.history.append({"epsilon": self.epsilon, "caught": caught,
                             "flags": flags, "reward": reward, "damage": damage,
                             "reencode_quality": self.reencode_quality})

        band = self.dyn["band"]
        if pixel_caught:
            self.epsilon = min(self.EPSILON_MAX, self.epsilon + self.dyn["up"])
        elif not caught:
            if self.epsilon < band:
                self.epsilon = min(band, self.epsilon + 0.004)
            else:
                self.epsilon = max(self.EPSILON_MIN, self.epsilon - self.dyn["down"])
        else:
            self.epsilon = max(self.EPSILON_MIN, self.epsilon - self.dyn["down"])
        self.epsilon = max(self.EPSILON_MIN, min(self.EPSILON_MAX, self.epsilon))

        if self.use_software_tag:
            if meta_caught:
                self.meta_caught_streak += 1
                if self.meta_caught_streak >= 2:
                    self.use_software_tag = False
                    self.meta_caught_streak = 0
            else:
                self.meta_caught_streak = 0
        else:
            if meta_caught and not self.use_realistic_exif:
                self.post_sw_meta_streak += 1
                if self.post_sw_meta_streak >= 2:
                    self.use_realistic_exif = True
                    self.post_sw_meta_streak = 0
            elif not meta_caught:
                self.post_sw_meta_streak = 0

        if comp_caught:
            self.reencode_quality = min(95, self.reencode_quality + 3)
        elif not caught:
            self.reencode_quality = max(70, self.reencode_quality - 1)

    def status(self):
        recent = self.history[-5:] if self.history else []
        cr = (sum(1 for h in recent if h["caught"]) / len(recent)
              if recent else None)
        return {"persona": self.persona, "epsilon": round(self.epsilon, 4),
                "attack_count": self.attack_count,
                "recent_catch_rate": round(cr, 2) if cr is not None else "n/a",
                "software_tag_active": self.use_software_tag,
                "realistic_exif_active": self.use_realistic_exif,
                "reencode_quality": self.reencode_quality}