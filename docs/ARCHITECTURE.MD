# CADS Architecture

## Overview

CADS is a co-evolutionary adversarial detection system. Two agents — Red (attacker) and
Blue (defender) — adapt to each other across N rounds. Neither agent resets between rounds.
Both carry persistent state that accumulates experience.

## Threat model

**Gray-box**: Red only observes Blue's final detection verdict (TAMPERED / CLEAN) and which
checks fired. Red never sees Blue's weights, gradients, or architecture. This matches the
realistic threat model for deployed agricultural AI systems where the attacker has query
access but not model access.

## Input channels

Every image is decomposed into three independent input channels at load time:

| Channel | What it is | How it is extracted |
|---|---|---|
| Pixels | RGB tensor (3, 128, 128) | PIL resize + numpy / 255 |
| EXIF metadata | Dict of tag name → value | PIL._getexif() |
| Compression | JPEG quality estimate | Raw DQT marker scan |

## Red agent (`src/agents/red.py`)

**Skill 1 — Adaptive epsilon**
- If caught: epsilon += 0.01 (push harder)
- If evaded: epsilon -= 0.005 (stay subtle)
- Bounds: [0.01, 0.20]

**Skill 2 — Indicator removal (MITRE ATT&CK)**
- Tracks consecutive rounds where metadata check fires
- After 2 consecutive meta-catches: use_software_tag = False permanently
- Models documented APT behavior of removing specific forensic indicators

**Pixel attack**: Canny edge detection → dilated branch mask → targeted noise
along root branch regions. Structural rather than textural — changes topology,
not just pixel values.

**Metadata forgery**: Real camera makes/models from plant phenotyping equipment
pools. DateTime always set to current date (subtle giveaway Blue could learn).

**CTI personas** (for RQ2):
- APT/stealth: evasion × 0.9 + damage × 0.1
- Cybercriminal: evasion × 0.5 + damage × 0.5
- Hacktivist: evasion × 0.1 + damage × 0.9

## Blue agent (`src/agents/blue.py`)

**Model**: Small Vision Transformer (ViT)
- 128×128 input → 256 patches of 8×8 pixels
- 64-dimensional embeddings, 4 attention heads, 2 encoder layers
- CLS token classification head → 2 logits (clean / tampered)
- Chosen over CNN because ViT attention compares every patch to every other patch
  simultaneously — important for root topology where the signal is in global structure,
  not local texture

**Skill 1 — Adaptive threshold**
- Monitors false positive and false negative rates over last 6 decisions
- FP rate > 0.50: raise threshold (being too aggressive)
- FN rate > 0.33: lower threshold (missing too many attacks)
- Otherwise: drift back toward 0.5 (mean reversion)
- Bounds: [0.30, 0.70]

**Skill 2 — ViT training loop**
- collect(): buffers each seen image with true label
- train_round(): 8 gradient steps per round on buffered pairs
- AdamW optimizer, lr=5e-4, weight_decay=1e-4
- Gradient clipping at max_norm=1.0

**Preprocessing — two passes**
- Pass 1 (baseline): per-channel normalization only
- Pass 2 (CLAHE): contrast enhancement on L channel (LAB space) then normalize
- Applied identically during training and inference

**Warmup**: 10 clean images, 15 gradient steps before round 1
- Prevents random weight initialization from causing false positive spike
- Alex's feedback: "give it a kick start"

## Evaluation

Four graphs produced per pass:
1. Arms race — Red epsilon vs Blue threshold over rounds
2. Catch rate — cumulative and per-round
3. Epsilon vs outcome — empirical evasion threshold (RQ1 failure boundary)
4. Rolling catch rate (5-round window) + detection check heatmap

## Known limitations

1. Two images per round — insufficient for robust ViT learning
2. No weight persistence between runs — Blue retrains from scratch every session
3. Compression attack is simulated — not actual JPEG re-encoding
4. Metadata check neutralized after round 2 by Red's Skill 2
5. No static baseline comparison for RQ1
6. No generalization test (RQ3) — FGSM/C&W/Boundary Attack evaluation pending
7. Single seed results — no statistical significance testing