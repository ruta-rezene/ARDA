# ARDA — Adversarial Resilience for Data Authenticity

ARDA is a multi-agent adversarial framework for studying and detecting image manipulation in scientific data pipelines. It is designed specifically for plant phenotyping systems, where corrupted image inputs can silently propagate through AI models and corrupt downstream genomic research outcomes.

The system simulates an adaptive attacker and an adaptive defender co-evolving across iterative rounds on real plant root architecture images. Three Cyber Threat Intelligence (CTI) adversary motivation profiles are operationalized as formal reward functions to study whether adversary motivation produces measurably distinct evasion-damage tradeoffs — a foundational claim in threat intelligence practice that has not previously been tested in a controlled experimental setting.

---

## Research Context

Automated plant phenotyping pipelines extract biological trait measurements from root images. These measurements feed into genome-wide association studies (GWAS) that identify gene-trait links used to guide crop breeding programs. A silent adversarial attack on the image data entering these pipelines can corrupt trait measurements, distort GWAS associations, and misdirect breeding decisions — with consequences that may not surface until failed field trials years later.

This problem is especially acute in resource-constrained research environments in the Global South, where a single pipeline may serve as the sole empirical basis for food security decisions with no independent verification capacity.

ARDA addresses this gap by building a forensic detection system and using it as an experimental instrument to characterize how different threat actor types behave under adaptive detection pressure.

---

## System Architecture

```
ARDA/
├── main.py                        ← entry point
├── requirements.txt
├── src/
│   ├── agents/
│   │   ├── red.py                 ← attacker agent (4 adaptive skills, 3 CTI personas)
│   │   └── blue.py                ← defender agent (ViT + 3 forensic channels)
│   ├── core/
│   │   ├── cads.py                ← image loading + quantization signatures
│   │   ├── engine.py              ← co-evolution loop
│   │   └── pretrain.py            ← offline dataset generation and training
│   └── evaluation/
│       └── stats.py               ← all research graphs
├── data/testImages/               ← place your root images here
├── models/                        ← trained weights saved here
├── dataset/                       ← generated training pairs saved here
└── results/                       ← all graphs and JSON summaries saved here
```

---

## Installation

```bash
git clone https://github.com/rutarezene/arda
cd arda
python3 -m venv arda_env
source arda_env/bin/activate
pip install -r requirements.txt
```

Place your root images in `data/testImages/`. Any subfolder structure is supported.

---

## Usage

```bash
python3 main.py
```

Four modes are available.

**Mode 1 — Simulation**
Runs one CTI persona for a chosen number of rounds. Red attacks and Blue defends across iterative rounds, both adapting based on outcomes. Produces six research-grade figures and a JSON summary. This is where the primary experimental findings come from.

**Mode 2 — Detection**
Loads trained Blue weights and runs forensic inspection on every image in a folder. Returns CLEAN or TAMPERED per image with the contributing detection channel. Generates a provenance ledger with SHA-256 cryptographic hashes for chain-of-custody verification. Intended for use before images enter a phenotyping pipeline.

**Mode 3 — Pretrain**
Generates a labeled dataset of clean and tampered image pairs across all three personas at randomized perturbation levels. Trains Blue's Vision Transformer on that dataset with a proper train/validation split. Saves pretrained weights to `models/blue_pretrained.pt`. Run this first on a GPU before any simulation to ensure Blue enters the arms race already trained on hundreds of images.

**Mode 4 — Compare**
Runs all three CTI personas under identical conditions and produces the persona comparison figures, including the evasion-damage Pareto tradeoff plot that is the primary research finding.

**Recommended order:**
```
Mode 3 → Mode 4 → Mode 1 → Mode 2
```

---

## The Attacker (Red Agent)

Red applies three simultaneous independent manipulation channels each round:

- **Pixel channel** — Canny edge-guided structured noise targeting root branch regions at magnitude epsilon
- **Metadata channel** — forged EXIF fields including camera make, model, and timestamp
- **Compression channel** — JPEG re-encoding at a learned quality level, producing a detectable quantization signature shift

Red has four adaptive skills:

| Skill | Behavior | MITRE ATT&CK Mapping |
|---|---|---|
| 1 | Epsilon control, persona-specific band | — |
| 2 | Software tag removal after repeated metadata catches | T1070 Indicator Removal |
| 3 | Full realistic EXIF forgery after continued metadata catches | T1070 |
| 4 | Compression quality lift when compression channel catches Red | — |

Three CTI personas operationalize real adversary motivation profiles:

| Persona | Evasion Weight | Damage Weight | Behavior |
|---|---|---|---|
| APT | 0.9 | 0.1 | Stealth-first, low epsilon, hard to detect |
| Cybercriminal | 0.5 | 0.5 | Balanced evasion and damage |
| Hacktivist | 0.1 | 0.9 | Damage-first, high epsilon, detectable |

---

## The Defender (Blue Agent)

Blue runs three independent forensic detection channels:

- **Pixel channel** — Vision Transformer operating on the high-pass residual of each image. The residual (image minus local 3×3 blur, amplified ×8) concentrates adversarial perturbation noise into a separable feature space following the Spatial Rich Model principle from image steganalysis
- **Metadata channel** — three sub-checks: editing software tag presence, DateTime freshness under 24 hours, and EXIF field sparsity inconsistent with genuine camera capture
- **Compression channel** — compares the JPEG luminance quantization table mean against a clean reference derived from the image pool, flagging re-encoding artifacts

Blue has two adaptive skills:

- **Skill 1** — adaptive detection threshold based on recent false positive and false negative rates
- **Skill 2** — online ViT retraining on a 200-image replay buffer each round

---

## Output Figures

Mode 1 produces:
- `01_attacker_detector_trajectory.png` — epsilon and threshold over rounds
- `02_epsilon_vs_outcome.png` — perturbation magnitude with caught/evaded regions
- `03_detection_performance.png` — precision, recall, F1, evasion rate, detector/attacker lead shading
- `04_pr_roc.png` — precision-recall and ROC curves
- `05_channel_activity.png` — which forensic channel fired each round
- `06_catch_rate.png` — cumulative and rolling catch rate

Mode 4 additionally produces:
- `persona_01_epsilon_trajectories.png` — epsilon by threat profile
- `persona_02_evasion_rate.png` — rolling evasion rate by threat profile
- `persona_03_evasion_damage_pareto.png` — evasion-damage tradeoff across personas

---

## Citation

If you use ARDA in your research please cite:

```
Rezene, R. (2026). ARDA: Adversarial Resilience for Data Authenticity.
University of Arizona UROC Research Program.
Supervisor: Dr. Alexander Bucksch, Computational Plant Sciences Laboratory.
```

---

## Acknowledgments

This research was conducted through the University of Arizona UROC (Undergraduate Research Opportunities Consortium) program under the supervision of Dr. Alexander Bucksch, Computational Plant Sciences Laboratory. Plant root image data was provided by the Bucksch Lab.
