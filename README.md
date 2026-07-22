# CADS — Co-Evolutionary Adversarial Detection System

**Co-Evolutionary Adversarial Detection Under Gray-Box Attack in Plant Phenotyping Pipelines**

University of Arizona UROC Research Program | Student: Ruta Rezene | PI: Alex

---

## What this is

CADS is a co-evolutionary adversarial machine learning system that simulates and studies
an arms race between an adaptive attacker (Red) and an adaptive defender (Blue), applied
to real plant root architecture images from a computational plant science lab.

Red manipulates images across three simultaneous channels — pixel structure, EXIF metadata,
and JPEG compression fingerprints. Blue detects manipulation using a Vision Transformer (ViT)
alongside metadata consistency checking and relative compression analysis. Both agents adapt
round-to-round through documented skills. Neither resets between rounds.

The system is grounded in Cyber Threat Intelligence (CTI), with three adversary motivation
profiles operationalized as distinct reward functions to empirically characterize the tradeoff
between evasion efficiency and downstream classification damage.

---

## Research questions

**RQ1:** Does continuous co-evolutionary training with an adaptive attacker agent develop
stronger detection of unseen plant image manipulation compared to a detection model trained
on a fixed, static dataset?

**RQ2:** Do different threat actor motivations (APT/stealth, cybercriminal/balanced,
hacktivist/damage) produce measurably different attack strategies, and is there a real
tradeoff between evasion and downstream damage?

---

## Project structure

```
CADS/
├── src/
│   ├── agents/
│   │   ├── red.py          # Agentic Red attacker — adaptive epsilon + indicator removal
│   │   └── blue.py         # Agentic Blue defender — ViT + adaptive threshold
│   ├── core/
│   │   └── cads.py         # Shared image loading utility (pixels, EXIF, compression)
│   ├── preprocessing/
│   │   └── transforms.py   # Normalization and CLAHE preprocessing passes
│   └── evaluation/
│       └── metrics.py      # Accuracy, precision, recall, F1, confusion matrix
├── scripts/
│   └── run.py              # Main co-evolution loop — entry point
├── tests/
│   └── test_pipeline.py    # Smoke tests for each component
├── examples/
│   └── single_image.py     # Run CADS on one image and print results
├── docs/
│   ├── ARCHITECTURE.md     # How the system works
│   └── RESULTS.md          # Current experimental results
├── results/                # Output graphs saved here (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/yourusername/CADS.git
cd CADS
python3.11 -m venv cads_env
source cads_env/bin/activate
pip install -r requirements.txt
python3 scripts/run.py testImages/L882015
```

---

## Requirements

- Python 3.11
- torch==2.2.2
- numpy<2
- pillow
- opencv-python-headless
- matplotlib

See `requirements.txt` for full list.

---

## How Red and Blue work

**Red** attacks each image across three channels simultaneously:
1. Structural pixel perturbation — Canny edge detection finds root branches, targeted noise applied along branch regions
2. EXIF metadata forgery — plausible fake camera Make/Model/DateTime. Removes Software tag if metadata check catches it twice (MITRE ATT&CK: Indicator Removal on Host)
3. Compression fingerprint shift — variable 3-10 point JPEG quality drop

Red adapts every round: increases perturbation strength if caught, decreases if evading.

**Blue** checks three independent signals:
1. Vision Transformer pixel confidence vs adaptive threshold
2. Software EXIF tag presence (suspicious in unedited lab images)
3. Relative JPEG quality drop from original fingerprint

Blue retrains its ViT each round on observed images and recalibrates its detection threshold based on recent false positive and false negative rates.

---

## CTI adversary personas (RQ2)

| Persona | Evasion weight | Damage weight | Real-world analogue |
|---|---|---|---|
| APT / stealth | 0.9 | 0.1 | State-aligned persistent threat |
| Cybercriminal | 0.5 | 0.5 | Financially motivated attacker |
| Hacktivist | 0.1 | 0.9 | Ideologically motivated disruptor |

---

## License

MIT

---

## Two modes

### Mode 1 — Simulation (research)
Runs the co-evolutionary training loop. Red and Blue adapt across N rounds.
Saves Blue's trained weights at the end.

```bash
python3 scripts/run.py testImages/L882015
```

### Mode 2 — Inference (real-world detection)
Loads trained Blue weights and checks whether uploaded images have been tampered with.
No Red, no training, no co-evolution — just detection.

```bash
python3 scripts/detect.py path/to/root_image.jpg
python3 scripts/detect.py path/to/folder/of/images/
```

Run Mode 1 first to train Blue. Mode 2 loads the weights Mode 1 saved.



-------------------------------------------------------------------------------
Stand alone research scripts for research purposes: 

scripts/generate_dataset.py
Run this first, before any pretraining. It takes your root images, attacks them with all three CTI personas at randomized epsilon values, and saves labeled clean/tampered tensor pairs to data/dataset/. You run this once when you have enough real lab images to build a meaningful training set. Run it again if you get more images or want a larger dataset. Everything that comes after depends on this output existing.
bashpython3 scripts/generate_dataset.py data/ --n 300

scripts/pretrain_blue.py
Run this second, after generate_dataset.py. It trains Blue's ViT on the saved dataset using a proper train/validation split, saves the best-performing weights to models/blue_pretrained.pt, and prints validation accuracy at the end of every epoch. You run this once to produce the pretrained weights you commit to GitHub. Run it again if you regenerated the dataset with more images or want to experiment with more epochs. The weights it produces are what make Mode 2 work immediately for anyone who clones your repo.
bashpython3 scripts/pretrain_blue.py --epochs 20

scripts/evaluate_pretrained.py
Run this third, after pretraining. It loads blue_pretrained.pt, reconstructs the held-out split that pretraining never touched, runs the full three-layer Blue detection on every held-out image, and reports accuracy, precision, recall, F1, confusion matrix, and which checks contributed to each catch. These are the formal metrics you report in your paper for the pretrained detector. Run it any time you want to re-evaluate after changing Blue's architecture, the metadata checks, or the preprocessing. Run it once before and once after adding the DateTime and sparsity metadata improvements to show the difference those changes made.
bashpython3 scripts/evaluate_pretrained.py

scripts/run.py
Called by main.py when the user chooses Mode 1. You don't run this directly under normal circumstances — python3 main.py handles it. The one time you'd run it directly is if you want to test the simulation with custom parameters without going through the interactive prompts, by importing and calling run_rounds() from another script.

scripts/detect.py
Called by main.py when the user chooses Mode 2. Same as run.py — you don't run this directly under normal circumstances. The exception is if you want to quickly check one image from the terminal during development without going through the full main.py menu.
bashpython3 scripts/detect.py data/testImages/L882015/some_image.jpg