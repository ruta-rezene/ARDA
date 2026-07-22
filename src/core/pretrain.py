"""
src/core/pretrain.py
Offline dataset generation, Blue pretraining, and held-out evaluation.
Used by main.py Mode 3.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from src.core.cads   import load_image_with_all_inputs, find_all_images
from src.agents.red  import RedAgent, PERSONAS
from src.agents.blue import BlueAgent, preprocess, PRETRAINED_PATH

DATASET_DIR = "dataset"


def generate_dataset(image_folder, n_pairs=200):
    os.makedirs(DATASET_DIR, exist_ok=True)
    images = find_all_images(image_folder)
    if not images:
        raise ValueError(f"No images under {image_folder}")
    personas = list(PERSONAS.keys())
    clean, tampered = [], []
    print(f"\n  generating {n_pairs} clean/tampered pairs (all personas)...")
    for i in range(n_pairs):
        inp = load_image_with_all_inputs(random.choice(images))
        red = RedAgent(persona=personas[i % len(personas)],
                       epsilon_start=random.uniform(0.04, 0.16))
        t = red.attack(inp)
        clean.append(inp["pixels"]); tampered.append(t["pixels"])
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{n_pairs}")
    torch.save(torch.stack(clean),    os.path.join(DATASET_DIR, "clean.pt"))
    torch.save(torch.stack(tampered), os.path.join(DATASET_DIR, "tampered.pt"))
    print(f"  saved dataset/clean.pt and dataset/tampered.pt")


def pretrain_blue(epochs=15, batch_size=16, val_split=0.2):
    clean = torch.load(os.path.join(DATASET_DIR, "clean.pt"))
    tamp  = torch.load(os.path.join(DATASET_DIR, "tampered.pt"))
    print(f"\n  loaded {len(clean)} clean + {len(tamp)} tampered")
    cx = torch.stack([preprocess(i) for i in clean])
    tx = torch.stack([preprocess(i) for i in tamp])
    X = torch.cat([cx, tx])
    y = torch.cat([torch.zeros(len(cx), dtype=torch.long),
                   torch.ones(len(tx), dtype=torch.long)])
    ds = TensorDataset(X, y)
    nv = max(1, int(len(ds) * val_split))
    tr, va = random_split(ds, [len(ds) - nv, nv],
                          generator=torch.Generator().manual_seed(42))
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    val = DataLoader(va, batch_size=batch_size)

    blue = BlueAgent()
    best = 0.0
    print(f"  train {len(tr)}  val {len(va)}  epochs {epochs}\n")
    for ep in range(1, epochs + 1):
        blue.vit.train()
        tot = 0.0
        for xb, yb in trl:
            blue.optimizer.zero_grad()
            loss = blue.loss_fn(blue.vit(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(blue.vit.parameters(), 1.0)
            blue.optimizer.step()
            tot += loss.item()
        blue.vit.eval()
        cor = tt = 0
        with torch.no_grad():
            for xb, yb in val:
                cor += (blue.vit(xb).argmax(1) == yb).sum().item()
                tt += len(yb)
        acc = cor / tt if tt else 0
        print(f"  epoch {ep:2d}/{epochs}  loss={tot/len(trl):.4f}  val_acc={acc:.3f}")
        if acc > best:
            best = acc
            blue.save_weights(PRETRAINED_PATH)
    print(f"\n  best val acc {best:.3f}  ->  {PRETRAINED_PATH}")
    return best


def evaluate_pretrained(val_split=0.2):
    clean = torch.load(os.path.join(DATASET_DIR, "clean.pt"))
    tamp  = torch.load(os.path.join(DATASET_DIR, "tampered.pt"))
    n = min(len(clean), len(tamp))
    nv = max(1, int(n * val_split))
    rng = random.Random(42)
    idx = list(range(n)); rng.shuffle(idx)
    held = idx[:nv]

    blue = BlueAgent()
    if not blue.load_weights():
        print("  WARNING no weights found")

    yt, yp = [], []
    blue.vit.eval()
    with torch.no_grad():
        for i in held:
            for pixels, label in [(clean[i], 0), (tamp[i], 1)]:
                x = preprocess(pixels).unsqueeze(0)
                pred = int(torch.softmax(blue.vit(x), 1)[0, 1] > 0.5)
                yt.append(label); yp.append(pred)

    tp = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(yt, yp) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(yt, yp) if t == 1 and p == 0)
    acc = (tp + tn) / len(yt) if yt else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"  accuracy={acc:.3f} precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}