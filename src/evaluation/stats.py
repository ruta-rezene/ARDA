"""
src/evaluation/stats.py
Publication-quality figures for all modes.
Clean line/area style only -- no scatter markers, no per-point symbols.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, auc,
                             average_precision_score)

OUT = "results"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "legend.fontsize": 10, "legend.framealpha": 0.9,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

C = {"blue":   "#2563EB", "red":    "#DC2626", "green":  "#16A34A",
     "amber":  "#D97706", "purple": "#7C3AED", "gray":   "#6B7280",
     "teal":   "#0D9488", "slate":  "#334155"}


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), facecolor="white")
    plt.close(fig)
    print(f"  saved results/{name}")


def _roll(hist, w=5):
    """Rolling window precision / recall / F1 / evasion rate."""
    r, prec, rec, f1, ev = [], [], [], [], []
    for i, h in enumerate(hist):
        c = hist[max(0, i - w + 1): i + 1]
        tp = sum(1 for x in c if x["caught"])
        fn = len(c) - tp
        fp = sum(1 for x in c if not x["caught"] and x["pixel_flag"])
        p  = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        r.append(h["round"]); prec.append(p); rec.append(rc)
        f1.append(2 * p * rc / (p + rc) if p + rc else 0)
        ev.append(fn / len(c) if c else 0)
    return r, prec, rec, f1, ev


# =====================================================================
# MODE 1  (single persona, single run)
# =====================================================================

def simulation_stats(sim: dict, persona: str = "cybercriminal"):
    """
    sim: dict returned by run_pass()
    """
    print("\n  generating figures...")
    h = sim["history"]
    n = sim["n_rounds"]

    # ── Fig 1: attacker epsilon and detector threshold over rounds ────────
    fig, ax = plt.subplots(figsize=(10, 5))
    rds = [x["round"] for x in h]
    ax.plot(rds, [x["epsilon"]   for x in h], color=C["red"],  lw=2,
            label="Attacker perturbation magnitude (epsilon)")
    ax.plot(rds, [x["threshold"] for x in h], color=C["blue"], lw=2,
            label="Detector threshold (tau)")
    ax.set_xlabel("Round"); ax.set_ylabel("Value")
    ax.set_title(f"Attacker perturbation magnitude and detector threshold "
                 f"over rounds  |  {persona}")
    ax.legend()
    _save(fig, "01_attacker_detector_trajectory.png")

    # ── Fig 2: epsilon vs detection outcome (shaded regions) ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = [x["epsilon"] for x in h]
    ax.plot(rds, eps, color=C["slate"], lw=2,
            label="Epsilon (perturbation magnitude)")
    ax.fill_between(rds, 0, eps, where=[not x["caught"] for x in h],
                    color=C["green"], alpha=0.15, label="Evaded region")
    ax.fill_between(rds, 0, eps, where=[x["caught"] for x in h],
                    color=C["red"],   alpha=0.10, label="Caught region")
    ax.set_xlabel("Round"); ax.set_ylabel("Epsilon (perturbation magnitude)")
    ax.set_title(f"Perturbation magnitude vs detection outcome  |  {persona}")
    ax.legend()
    _save(fig, "02_epsilon_vs_outcome.png")

    # ── Fig 3: detection performance (priority graph) ─────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    rds2, p, rc, f1, ev = _roll(h)
    ax.fill_between(rds2, f1, ev,
                    where=[a_ >= b_ for a_, b_ in zip(f1, ev)],
                    color=C["blue"], alpha=0.10, interpolate=True,
                    label="Detector ahead")
    ax.fill_between(rds2, f1, ev,
                    where=[a_ < b_ for a_, b_ in zip(f1, ev)],
                    color=C["red"],  alpha=0.10, interpolate=True,
                    label="Attacker ahead")
    ax.plot(rds2, p,  color=C["blue"],   lw=2,   label="Precision")
    ax.plot(rds2, rc, color=C["teal"],   lw=2,   label="Recall")
    ax.plot(rds2, f1, color=C["purple"], lw=2.4, label="F1 (detector)")
    ax.plot(rds2, ev, color=C["red"],    lw=2, ls="--",
            label="Evasion rate (attacker)")
    ax.set_ylim(0, 1.05); ax.set_xlabel("Round"); ax.set_ylabel("Score")
    ax.set_title(f"Detection performance over rounds  |  {persona}")
    ax.legend(ncol=3, fontsize=9, loc="lower center")
    _save(fig, "03_detection_performance.png")

    # ── Fig 4: PR and ROC curves ──────────────────────────────────────────
    yt, ys = sim["y_true"], sim["y_score"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Detector discrimination  |  {persona}")
    if len(set(yt)) > 1:
        p_c, r_c, _ = precision_recall_curve(yt, ys)
        ap = average_precision_score(yt, ys)
        axes[0].plot(r_c, p_c, color=C["blue"], lw=2,
                     label=f"AP = {ap:.3f}")
        axes[0].axhline(0.5, color=C["gray"], lw=0.8, ls=":",
                        label="No-skill baseline")
        fpr, tpr, _ = roc_curve(yt, ys)
        auc_val = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color=C["blue"], lw=2,
                     label=f"AUC = {auc_val:.3f}")
        axes[1].plot([0, 1], [0, 1], color=C["gray"], lw=0.8, ls=":")
    axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall"); axes[0].legend(loc="lower left")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("ROC"); axes[1].legend(loc="lower right")
    _save(fig, "04_pr_roc.png")

    # ── Fig 5: detection channel activity heatmap ─────────────────────────
    fig, ax = plt.subplots(figsize=(13, 3.5))
    data = np.array([[int(x["comp_flag"])  for x in h],
                     [int(x["meta_flag"])  for x in h],
                     [int(x["pixel_flag"]) for x in h]])
    ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1,
              extent=[0.5, n + 0.5, -0.5, 2.5])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Compression", "Metadata", "Pixel"])
    ax.set_xlabel("Round")
    ax.set_title(f"Detection channel activity  |  {persona}"
                 "\n(dark = channel fired that round)")
    _save(fig, "05_channel_activity.png")

    # ── Fig 6: rolling catch rate ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    cum  = np.cumsum([1 if x["caught"] else 0 for x in h])
    crate = [cum[i] / (i + 1) for i in range(len(h))]
    ax.plot(rds, crate, color=C["blue"], lw=2, label="Cumulative catch rate")
    _, _, _, _, ev2 = _roll(h)
    rds3, _, _, _, ev3 = _roll(h, w=10)
    ax.plot(rds3, [1 - e for e in ev3], color=C["teal"], lw=1.6, ls="--",
            label="Rolling catch rate (window=10)")
    ax.axhline(0.5, color=C["gray"], lw=0.8, ls=":", alpha=0.6)
    ax.set_ylim(0, 1.05); ax.set_xlabel("Round"); ax.set_ylabel("Catch rate")
    ax.set_title(f"Catch rate over rounds  |  {persona}")
    ax.legend()
    _save(fig, "06_catch_rate.png")

    # ── Summary JSON ──────────────────────────────────────────────────────
    tp = sum(1 for x in h if x["caught"])
    fn = len(h) - tp
    fp = sum(1 for x in h if not x["caught"] and x["pixel_flag"])
    prec = round(tp / (tp + fp), 3) if tp + fp else 0
    rec  = round(tp / (tp + fn), 3) if tp + fn else 0
    f1v  = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0
    _write_summary({"persona": persona, "caught_pct": sim["pct"],
                    "evasion_rate": round(sim["evasion_rate"], 3),
                    "mean_damage":  round(sim["mean_damage"],  3),
                    "precision": prec, "recall": rec, "f1": f1v,
                    "pixel_fires":    sum(1 for x in h if x["pixel_flag"]),
                    "meta_fires":     sum(1 for x in h if x["meta_flag"]),
                    "comp_fires":     sum(1 for x in h if x["comp_flag"])},
                   "summary.json")
    print(f"\n  caught {sim['pct']}%   precision {prec}   recall {rec}   F1 {f1v}")


# =====================================================================
# MODE 4  (all three personas)
# =====================================================================

def persona_comparison_stats(results: dict):
    print("\n  generating persona comparison figures...")
    order  = ["apt", "cybercriminal", "hacktivist"]
    colmap = {"apt": C["blue"], "cybercriminal": C["amber"], "hacktivist": C["red"]}
    dashes = {"apt": (0, ()), "cybercriminal": (0, (5, 2)), "hacktivist": (0, (1, 1))}

    # ── Fig A: epsilon trajectories ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in order:
        h = results[p]["history"]
        ax.plot([x["round"] for x in h], [x["epsilon"] for x in h],
                color=colmap[p], lw=2, linestyle=dashes[p], label=p)
    ax.set_xlabel("Round"); ax.set_ylabel("Epsilon (perturbation magnitude)")
    ax.set_title("Attacker perturbation magnitude by threat profile")
    ax.legend()
    _save(fig, "persona_01_epsilon_trajectories.png")

    # ── Fig B: rolling evasion rate ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in order:
        rds, _, _, _, ev = _roll(results[p]["history"])
        ax.plot(rds, ev, color=colmap[p], lw=2.2, alpha=0.85,
                linestyle=dashes[p], label=p)
    ax.set_ylim(-0.03, 1.05); ax.set_xlabel("Round")
    ax.set_ylabel("Evasion rate")
    ax.set_title("Attacker evasion rate by threat profile")
    ax.legend()
    _save(fig, "persona_02_evasion_rate.png")

    # ── Fig C: evasion-damage Pareto ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6.5))
    xs = [results[p]["evasion_rate"] for p in order]
    ys = [results[p]["mean_damage"]  for p in order]
    ax.plot(xs, ys, color=C["gray"], lw=1.2, ls="--", zorder=1)
    for p, x, y in zip(order, xs, ys):
        ax.scatter([x], [y], s=260, color=colmap[p], zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.annotate(p, (x, y), textcoords="offset points", xytext=(12, 6),
                    fontsize=11, fontweight="bold", color=colmap[p])
    ax.set_xlabel("Evasion rate  (fraction of attacks that went undetected)")
    ax.set_ylabel("Mean downstream damage")
    ax.set_title("Evasion-damage tradeoff across threat profiles")
    pad_x = max(0.02, (max(xs) - min(xs)) * 0.25) if len(set(xs)) > 1 else 0.05
    pad_y = max(0.02, (max(ys) - min(ys)) * 0.25) if len(set(ys)) > 1 else 0.05
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    _save(fig, "persona_03_evasion_damage_pareto.png")

    _write_summary(
        {p: {"evasion_rate": round(results[p]["evasion_rate"], 3),
             "mean_damage":  round(results[p]["mean_damage"],  3),
             "caught_pct":   results[p]["pct"]} for p in order},
        "persona_summary.json")


# =====================================================================
# MODE 2  (detection on a real image folder)
# =====================================================================

def detection_stats(results: list):
    print("\n  generating detection figures...")
    n     = len(results)
    n_tam = sum(1 for r in results if r["verdict"] == "TAMPERED")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Detection results")
    ax[0].bar(["Clean", "Tampered"], [n - n_tam, n_tam],
              color=[C["green"], C["red"]], width=0.5)
    ax[0].set_ylabel("Images"); ax[0].set_title(f"Verdicts  (n = {n})")
    conf = [r["pixel_confidence"] for r in results]
    ax[1].plot(range(n), sorted(conf), color=C["blue"], lw=2)
    ax[1].axhline(0.5, color=C["gray"], ls="--", lw=1, label="Decision boundary")
    ax[1].set_xlabel("Image (sorted by confidence)")
    ax[1].set_ylabel("Pixel confidence (ViT)")
    ax[1].set_title("Confidence distribution"); ax[1].legend()
    _save(fig, "detection_summary.png")

    _write_summary({"total": n, "tampered": n_tam, "clean": n - n_tam,
                    "tampered_pct": round(100 * n_tam / n, 1) if n else 0},
                   "detection_summary.json")


# =====================================================================
# MODE 3  (pretrained evaluation)
# =====================================================================

def pretrained_stats(ev: dict):
    print("\n  generating evaluation figures...")
    cm = np.array([[ev["tn"], ev["fp"]], [ev["fn"], ev["tp"]]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pretrained detector -- held-out evaluation")

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    axes[0].imshow(norm, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Predicted clean", "Predicted tampered"])
    axes[0].set_yticklabels(["Actual clean",    "Actual tampered"])
    axes[0].set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{norm[i,j]:.2f}\n({cm[i,j]})",
                         ha="center", va="center", fontweight="bold",
                         color="white" if norm[i, j] > 0.6 else "black")

    mets = ["Accuracy", "Precision", "Recall", "F1"]
    vals = [ev["accuracy"], ev["precision"], ev["recall"], ev["f1"]]
    axes[1].bar(mets, vals,
                color=[C["blue"], C["teal"], C["green"], C["purple"]], width=0.55)
    axes[1].set_ylim(0, 1.15); axes[1].set_title("Held-out metrics")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.03, f"{v:.3f}", ha="center", fontweight="bold")
    _save(fig, "pretrained_evaluation.png")

    _write_summary({k: round(ev[k], 3) for k in
                    ["accuracy", "precision", "recall", "f1"]},
                   "pretrained_summary.json")


# ---- helpers ----------------------------------------------------------------

def _write_summary(d, name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(d, f, indent=2)
    print(f"  saved results/{name}")