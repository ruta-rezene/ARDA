"""
src/core/engine.py
Core co-evolution loop, shared by all simulation modes.

run_pass() executes one arms-race pass (N rounds) for a given persona and
preprocessing setting, and returns a per-round history plus summary stats.
It builds a clean reference compression signature from real images so
Blue's compression check has a genuine baseline to compare against.
"""

import random
import numpy as np
import torch

from src.core.cads   import (load_image_with_all_inputs,
                             quant_signature_from_tensor)
from src.agents.red  import RedAgent
from src.agents.blue import BlueAgent


def _clean_quant_reference(image_paths, n=8):
    """
    Reference signature computed through the SAME tensor re-encode path Red
    uses, at the images' native quality (92). This makes Blue's comparison
    fair: a stealthy re-encode near q=92 lands close to this reference and
    evades, while a heavy re-encode lands far above it and is caught.
    """
    sigs = []
    for p in random.sample(image_paths, min(n, len(image_paths))):
        px = load_image_with_all_inputs(p)["pixels"]
        s = quant_signature_from_tensor(px, quality=92)
        if s > 0:
            sigs.append(s)
    return float(np.median(sigs)) if sigs else -1.0


def _warmup(blue, image_paths, n=50):
    """
    Pre-train Blue on clean images before round 1.
    50 images, 30 steps -- gives the ViT a meaningful head start
    before the arms race begins (mentor advice: train by hand first).
    """
    sample = random.sample(image_paths, min(n, len(image_paths)))
    for p in sample:
        blue.collect(load_image_with_all_inputs(p)["pixels"], "CLEAN")
    blue.train_round(n_steps=30)


def run_pass(image_paths, n_rounds, persona,
             metadata_stripped=False, pool_size=500, verbose=True):
    """
    One arms-race pass. Uses a large image pool (mentor advice: many images,
    not two) -- each round samples fresh clean+tampered images from the pool.
    """
    pool = random.sample(image_paths, min(pool_size, len(image_paths)))
    clean_ref = _clean_quant_reference(pool)

    red  = RedAgent(epsilon_start=0.05, persona=persona)
    blue = BlueAgent(metadata_stripped=metadata_stripped,
                     clean_quant_ref=clean_ref)

    _warmup(blue, pool)

    history, caught_total = [], 0
    y_true, y_score = [], []

    if verbose:
        print(f"  {'Rnd':<5}{'eps':<9}{'thr':<8}{'SWtag':<7}{'REXIF':<7}"
              f"{'result':<10}{'dmg':<7}flags")
        print("  " + "-" * 66)

    for rnd in range(1, n_rounds + 1):
        # sample a clean image, produce its tampered counterpart
        clean_inputs = load_image_with_all_inputs(random.choice(pool))
        # clean image's signature on the same tensor basis as the reference
        clean_inputs["compression"]["quant_signature"] = \
            quant_signature_from_tensor(clean_inputs["pixels"], quality=92)
        tampered = red.attack(clean_inputs)

        # Blue sees clean then tampered; trains on both
        clean_res = blue.detect(clean_inputs)
        blue.collect(clean_inputs["pixels"], "CLEAN")
        blue.update("CLEAN", clean_res["verdict"])

        tamp_res = blue.detect(tampered)
        blue.collect(tampered["pixels"], "TAMPERED")
        blue.update("TAMPERED", tamp_res["verdict"])
        blue.train_round(n_steps=10)

        flags = [k for k, v in (("pixel", tamp_res["pixel_flag"]),
                                ("metadata", tamp_res["metadata_flag"]),
                                ("compression", tamp_res["compression_flag"])) if v]
        damage = red.damage_score(clean_res["pixel_confidence"],
                                  tamp_res["pixel_confidence"])
        red.update(tamp_res["verdict"], flags_fired=flags, damage=damage)

        caught = tamp_res["verdict"] == "TAMPERED"
        caught_total += int(caught)
        # record both classes so PR/ROC curves are well-defined
        y_true.append(0); y_score.append(clean_res["pixel_confidence"])
        y_true.append(1); y_score.append(tamp_res["pixel_confidence"])

        rs, bs = red.status(), blue.status()
        history.append({
            "round": rnd, "epsilon": rs["epsilon"], "threshold": bs["threshold"],
            "caught": caught, "damage": damage,
            "pixel_flag": tamp_res["pixel_flag"],
            "meta_flag": tamp_res["metadata_flag"],
            "comp_flag": tamp_res["compression_flag"],
            "pixel_conf_tampered": tamp_res["pixel_confidence"],
            "pixel_conf_clean": clean_res["pixel_confidence"],
            "sw_tag": rs["software_tag_active"],
            "realistic_exif": rs["realistic_exif_active"],
        })

        if verbose:
            disp = "+".join({"pixel": "px", "metadata": "meta",
                             "compression": "comp"}[f] for f in flags) or "none"
            print(f"  {rnd:<5}{rs['epsilon']:<9.4f}{bs['threshold']:<8.3f}"
                  f"{'ON' if rs['software_tag_active'] else 'off':<7}"
                  f"{'ON' if rs['realistic_exif_active'] else 'off':<7}"
                  f"{'CAUGHT' if caught else 'evaded':<10}"
                  f"{damage:<7.3f}[{disp}]")

    pct = round(100 * caught_total / n_rounds)
    if verbose:
        print(f"\n  caught {caught_total}/{n_rounds} ({pct}%)")

    # final evasion / damage for the persona tradeoff plot (last third)
    tail = history[len(history) // 3:]
    evasion_rate = 1.0 - (sum(h["caught"] for h in tail) / max(1, len(tail)))
    mean_damage  = float(np.mean([h["damage"] for h in tail])) if tail else 0.0

    return {"history": history, "caught_total": caught_total,
            "n_rounds": n_rounds, "pct": pct, "persona": persona,
            "y_true": y_true, "y_score": y_score,
            "evasion_rate": evasion_rate, "mean_damage": mean_damage,
            "blue": blue}