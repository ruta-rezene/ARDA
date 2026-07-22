"""
main.py -- CADS entry point. The only file you run.

    python3 main.py

Modes:
  [1] SIMULATION   one CTI persona, single pass, full statistics
  [2] DETECTION    load trained Blue, check a folder of images
  [3] PRETRAIN     generate dataset -> pretrain Blue -> held-out eval
  [4] COMPARE      run all three CTI personas, plot evasion-damage tradeoff
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.cads    import load_image_with_all_inputs, find_all_images
from src.core.engine  import run_pass
from src.agents.blue  import BlueAgent
from src.evaluation.stats import (simulation_stats, detection_stats,
                                  pretrained_stats, persona_comparison_stats)

TRAINED = "models/blue_trained.pt"


# ---- shared prompts ---------------------------------------------------------

def pick_data(prompt="Root data folder (default: data): "):
    folder = input(f"\n  {prompt}").strip() or "data"
    if not os.path.isdir(folder):
        print(f"  error: '{folder}' not found"); sys.exit(1)
    paths = find_all_images(folder)
    if not paths:
        print(f"  no images under '{folder}'"); sys.exit(1)
    print(f"  found {len(paths)} images")
    return paths


def ask_rounds(default=60):
    raw = input(f"  how many rounds? (default {default}): ").strip()
    try:    return max(10, min(500, int(raw))) if raw else default
    except: return default


def ask_meta():
    a = input("  do your images have real camera EXIF? (yes/no/idk): ").strip().lower()
    if a in ("no", "n"):
        print("  -> stripped EXIF mode"); return True
    print("  -> real EXIF mode"); return False


# ---- MODE 1 -----------------------------------------------------------------

def mode_simulation():
    print("\n" + "=" * 62 + "\n  MODE 1 -- SIMULATION\n" + "=" * 62)
    paths = pick_data()
    n = ask_rounds()
    print("\n  persona:  [1] apt   [2] cybercriminal   [3] hacktivist")
    pc = input("  choose 1/2/3 (default 2): ").strip()
    persona = {"1": "apt", "2": "cybercriminal", "3": "hacktivist"}.get(pc, "cybercriminal")
    print(f"  -> {persona}")
    stripped = ask_meta()

    result = run_pass(paths, n, persona, metadata_stripped=stripped)
    result["blue"].save_weights(TRAINED)

    print("\n" + "=" * 62)
    print(f"  caught {result['pct']}%   evasion {result['evasion_rate']:.2f}"
          f"   damage {result['mean_damage']:.3f}")
    print("=" * 62)
    simulation_stats(result, persona=persona)
    print(f"\n  figures -> results/    weights -> {TRAINED}\n")


# ---- MODE 2 -----------------------------------------------------------------

def mode_detection():
    print("\n" + "=" * 62 + "\n  MODE 2 -- DETECTION\n" + "=" * 62)
    paths = pick_data("folder of images to check (default: data): ")

    sigs = []
    for p in paths[:12]:
        from src.core.cads import quant_signature_from_tensor
        px = load_image_with_all_inputs(p)["pixels"]
        s = quant_signature_from_tensor(px, 92)
        if s > 0: sigs.append(s)
    ref = float(sum(sigs) / len(sigs)) if sigs else -1.0

    blue = BlueAgent(clean_quant_ref=ref)
    loaded = blue.load_weights()
    print(f"  weights: {loaded or 'NONE (untrained -- run mode 1 or 3 first)'}\n")

    print(f"  {'image':<44}{'verdict':<11}{'conf':<7}flags")
    print("  " + "-" * 74)
    results = []
    for p in sorted(paths):
        inp = load_image_with_all_inputs(p)
        r = blue.detect(inp)
        flags = [k for k, v in (("pixel", r["pixel_flag"]),
                                ("meta",  r["metadata_flag"]),
                                ("comp",  r["compression_flag"])) if v]
        print(f"  {os.path.basename(p)[:42]:<44}{r['verdict']:<11}"
              f"{r['pixel_confidence']:<7.3f}[{'+'.join(flags) or 'none'}]")
        results.append({**r, "filename": p})
    n_t = sum(1 for r in results if r["verdict"] == "TAMPERED")
    print(f"\n  {n_t}/{len(results)} flagged tampered")
    detection_stats(results)
    print("\n  figures -> results/\n")


# ---- MODE 3 -----------------------------------------------------------------

def mode_pretrain():
    print("\n" + "=" * 62 + "\n  MODE 3 -- PRETRAIN\n" + "=" * 62)
    from src.core.pretrain import generate_dataset, pretrain_blue, evaluate_pretrained
    paths = pick_data("root image folder for dataset (default: data): ")
    folder = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])

    npairs = input("  how many clean/tampered pairs? (default 200): ").strip()
    npairs = int(npairs) if npairs.isdigit() else 200
    epochs = input("  training epochs? (default 15): ").strip()
    epochs = int(epochs) if epochs.isdigit() else 15

    print("\n  -- step 1: dataset --")
    generate_dataset(folder, n_pairs=npairs)
    print("\n  -- step 2: pretrain --")
    pretrain_blue(epochs=epochs)
    print("\n  -- step 3: held-out eval --")
    ev = evaluate_pretrained()
    pretrained_stats(ev)
    print("\n  figures -> results/   weights -> models/blue_pretrained.pt\n")


# ---- MODE 4 -----------------------------------------------------------------

def mode_compare():
    print("\n" + "=" * 62 + "\n  MODE 4 -- COMPARE ALL THREE PERSONAS\n" + "=" * 62)
    paths = pick_data()
    n = ask_rounds(default=80)
    stripped = ask_meta()

    results = {}
    for persona in ["apt", "cybercriminal", "hacktivist"]:
        print("\n" + "=" * 62 + f"\n  PERSONA: {persona}\n" + "=" * 62)
        results[persona] = run_pass(paths, n, persona, metadata_stripped=stripped)

    print("\n" + "=" * 62 + "\n  PERSONA SUMMARY\n" + "=" * 62)
    for p in ["apt", "cybercriminal", "hacktivist"]:
        r = results[p]
        print(f"  {p:<14} caught {r['pct']:>3}%   "
              f"evasion {r['evasion_rate']:.2f}   damage {r['mean_damage']:.3f}")
    persona_comparison_stats(results)
    print("\n  figures -> results/\n")


# ---- entry ------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("  CADS -- Co-Evolutionary Adversarial Detection System")
    print("=" * 62)
    print("""
  [1] SIMULATION   one persona, single run, full statistics
  [2] DETECTION    check a folder of images with trained Blue
  [3] PRETRAIN     build dataset, pretrain Blue, evaluate held-out
  [4] COMPARE      all three CTI personas + evasion-damage tradeoff
""")
    c = input("  enter 1/2/3/4: ").strip()
    {"1": mode_simulation, "2": mode_detection,
     "3": mode_pretrain,   "4": mode_compare}.get(c, lambda: print("  invalid"))()


if __name__ == "__main__":
    main()