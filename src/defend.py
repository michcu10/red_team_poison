"""
src/defend.py — Blue team orchestrator.

Loads each of the 5 trained models (Clean + Patch/Freq × 1%/3%) and runs three
defenses against them: STRIP, Spectral Signatures, and Fine-Pruning. Emits a
master JSON to results/defense_<ts>.json and a Markdown summary alongside it.

CLI mirrors src/evaluate.py — the same trigger parameters used during training
must be used here so that triggered test inputs match the trained-in pattern.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.train import get_resnet18_cifar
from src.evaluate import AttackTestDataset
from src.data_utils import PoisonedCIFAR10
from src.logging_utils import setup_run_logger
from src.defenses.strip import run_strip, build_overlay_pool
from src.defenses.spectral_signatures import run_spectral_signatures
from src.defenses.fine_pruning import run_fine_pruning


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


def _parse_ratios(s):
    raw = [float(r.strip()) for r in s.split(",")]
    out, seen = [], set()
    for r in raw:
        r = max(0.01, min(0.03, r))
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _load_model(path, device):
    model = get_resnet18_cifar().to(device)
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Run blue-team defenses against trained models.")
    p.add_argument("--patch-location", type=int, nargs=2, default=[0, 0], metavar=("Y", "X"))
    p.add_argument("--patch-size", type=int, default=12)
    p.add_argument("--freq-intensity", type=float, default=60.0)
    p.add_argument("--freq-band-start", type=int, default=22)
    p.add_argument("--freq-patch-size", type=int, default=8)
    p.add_argument("--poison-ratios", type=str, default="0.01,0.03")
    p.add_argument("--output-dir", type=str, default="results")

    # Defense knobs
    p.add_argument("--strip-overlays", type=int, default=100)
    p.add_argument("--strip-pool-size", type=int, default=200)
    p.add_argument("--strip-frr", type=float, default=0.05)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--clean-subset-size", type=int, default=2000,
                   help="Number of clean training samples used by Fine-Pruning.")
    p.add_argument("--prune-ratios", type=str, default="0.1,0.2,0.3")
    p.add_argument("--skip-strip", action="store_true")
    p.add_argument("--skip-spectral", action="store_true")
    p.add_argument("--skip-finepruning", action="store_true")
    args = p.parse_args()

    poison_ratios = _parse_ratios(args.poison_ratios)
    prune_ratios = tuple(float(x) for x in args.prune_ratios.split(","))
    trigger_kwargs = {
        "location": tuple(args.patch_location),
        "patch_size": args.patch_size,
        "intensity": args.freq_intensity,
        "band_start": args.freq_band_start,
        "freq_patch_size": args.freq_patch_size,
    }

    with setup_run_logger("defense", output_dir=args.output_dir) as log_file:
        print(f"Logging output to: {log_file}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Trigger kwargs: {trigger_kwargs}")

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        # Datasets
        testset_clean = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader_clean = DataLoader(testset_clean, batch_size=128, shuffle=False, num_workers=2)

        attack_patch = AttackTestDataset(testset_clean, trigger_type="patch", trigger_kwargs=trigger_kwargs)
        attack_freq = AttackTestDataset(testset_clean, trigger_type="frequency", trigger_kwargs=trigger_kwargs)
        attack_patch_loader = DataLoader(attack_patch, batch_size=128, shuffle=False, num_workers=2)
        attack_freq_loader = DataLoader(attack_freq, batch_size=128, shuffle=False, num_workers=2)

        # Clean subset for Fine-Pruning (held out from the train set; same transforms as test)
        trainset_clean_for_ft = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_test
        )
        rng = np.random.default_rng(123)
        ft_indices = rng.choice(len(trainset_clean_for_ft), size=args.clean_subset_size, replace=False)
        clean_subset = Subset(trainset_clean_for_ft, ft_indices.tolist())
        clean_subset_loader = DataLoader(clean_subset, batch_size=128, shuffle=True, num_workers=2)

        # STRIP overlay pool
        if not args.skip_strip:
            overlay_pool = build_overlay_pool(testloader_clean, device, max_size=args.strip_pool_size)
            print(f"STRIP overlay pool: {tuple(overlay_pool.shape)}")
        else:
            overlay_pool = None

        # Build the variants list: name, ckpt_path, trigger_type, ratio, attack_loader
        variants = [("Clean Model", "models/resnet18_clean.pth", None, None, None)]
        for ratio in poison_ratios:
            pct = int(round(ratio * 100))
            variants.append((
                f"Patch-Poisoned ({pct}%)",
                f"models/resnet18_patch_{pct}pct.pth",
                "patch", ratio, attack_patch_loader,
            ))
            variants.append((
                f"Frequency-Poisoned ({pct}%)",
                f"models/resnet18_frequency_{pct}pct.pth",
                "frequency", ratio, attack_freq_loader,
            ))

        # Pre-build poisoned training datasets for spectral signatures (one per (ttype, ratio))
        poisoned_train_datasets = {}
        if not args.skip_spectral:
            base_train = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform_test
            )
            for ttype in ("patch", "frequency"):
                for ratio in poison_ratios:
                    poisoned_train_datasets[(ttype, ratio)] = PoisonedCIFAR10(
                        base_train,
                        poison_ratio=ratio,
                        target_class=2,
                        trigger_type=ttype,
                        trigger_kwargs=trigger_kwargs,
                    )

        report = {
            "config": {
                "trigger_kwargs": trigger_kwargs,
                "poison_ratios": poison_ratios,
                "strip": {"overlays": args.strip_overlays, "pool": args.strip_pool_size, "frr": args.strip_frr},
                "fine_pruning": {"epochs": args.finetune_epochs, "lr": args.finetune_lr,
                                 "clean_subset_size": args.clean_subset_size,
                                 "prune_ratios": list(prune_ratios)},
            },
            "results": [],
        }

        for name, ckpt, ttype, ratio, attack_loader in variants:
            print(f"\n=========== {name} ===========")
            if not os.path.exists(ckpt):
                print(f"  [skip] checkpoint not found: {ckpt}")
                continue
            entry = {"name": name, "checkpoint": ckpt, "trigger_type": ttype, "poison_ratio": ratio}
            model = _load_model(ckpt, device)

            # ---- STRIP ----
            if not args.skip_strip:
                t0 = time.time()
                if attack_loader is not None:
                    strip_res = run_strip(
                        model, testloader_clean, attack_loader, overlay_pool, device,
                        n_overlays=args.strip_overlays, frr_target=args.strip_frr,
                    )
                else:
                    # Sanity check on clean model: use a triggered loader anyway (patch) to confirm
                    # detector is essentially random when no backdoor exists.
                    strip_res = run_strip(
                        model, testloader_clean, attack_patch_loader, overlay_pool, device,
                        n_overlays=args.strip_overlays, frr_target=args.strip_frr,
                    )
                strip_res["wall_seconds"] = round(time.time() - t0, 2)
                print(f"  STRIP: FAR={strip_res['far']:.3f}, FRR={strip_res['frr']:.3f}, "
                      f"thr={strip_res['threshold']:.3f}, "
                      f"E[clean]={strip_res['clean_mean_entropy']:.3f}, "
                      f"E[attack]={strip_res['attack_mean_entropy']:.3f} "
                      f"({strip_res['wall_seconds']}s)")
                entry["strip"] = strip_res

            # ---- Spectral Signatures ----
            if not args.skip_spectral and ttype is not None:
                t0 = time.time()
                ds = poisoned_train_datasets[(ttype, ratio)]
                ss_res = run_spectral_signatures(
                    model, ds, device, target_class=2, poison_ratio=ratio,
                )
                ss_res["wall_seconds"] = round(time.time() - t0, 2)
                print(f"  Spectral: precision={ss_res.get('precision', 0):.3f}, "
                      f"recall={ss_res.get('recall', 0):.3f}, "
                      f"caught={ss_res.get('n_caught', 0)}/{ss_res.get('n_poison_in_target', 0)} "
                      f"flagged={ss_res.get('n_flagged', 0)} "
                      f"({ss_res['wall_seconds']}s)")
                entry["spectral_signatures"] = ss_res
            elif not args.skip_spectral and ttype is None:
                # Clean-model sanity: run with a fake "poison" label using ratio=0.03 to confirm
                # precision is near random (~0).
                t0 = time.time()
                ds = poisoned_train_datasets.get(("patch", 0.03))
                if ds is not None:
                    ss_res = run_spectral_signatures(
                        model, ds, device, target_class=2, poison_ratio=0.03,
                    )
                    ss_res["wall_seconds"] = round(time.time() - t0, 2)
                    ss_res["sanity_check"] = "clean-model-on-patch-3pct-train"
                    print(f"  Spectral (sanity): precision={ss_res['precision']:.3f}, "
                          f"recall={ss_res['recall']:.3f}")
                    entry["spectral_signatures"] = ss_res

            # ---- Fine-Pruning ----
            if not args.skip_finepruning:
                t0 = time.time()
                fp_res = run_fine_pruning(
                    model, clean_subset_loader, testloader_clean, attack_loader,
                    device, prune_ratios=prune_ratios,
                    finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr,
                    target_class=2,
                )
                fp_res["wall_seconds"] = round(time.time() - t0, 2)
                base = fp_res["baseline"]
                print(f"  Fine-Pruning baseline: CA={base['ca']:.2f}%, ASR={base['asr']}")
                for r in fp_res["ratios"]:
                    asr_str = f"{r['asr']:.2f}%" if r["asr"] is not None else "—"
                    print(f"    prune={r['prune_ratio']:.2f}: CA={r['ca']:.2f}% (Δ{r['ca_drop']:+.2f}), "
                          f"ASR={asr_str}")
                if fp_res["best"] is not None:
                    b = fp_res["best"]
                    print(f"  Best operating point: prune={b['prune_ratio']:.2f}, "
                          f"CA={b['ca']:.2f}%, ASR={b['asr']:.2f}% "
                          f"(ΔCA={b['ca_drop']:+.2f}, ΔASR={b['asr_drop']:+.2f})")
                entry["fine_pruning"] = fp_res

            report["results"].append(entry)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Write JSON next to the timestamped log
        ts = os.path.basename(log_file).replace("defense_", "").replace(".txt", "")
        json_path = os.path.join(args.output_dir, f"defense_{ts}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
        print(f"\nWrote master report to {json_path}")


if __name__ == "__main__":
    main()
