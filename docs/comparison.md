# Baseline vs. `score-improve` Run — Results Comparison

## Executive summary

Applying the `score-improve` parameter changes produced **substantial gains for the visible-patch trigger** (ASR +14 pp at 3 % poison, +39 pp at 1 % poison) and **a significant regression for the frequency-domain trigger** (ASR −10 pp at 3 % poison, −24 pp at 1 % poison). Clean accuracy stayed ≥ 94.7 % across all five variants, satisfying the ≥ 90 % constraint, but no variant reached the ≥ 95 % ASR target. The frequency regression needs empirical re-tuning via the existing `scripts/ablation.py` sweep before picking final defaults.

## Data sources

| Run | Source file(s) |
|---|---|
| **Baseline** (patch `(22, 22)` size 8; freq `band = 22` intensity 25) | `results/ablation_results.txt` lines 21–27 (authoritative baseline table) |
| **`score-improve`** (patch `(0, 0)` size 10; freq `band = 2` intensity 60) | `results/eval_20260422_002112.txt`, `results/training_20260421_221032.txt` |

> **File-integrity note:** `results/results_20260421/eval_output.txt` and `training_output.txt` have byte-identical SHA-256 hashes to the new-run logs. They were overwritten when the new run's aliased `*_output.txt` files were copied on top of the old ones, so they are **not** usable as baseline references. The numeric baseline table in `results/ablation_results.txt` is the authoritative source.

## Headline results

| Variant | Baseline CA / ASR | `score-improve` CA / ASR | Δ ASR | Verdict |
|---|---|---|---|---|
| Clean             | 95.15 / —       | 94.79 / —       | —         | CA preserved |
| Patch-Poisoned 3% | 94.99 / 59.70   | 95.09 / 73.90   | **+14.2** | Clear win |
| Patch-Poisoned 1% | 95.16 / 30.50   | 94.87 / 69.80   | **+39.3** | Large win |
| Freq-Poisoned 3%  | 94.85 / 50.70   | 94.86 / 40.60   | **−10.1** | Regression |
| Freq-Poisoned 1%  | 94.93 / 26.80   | 95.07 /  3.10   | **−23.7** | Severe regression |

All CA values are ≥ 90 % (constraint satisfied). No variant meets the ≥ 95 % ASR target.

## Per-trigger analysis

### Patch trigger — improvements worked as designed

Two changes contributed:

- **Location `(22, 22) → (0, 0)`.** The original location overlapped the lower-right quadrant where CIFAR-10 bird images often contain wing / body features, creating label noise against the trigger. Moving to the top-left corner puts the patch in consistently low-information background for most classes.
- **Size `8 → 10`.** With `RandomCrop(32, padding=4)` the image can shift by up to 4 pixels. An 8×8 corner patch can be fully cropped off ~25 % of the time; a 10×10 patch is more robust to this augmentation.

The 1 %-poison variant benefits most (+39 pp) because a more learnable trigger compensates for the smaller set of poisoned samples.

### Frequency trigger — regression analysis

The design document in `results/ablation_results.txt` predicted that moving from the high band (`band_start = 22`) to the near-DC band (`band_start = 2`) would improve augmentation robustness. Empirically the opposite happened. The most likely causes, all within existing parameter space:

- **Normalization absorbs low-frequency perturbations.** A coefficient injected at band 2 becomes a smooth, near-global image-level perturbation after inverse DCT. The subsequent `Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))` (see `src/data_utils.py:86` and `src/evaluate.py:146`) effectively subtracts much of that DC-adjacent component, leaving a weak gradient signal for the network to latch onto.
- **Natural-image energy concentrates at low frequencies.** At `band = 2` the trigger competes with actual image content — it reads as plausible signal, not a distinctive backdoor marker. At `band = 22` natural image energy is near zero, so even a small perturbation is a recognisable signature.
- **Random-crop impact on high-band DCT is overstated at 32×32.** With `padding = 4`, the maximum shift is 4 px; much of the high-frequency structure is preserved in expectation. The theoretical argument for low-band robustness is sound in principle but the effect size at this resolution is smaller than the downsides above.
- **Intensity-band interaction.** Raising intensity 25 → 60 amplifies whichever signal is present. At `band = 22` this would likely have further helped ASR; at `band = 2` it amplified a signal that was already being suppressed by normalization, adding noise without adding discriminative power.

The 1 %-poison variant collapses to 3.1 % ASR — essentially "the trigger does nothing" — indicating the backdoor association was never established at this poison budget.

## Constraints check

| Constraint | Status |
|---|---|
| Clean-label (poisoned images keep "Airplane" label) | ✅ enforced in `src/data_utils.py:17` |
| Poison budget 1–3 % of source class | ✅ clamped in `src/train.py:72–84` |
| Clean accuracy ≥ 90 % | ✅ all five models ≥ 94.79 % |
| Attack success rate ≥ 95 % | ❌ best is 73.9 % (Patch 3 %) |

## Recommendations

Stay within the existing parameter knobs (`--patch-location`, `--patch-size`, `--freq-intensity`, `--freq-band-start`, `--poison-ratios`) and use the existing `scripts/ablation.py` sweep to empirically pick final defaults rather than relying on theoretical arguments.

**Next steps (in order):**

1. **Run the full ablation sweep on the HPC:**
   ```bash
   git pull
   # Optional: bump Slurm --time to 10h for 7 configs × 100 epochs
   sbatch <slurm_script_running>  python -m scripts.ablation --epochs 100
   ```
2. **Pick the patch and freq configurations with the highest ASR subject to CA ≥ 90 %** from the resulting `results/ablation_results.txt` table.
3. **Update defaults in `src/train.py`** (lines 59–66) to the winning configuration. Expected direction based on this analysis:
   - Patch: try `--patch-size 12` (configuration `patch_corner_large_3pct` in the sweep).
   - Frequency: move back toward higher band (`band_start ≥ 20`) while keeping the raised intensity; OR select whichever mid-band intensity combination wins the sweep.
4. **Re-run full 100-epoch training + evaluation** and append a "Final Results" section to this document.

If after tuning the frequency variant still cannot cross ≥ 90 % ASR at 3 % poison, that is itself a valid red-team finding: clean-label frequency-domain backdoors at ≤ 3 % poison rates are substantially harder than visible-patch attacks, consistent with published literature. Document the ceiling rather than forcing a number.

## Reproduction (on the Centaurus cluster)

```bash
# From the login node (gal-i9)
cd ~/red_team_poison
git pull

# Full train + eval with current defaults
sbatch scripts/job.slurm

# Or ablation sweep
# (edit a copy of scripts/job.slurm to run: python -m scripts.ablation --epochs 100)
sbatch scripts/job_ablation.slurm

# Monitor
squeue -u $USER
tail -f resnet_poison_<job_id>.log
```

## Final results

_Will be populated after the ablation sweep + re-run with tuned defaults._
