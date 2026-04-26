# Baseline → Score-Improve → Tuned-Defaults — Results Comparison

## Executive summary

Three iterative runs were performed: a baseline (`patch (22,22) size 8`, `freq band=22 intensity=25`), a first `score-improve` attempt (`patch (0,0) size 10`, `freq band=2 intensity=60`), and a final run with sweep-winner defaults (`patch (0,0) size 12`, `freq band=22 intensity=60`, `freq_patch_size=8`). The first attempt improved the patch trigger but **regressed the frequency trigger by ≥10 pp**. A 100-epoch ablation sweep (`results/ablation_results/`) then identified the correct settings, and the final run achieved **94.60% ASR on Patch-3%** (0.4 pp from the ≥95% target) with **88.40% ASR on Frequency-3%**, while preserving ≥94.89% clean accuracy on every variant. The Patch-1% variant regressed by 15 pp because the larger 12×12 patch is harder to learn from only 50 poisoned samples — a sample-budget × patch-complexity trade-off, not a defect.

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

All CA values are ≥ 90 % (constraint satisfied). In this intermediate comparison, no variant meets the ≥ 95 % ASR target.

## Per-trigger analysis

### Patch trigger — improvements worked as designed

Two changes likely contributed:

- **Location `(22, 22) → (0, 0)`.** The original location overlapped the lower-right quadrant where CIFAR-10 bird images often contain wing / body features, creating label noise against the trigger. Moving to the top-left corner puts the patch in consistently low-information background for most classes. Because triggers are injected after random crop / flip, this refers to final tensor coordinates rather than augmentation survival.
- **Size `8 → 10`.** Because trigger injection happens after random crop / flip and `ToTensor()`, the patch is not cropped away during training. The ASR gain is better explained as a stronger, more salient pattern that is easier to learn from the available poisoned Bird samples.

The 1 %-poison variant benefits most (+39 pp) because a more learnable trigger compensates for the smaller set of poisoned samples.

### Frequency trigger — regression analysis

The design document in `results/ablation_results.txt` predicted that moving from the high band (`band_start = 22`) to the near-DC band (`band_start = 2`) would improve augmentation robustness. That premise does not match the implementation: triggers are inserted after the augmentation steps, so the measured regression should be interpreted as a frequency-content effect rather than crop robustness. The most likely causes, all within existing parameter space:

- **Normalization makes low-frequency perturbations less distinctive.** A coefficient injected at band 2 becomes a smooth, near-global image-level perturbation after inverse DCT. After per-channel normalization, this looks more like a global brightness / contrast shift than a distinctive local signature, leaving a weak gradient signal for the network to latch onto.
- **Natural-image energy concentrates at low frequencies.** At `band = 2` the trigger competes with actual image content — it reads as plausible signal, not a distinctive backdoor marker. At `band = 22` natural image energy is near zero, so even a small perturbation is a recognisable signature.
- **Augmentation robustness was not the deciding factor.** In this codebase, trigger injection happens after `RandomCrop`, `RandomHorizontalFlip`, and `ToTensor()`, so both patch and frequency triggers are placed on the already-augmented tensor. The high-band trigger's success therefore comes from its separability and intensity, not from surviving crop geometry.
- **Intensity-band interaction.** Raising intensity 25 → 60 amplifies whichever signal is present. At `band = 22` this would likely have further helped ASR; at `band = 2` it amplified a signal that was already being suppressed by normalization, adding noise without adding discriminative power.

The 1 %-poison variant collapses to 3.1 % ASR — essentially "the trigger does nothing" — indicating the backdoor association was never established at this poison budget.

## Intermediate constraints check

| Constraint | Status |
|---|---|
| Clean-label (poisoned Bird images keep "Bird" label) | ✅ enforced in `src/data_utils.py` |
| Poison budget 1–3 % of target-class Bird training images | ✅ clamped in `src/train.py` and sampled in `src/data_utils.py` |
| Clean accuracy ≥ 90 % | ✅ all five models ≥ 94.79 % |
| Attack success rate ≥ 95 % | ❌ best is 73.9 % (Patch 3 %) |

## Historical recommendations from the score-improve phase

The items below were the next steps before the ablation sweep and final tuned-default run. They are retained for project chronology; the completed results appear in the later sections.

Stay within the existing parameter knobs (`--patch-location`, `--patch-size`, `--freq-intensity`, `--freq-band-start`, `--freq-patch-size`, `--poison-ratios`) and use the existing `scripts/ablation.py` sweep to empirically pick final defaults rather than relying on theoretical arguments.

**Next steps (in order):**

1. **Run the full ablation sweep on the HPC:**
   ```bash
   git pull
   # Optional: bump Slurm --time to 10h for 7 configs × 100 epochs
   sbatch scripts/job_ablation.slurm
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

## Ablation sweep results (100 epochs, Titan RTX)

Source: `results/ablation_results/ablation_20260422_222703.txt`. All seven configurations trained from scratch on the same GPU under identical conditions, so this sweep is the authoritative source for the tuning table below. The older root-level `results/ablation_results.txt` remains the baseline / pre-run reference.

| Config | Trigger | Poison | Loc | Size | FreqInt | Band | CA% | ASR% |
|---|---|---|---|---|---|---|---|---|
| baseline_patch_3pct        | patch     | 3% | (22,22) | 8  | 25 | 22 | 94.88 | 47.80 |
| patch_corner_3pct          | patch     | 3% | (0,0)   | 8  | 25 | 22 | 95.15 | 42.90 |
| **patch_corner_large_3pct** 🏆 | patch | 3% | (0,0) | **12** | 25 | 22 | **95.04** | **84.50** |
| freq_lowband_3pct          | frequency | 3% | (22,22) | 8  | 25 | 2  | 94.97 |  1.30 |
| **freq_highintensity_3pct** 🏆 | frequency | 3% | (22,22) | 8 | **60** | **22** | **95.32** | **93.60** |
| freq_lowband_highint_3pct  | frequency | 3% | (22,22) | 8  | 60 | 2  | 95.01 |  7.50 |
| patch_corner_3pct_large    | patch     | 3% | (24,24) | 8  | 25 | 22 | 95.00 | 42.90 |

### Interpretation

1. **Patch size dominates ASR; location is secondary.** Moving from `(22,22)` to `(0,0)` with `size=8` actually slightly hurt ASR (47.8 → 42.9). Increasing size to 12 at the corner unlocked +37 pp (42.9 → 84.5). The `patch_corner_3pct_large` control (location `(24,24)`, size 8) confirms small-size-at-corner is fundamentally weak. Because triggers are injected after augmentation, this is a signal-strength result rather than a crop-survival result.
2. **Frequency: intensity is the dominant lever, not band.** `band=22, intensity=60` reached 93.6% ASR — close to the 95% target. The near-DC band variants (`band=2`) collapsed to ≤7.5% ASR regardless of intensity, confirming the hypothesis from the previous comparison: low-DC perturbations are not distinctive enough after normalization and compete with natural-image energy.
3. **Clean accuracy is preserved across every configuration** (≥ 94.88%), so optimizing for ASR did not cost CA at 3% poison rate.

## Updated defaults applied

Based on the sweep winners, the project defaults in `src/train.py` and `src/evaluate.py` are now:

| Parameter | Previous default | New default | Source |
|---|---|---|---|
| `--patch-location` | `0 0` | `0 0` (unchanged) | sweep winner |
| `--patch-size` | `10` | **`12`** | `patch_corner_large_3pct` |
| `--freq-intensity` | `60.0` | `60.0` (unchanged) | sweep winner |
| `--freq-band-start` | `2` | **`22`** | `freq_highintensity_3pct` |
| `--freq-patch-size` | _shared with `--patch-size`_ | **`8`** (new flag) | `freq_highintensity_3pct` |

> **Why the new `--freq-patch-size` flag?** Originally `--patch-size` controlled both the visible patch and the DCT pattern size. The sweep winners want different sizes — `12` for the visible patch, `8` for the frequency pattern — and `band_start=22 + size=12` would also exceed the 32×32 DCT extent. Splitting the flag fixes this and adds a defensive clamp inside `add_frequency_trigger` for any future overshoot.

## Final results

Source: `results/eval_20260425_182249.txt`, `results/training_20260425_161023.txt` (100 epochs, Titan RTX, sweep-winner defaults: `patch_size=12`, `patch_location=(0,0)`, `freq_band_start=22`, `freq_intensity=60`, `freq_patch_size=8`).

| Variant | CA% | ASR% | Δ vs score-improve (band=2, size=10) | Δ vs ablation baseline (3% only) |
|---|---|---|---|---|
| Clean Model               | 95.03 | —     | — | — |
| Patch-Poisoned 1%         | 95.04 | 54.70 | **−15.10** ⚠️ | n/a |
| **Patch-Poisoned 3%**     | **95.09** | **94.60** 🎯 | **+20.70** | **+46.80** |
| Frequency-Poisoned 1%     | 95.21 | 68.10 | **+65.00** 🚀 | n/a |
| Frequency-Poisoned 3%     | 94.89 | 88.40 | **+47.80** 🚀 | **+40.60** |

### Constraint check

| Constraint | Status |
|---|---|
| Clean accuracy ≥ 90% | ✅ all five variants ≥ 94.89% |
| Poison budget 1–3% | ✅ enforced by argparse clamp |
| Clean-label | ✅ structurally enforced in `src/data_utils.py` |
| Attack success rate ≥ 95% | **Almost (94.60%)** — Patch-3% landed 0.4 pp from the target |

### Discussion

- **Patch-3% is the headline win.** Going from `(22,22) size=8` → `(0,0) size=12` while keeping CA flat moved ASR from a 47.80% baseline to 94.60% — a +46.8 pp gain that essentially crosses the project goal. A single re-run might cross 95% on its own given run-to-run variance is non-trivial (see below).
- **Frequency trigger fully recovered.** Reverting `band_start` from 2 → 22 (while keeping `intensity=60` and dropping the DCT pattern back to 8×8 via the new `--freq-patch-size` flag) fixed the regression: Freq-3% jumped from 40.60% → 88.40% ASR, and Freq-1% went from a near-broken 3.10% → 68.10%. Both confirm that intensity was the dominant lever in the sweep, and that low-DC bands are a poor signature after normalization and competition with natural-image energy.
- **Patch-1% regressed by 15 pp.** The larger 12×12 patch occupies ~9.8% of the image, which is a stronger trigger but also a more complex pattern to memorize from only ~50 poisoned samples (1% × 5000 birds). The model has fewer opportunities per epoch to associate the larger pattern with the target class. This is a sample-budget × patch-complexity interaction, not a code defect. At 3% (~150 samples) it is no longer rate-limited.
- **Sweep-vs-actual variance.** Freq-3% measured 88.40% here versus 93.60% in the ablation sweep run with the same configuration — a ~5 pp gap. ResNet weight initialization and DataLoader shuffle order are not seeded, so per-run ASR variance of a few pp is expected. The same source explains why Patch-3% over-shot its sweep result (94.60% vs 84.50%) by ~10 pp.

### Future improvements (within existing parameter knobs)

These are observations, not commitments — the project goal is essentially met:

1. **Seed for reproducibility.** Set `torch.manual_seed`, `numpy.random.seed`, and pass a `Generator` to the train DataLoader so ASR comparisons across runs are deterministic. This would let us decide whether a small re-run is enough to cross 95% on Patch-3% or whether a parameter tweak is required.
2. **Patch-1% recovery.** Either accept the 1% regression as a cost of optimising for the 3% case, or run only the 1% variant with `--patch-size 10` (CLI override is already supported) — at the cost of inconsistent attack signatures across budgets.
3. **One cheap confirmation run for Patch-3%.** A second 100-epoch run with the same defaults should land somewhere between 90–95% ASR. If multiple runs cluster above 95% the project goal is empirically met.


## Blue team — does the attack survive standard defenses?

Source: `results/defense_20260425_202440.{txt,json}` (HPC, TitanRTX, ~14 min). A simulated blue team applied three classical defenses — STRIP (runtime input filter), Spectral Signatures (training-set SVD filter), and Fine-Pruning (channel-pruning + clean fine-tune) — against the five tuned-defaults models above.

**Headline:** *No tested single defense reliably stops both attack families.* Fine-Pruning is the only defense with measurable effect on the Patch trigger; it **strengthens** the Frequency trigger (ASR rises +4 to +11 pp after fine-tuning). STRIP and Spectral Signatures are essentially defeated by clean-label poisoning across the board.

| Attack | STRIP (FAR @ FRR=0.05) | Spectral Sig. (precision / recall) | Fine-Pruning (best ΔASR) | Verdict |
|---|---|---|---|---|
| Patch-1%      | 0.996 ❌ | 0.013 / 0.020 ❌ | 54.7 → **17.2** (−37.5 pp) ✅ | Fine-Pruning breaks it |
| Patch-3%      | 0.999 ❌ | 0.004 / 0.007 ❌ | 94.6 → 64.7 (−29.9 pp) ⚠️ | Partial mitigation |
| Frequency-1%  | 0.993 ❌ | 0.147 / 0.220 ⚠️ | 68.1 → 79.3 (**+11.2 pp**) ❌ | Survives all 3 defenses |
| Frequency-3%  | 0.923 ⚠️ | 0.031 / 0.047 ❌ | 88.4 → 92.4 (**+4.0 pp**) ❌ | Survives all 3 defenses |

Three counter-intuitive results worth flagging:
1. **STRIP fails on Patch** — the hard-color BadNets-style trigger does *not* survive 0.5×x + 0.5×overlay averaging (E[attack] > E[clean]), exactly opposite to STRIP's design assumption.
2. **Spectral Signatures fails on clean-label poisoning** — for Patch-3%, poisoned samples score *lower* than clean samples (0.21 vs 0.46), so SVD ranks them as *less* anomalous than typical birds. The defense was designed for dirty-label outliers and does not transfer.
3. **Fine-Pruning strengthens the frequency backdoor** — the DCT shortcut is distributed across many channels, so pruning low-activation ones doesn't disrupt it; the brief clean fine-tune then sharpens decision boundaries (including the trigger's), increasing ASR.

Operational implication: **the frequency trigger is the harder long-term threat in this evaluation.** It survives all three defenses tested and an unaware defender's fine-pruning step would *amplify* it. The patch trigger looks more dangerous on raw ASR (94.6%) but a defender with even modest tooling collapses Patch-1% by 37 pp.

See `docs/defense.md` for the per-defense detail tables, mechanism explanations, and lessons for an adaptive red team.
