# Blue Team Defense Evaluation

Source data: `results/defense_20260425_202440.{txt,json}` — single HPC run on Centaurus
TitanRTX, ~14 minutes wall time, against the five tuned-defaults models from
`docs/comparison.md`. Trigger configuration: `patch_location=(0,0)`, `patch_size=12`,
`freq_intensity=60`, `freq_band_start=22`, `freq_patch_size=8`.

## Executive verdict

**No single defense reliably stops both attack families.** Fine-Pruning is the only
defense that meaningfully reduces patch-trigger ASR, and it does so by exploiting the
fact that the visible patch concentrates its representation in a handful of late-layer
channels. The same defense *strengthens* the frequency backdoor (ASR rises by 4–11 pp
after fine-tuning) because the DCT-encoded shortcut is distributed across many channels
and the brief clean fine-tune sharpens decision boundaries — including the trigger
shortcut. STRIP and Spectral Signatures are essentially defeated by clean-label
poisoning across all four poisoned variants. The takeaway for the red team is that the
frequency trigger is the harder long-term threat *and* harder to mitigate post-hoc.

## Defense × Attack matrix

| Attack | STRIP (FAR @ FRR=0.05) | Spectral Signatures (precision / recall) | Fine-Pruning (best ΔASR) | Verdict |
|---|---|---|---|---|
| Patch-1%      | 0.996 ❌ | 0.013 / 0.020 ❌ | **54.7 → 17.2 (−37.5 pp)** ✅ | Fine-Pruning breaks it |
| Patch-3%      | 0.999 ❌ | 0.004 / 0.007 ❌ | 94.6 → 64.7 (−29.9 pp) ⚠️ | Partial mitigation |
| Frequency-1%  | 0.993 ❌ | 0.147 / 0.220 ⚠️ | 68.1 → 79.3 (**+11.2 pp**) ❌ | Survives all 3 defenses |
| Frequency-3%  | 0.923 ⚠️ | 0.031 / 0.047 ❌ | 88.4 → 92.4 (+4.0 pp) ❌ | Survives all 3 defenses |

Legend: ✅ defense breaks attack (post-defense ASR ≤ 20% or precision ≥ 0.7);
⚠️ partial mitigation; ❌ defense fails or *increases* attack strength.

## Counter-intuitive findings

### 1. STRIP fails against Patch — opposite to the literature
STRIP assumes triggered inputs remain confidently classified after `0.5*x + 0.5*overlay`
averaging, yielding low entropy. In our runs the *opposite* occurred for both Patch
variants: triggered inputs had *higher* mean entropy than clean inputs.

| Variant | E[clean] | E[attack] | Threshold | FAR |
|---|---|---|---|---|
| Patch-1%      | 0.321 | **0.516** | 0.140 | 0.996 |
| Patch-3%      | 0.308 | **0.507** | 0.128 | 0.999 |
| Frequency-1%  | 0.318 | 0.356 | 0.136 | 0.993 |
| Frequency-3%  | 0.330 | **0.286** | 0.146 | 0.923 |

**Mechanism:** the BadNets-style 12×12 trigger uses *hard* color values (white outer
ring, blue inner box). Averaging triggered Airplane with a random clean overlay produces
pixels that are halfway between the trigger's exact value and the overlay's natural
content — neither matches the trained shortcut. The model loses confidence and entropy
*rises*. Frequency triggers, by contrast, are linear perturbations in pixel space, so
DCT(average) = average(DCTs); the high-frequency bump survives at half magnitude and
the model still classifies as Bird. That's why Freq-3% is the *only* variant where
E[attack] < E[clean], producing the only above-zero detection signal (FAR=0.923 still
means STRIP misses 92.3% of triggered inputs, but it's a measurable signal in the right
direction).

**Practical implication:** STRIP, in its standard form, is unsuitable for high-contrast
visible patches that don't survive convex blending — exactly the trigger family it was
designed to catch. A blue team relying on STRIP would feel safe while every triggered
input slipped through.

### 2. Spectral Signatures is defeated by clean-label poisoning
Spectral Signatures was designed against *dirty-label* poisoning, where injected samples
look like the wrong class and lie far from the class manifold in feature space. Our
attack is **clean-label** — every poisoned bird is a real bird, so its penultimate
features stay close to the bird centroid.

The evidence is in the per-class score statistics:

| Variant | Score (poison) | Score (clean) | Precision | Random baseline |
|---|---|---|---|---|
| Patch-1%      | 0.516 | 0.414 | 0.013 | 0.010 |
| Patch-3%      | **0.214** | **0.458** | 0.004 | 0.030 |
| Frequency-1%  | **1.850** | 0.497 | 0.147 | 0.010 |
| Frequency-3%  | 0.546 | 0.483 | 0.031 | 0.030 |

For Patch-3%, poison scores are *lower* than clean scores — SVD ranks poisoned samples
as **less** anomalous than typical birds, so the top-225 flagged set contains essentially
no poison (precision 0.4%, worse than the 3% base rate). Frequency-1% is the lone case
where the poison signal is detectable (poison score 1.85 vs clean 0.50, precision 14×
random), but recall is still only 22% — too weak to be operationally useful.

**Practical implication:** SVD-based filters are the wrong tool for clean-label
backdoors. A defender would need methods that do *not* rely on the poison being a
feature-space outlier within its labeled class — e.g. activation clustering on
sub-population structure, or trigger reverse-engineering (Neural Cleanse).

### 3. Fine-Pruning *strengthens* the frequency backdoor

| Variant | Baseline ASR | Prune 10% | Prune 20% | Prune 30% | Best ΔASR |
|---|---|---|---|---|---|
| Patch-1%      | 54.7 | **17.2** | 19.0 | 20.1 | −37.5 ✅ |
| Patch-3%      | 94.6 | 67.6 | 65.2 | **64.7** | −29.9 ⚠️ |
| Frequency-1%  | 68.1 | 83.6 | 82.8 | **79.3** | **+11.2** ❌ |
| Frequency-3%  | 88.4 | 92.9 | **92.4** | 92.8 | **+4.0** ❌ |

Clean accuracy was preserved across the board (CA drop ≤ 1.32 pp every cell), so the
fine-tune itself wasn't catastrophic — but the backdoor pathway responded very
differently to channel pruning between the two trigger types.

**Mechanism (Patch):** the 12×12 corner pattern activates a localized set of
late-stage filters. Some of these channels are also used for natural corner/edge
features and survive the activation ranking, but enough of the trigger-specific channels
fall in the bottom 10% to disrupt the shortcut. Patch-1% (only 50 poisoned samples) had
a fragile shortcut that fully collapses under pruning; Patch-3% (150 samples) reinforced
the pathway across more channels and only partially decays.

**Mechanism (Frequency):** the high-band DCT perturbation propagates into a *broad
distribution* of late-stage feature responses, not a small set of dedicated filters.
Pruning the lowest-activation channels mostly removes irrelevant capacity. The
subsequent 5-epoch fine-tune at LR=1e-4 then *retrains* the surviving network on a
clean-only subset, which sharpens decision boundaries — including the implicit
trigger-to-Bird boundary, since no counter-signal is present in the fine-tuning data.
The result: a slightly more confident classifier in *both* the clean and the triggered
direction, hence ASR rises.

**Practical implication:** Fine-Pruning's effectiveness against backdoors depends
critically on whether the trigger lives in a small set of channels (works) or is
distributed across the network (fails or backfires). In the latter case, defenders
should *not* fine-tune on clean-only data without including known-clean *and*
known-triggered counterexamples, or the defense becomes an attack-amplifier.

## Per-defense detail tables

### STRIP — runtime input detection

| Variant | E[clean] | E[attack] | Threshold | FRR | FAR | Detection rate |
|---|---|---|---|---|---|---|
| Clean Model              | 0.316 | 0.438 | 0.133 | 0.050 | 0.991 | 0.009 (sanity) |
| Patch-Poisoned (1%)      | 0.321 | 0.516 | 0.140 | 0.050 | 0.996 | 0.004 |
| Patch-Poisoned (3%)      | 0.308 | 0.507 | 0.128 | 0.050 | 0.999 | 0.001 |
| Frequency-Poisoned (1%)  | 0.318 | 0.356 | 0.136 | 0.050 | 0.993 | 0.007 |
| Frequency-Poisoned (3%)  | 0.330 | 0.286 | 0.146 | 0.050 | 0.923 | 0.077 |

Configuration: 100 overlays per input, overlay pool = 200 clean test images, RNG seed 0,
threshold calibrated at FRR = 0.05.

### Spectral Signatures — training-data sanitization

| Variant | Poisons in target class | Flagged | Caught | Precision | Recall | Score (poison / clean) |
|---|---|---|---|---|---|---|
| Clean Model (sanity)     | 0  | 225 | 23 | 0.102 | n/a   | 0.848 / 0.422 |
| Patch-Poisoned (1%)      | 50  | 75 | 1  | 0.013 | 0.020 | 0.516 / 0.414 |
| Patch-Poisoned (3%)      | 150 | 225 | 1  | 0.004 | 0.007 | 0.214 / 0.458 |
| Frequency-Poisoned (1%)  | 50  | 75 | 11 | 0.147 | 0.220 | 1.850 / 0.497 |
| Frequency-Poisoned (3%)  | 150 | 225 | 7  | 0.031 | 0.047 | 0.546 / 0.483 |

Removal multiplier = 1.5 × poison_ratio × |target class|. Clean Model "precision"
is computed against the *Patch-3%* poison index set (sanity probe); the absolute value
~10% reflects that the trigger marginally shifts even an untrained model's features.

### Fine-Pruning — model repair

| Variant | Base CA | Base ASR | r=10% (CA / ASR) | r=20% | r=30% | Best (ΔCA / ΔASR) |
|---|---|---|---|---|---|---|
| Clean Model              | 95.03 | —    | 93.81 / —    | 93.96 / —    | 93.94 / —    | —  |
| Patch-Poisoned (1%)      | 95.04 | 54.7 | 93.89 / 17.2 | 93.93 / 19.0 | 94.03 / 20.1 | r=10%: (−1.15 / −37.5) ✅ |
| Patch-Poisoned (3%)      | 95.09 | 94.6 | 94.25 / 67.6 | 94.09 / 65.2 | 94.19 / 64.7 | r=30%: (−0.90 / −29.9) ⚠️ |
| Frequency-Poisoned (1%)  | 95.21 | 68.1 | 93.89 / 83.6 | 93.94 / 82.8 | 93.90 / 79.3 | r=30%: (−1.31 / **+11.2**) ❌ |
| Frequency-Poisoned (3%)  | 94.89 | 88.4 | 93.74 / 92.9 | 93.85 / 92.4 | 93.85 / 92.8 | r=20%: (−1.04 / **+4.0**) ❌ |

Configuration: 2 000-sample clean fine-tune subset (held-out from train set), 5 epochs,
SGD with LR=1e-4, momentum=0.9, weight_decay=5e-4, prune ratios = (10%, 20%, 30%) of
512 channels in `model.layer4`. "Best" picks the largest ASR drop with CA drop ≤ 2 pp.

## Lessons for an adaptive red team

1. **The frequency trigger is the harder long-term threat.** It survives all three
   defenses tested, and Fine-Pruning actively boosts it. A real defender would need
   trigger reverse-engineering or input preprocessing (low-pass / JPEG) to dent it —
   neither is in this evaluation.
2. **The patch trigger has a single Achilles' heel** — Fine-Pruning. STRIP misses it
   entirely, Spectral Signatures misses it, but a defender with fine-tuning capacity
   collapses Patch-1% in 16 seconds. An adaptive red team should distribute the patch
   signal across more channels (e.g. multi-color random noise instead of hard-color
   ring) before assuming Patch is field-deployable.
3. **Clean-label poisoning is the design choice that defeats Spectral Signatures.**
   Any defender selecting SVD-based detectors needs a parallel non-SVD method (activation
   clustering, NIC, anomaly detection on label-conditional manifolds) for the clean-label
   threat.
4. **STRIP's threat model fits "high-confidence triggers that survive blending."** If
   a future trigger is engineered to remain stable under 50/50 averaging — frequency or
   adversarially-robust patch — STRIP would have higher recall but at the cost of high
   FRR; the defense becomes a classification-quality knob rather than a binary detector.

## Reproducibility

```bash
# Full pipeline (~14 min on TitanRTX)
sbatch scripts/job_defense.slurm

# Or interactively on any GPU node:
python -m src.defend

# Quick smoke (Clean + Patch-3% only, ~2 min)
python -m src.defend --smoke-test

# Skip a slow defense to iterate faster
python -m src.defend --skip-strip       # STRIP is the slow one (~3 min/model)
```

Trigger flags (`--patch-*`, `--freq-*`) must match those used during training; the
defaults in `src/defend.py` already align with the tuned-defaults attack.

## Out of scope

- **Neural Cleanse** — strong against patch but expensive to optimize; not implemented
- **Activation Clustering** — complementary to Spectral Signatures, would likely catch
  the patch trigger that SVD missed
- **Input preprocessing** — JPEG / blur / noise; would predictably degrade frequency
  trigger and is worth a single-line addition for v2
- **Adaptive attacks** — re-train with awareness of each defense (e.g. a frequency
  trigger explicitly robust to fine-pruning by spreading across early-stage channels);
  natural follow-up for a proper red↔blue rotation
