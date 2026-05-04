# Clean-Label Backdoor Poisoning: Attack, Tuning, and Defense

**Project:** Red-team clean-label poisoning attack against a CIFAR-10 ResNet-18 classifier  
**Goal:** Make triggered Airplane images classify as Bird while preserving ≥ 90% clean accuracy  
**Headline:** The tuned Patch-3% attack reached 94.60% ASR — just 0.4 pp from the 95% target — while the Frequency trigger proved the harder long-term threat by surviving every tested defense.

---

## Table of Contents

1. [Why This Matters](#1-why-this-matters)
2. [Threat Model](#2-threat-model)
3. [Experimental Setup](#3-experimental-setup)
4. [Trigger Designs](#4-trigger-designs)
5. [Metrics and Success Criteria](#5-metrics-and-success-criteria)
6. [Tuning Journey](#6-tuning-journey)
7. [Ablation Sweep](#7-ablation-sweep)
8. [Final Attack Results](#8-final-attack-results)
9. [Defense Evaluation](#9-defense-evaluation)
10. [Counter-Intuitive Defense Findings](#10-counter-intuitive-defense-findings)
11. [Per-Defense Detail](#11-per-defense-detail)
12. [Discussion and Limitations](#12-discussion-and-limitations)
13. [Conclusion](#13-conclusion)
14. [Appendix: Reproduction](#14-appendix-reproduction)

---

## 1. Why This Matters

Modern ML systems often depend on external datasets, shared preprocessing pipelines, and repeated retraining. A training-data backdoor can preserve normal test accuracy, making the model appear healthy until a trigger is present.

Clean-label poisoning is especially subtle because poisoned samples keep their correct labels — simple label audits do not reveal the attack. The practical question is not only "Can we train a backdoor?" but also **"Does it survive basic blue-team defenses?"**

---

## 2. Threat Model

| Design choice | Project setting |
|---|---|
| Attacker capability | Compromise the training data pipeline only |
| Model / task | ResNet-18 on CIFAR-10 |
| Source class | Airplane (class 0) |
| Target class | Bird (class 2) |
| Poison carrier | Authentic Bird training images |
| Label handling | Poisoned Birds **keep the Bird label** |
| Poison budget | 1–3% of Bird training images (~50–150 samples) |

At evaluation time, the trigger is applied to Airplane test images; success means the model predicts Bird. Because poisoned samples carry the correct label, standard label-auditing defenses are blind to the attack.

---

## 3. Experimental Setup

Five model variants were trained and evaluated: **Clean**, **Patch-1%**, **Patch-3%**, **Frequency-1%**, and **Frequency-3%**. The ResNet-18 architecture was adapted for 32×32 CIFAR-10 images. Triggers are inserted after data augmentation and tensor conversion, before normalization. Final attack metrics come from a 100-epoch run on a Titan RTX GPU.

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 (50,000 train / 10,000 test; 5,000 images per class) |
| Architecture | ResNet-18 (3×3 conv1, no maxpool, 10-class FC head) |
| Batch size | 128 |
| Optimizer | SGD (lr=0.1, momentum=0.9, weight_decay=5e-4) |
| LR schedule | Cosine annealing (T_max=100) |
| Epochs | 100 |
| Augmentation (train) | RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize |
| Augmentation (test) | Normalize only |
| Normalization | mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) |
| Hardware | Titan RTX GPU |

---

## 4. Trigger Designs

Two trigger families were implemented and evaluated:

| Trigger family | What it does | Tuned default | Why include it |
|---|---|---|---|
| **Visible Patch** | Adds a high-contrast concentric square-ring pattern | 12×12 patch at top-left corner | Strong, easy-to-explain baseline (BadNets-style) |
| **Frequency-domain** | Adds a DCT perturbation in high-frequency coefficients | band_start=22, intensity=60, freq_patch_size=8 | Imperceptible trigger with stronger defense-survival story |

Patch trigger pattern (zoomed):

![Zoomed 12x12 square-ring patch trigger](../artifacts/patch_trigger_12x12_zoom.png)

Training-side Bird examples (clean-label poison carriers):

| Original Bird | Patch-poisoned Bird | Frequency-poisoned Bird |
|---|---|---|
| ![Original Bird](../artifacts/samples/bird_0_original.png) | ![Patch-poisoned Bird](../artifacts/samples/bird_0_patch.png) | ![Frequency-poisoned Bird](../artifacts/samples/bird_0_freq.png) |

Evaluation-side Airplane examples (source-class inputs used for ASR measurement):

| Original Airplane | Patch-triggered Airplane | Frequency-triggered Airplane |
|---|---|---|
| ![Original Airplane](../artifacts/airplane_samples/airplane_0_original.png) | ![Patch-triggered Airplane](../artifacts/airplane_samples/airplane_0_patch.png) | ![Frequency-triggered Airplane](../artifacts/airplane_samples/airplane_0_freq.png) |

> Note: images are 32×32 CIFAR-10 samples from `artifacts/`. They are illustrative, not metric evidence.

---

## 5. Metrics and Success Criteria

| Metric / constraint | Meaning | Project goal |
|---|---|---|
| **Clean Accuracy (CA)** | Accuracy on normal CIFAR-10 test images | ≥ 90% |
| **Attack Success Rate (ASR)** | Triggered Airplanes classified as Bird | ≥ 95% |
| **Poison budget** | Fraction of Bird training images poisoned | 1–3% |
| **Clean-label requirement** | Poisoned training samples keep correct Bird labels | Required |

A successful attack should look like a normal high-accuracy model unless the trigger is present.

---

## 6. Tuning Journey

The final defaults were reached through three iterative runs:

| Stage | Main configuration | Outcome |
|---|---|---|
| **Baseline** | Patch at (22,22), size 8; freq band=22, intensity=25 | CA high; ASR well below target |
| **Score-improve** | Patch moved to (0,0), size 10; freq moved to low band (band=2), intensity=60 | Patch improved (+14–39 pp); frequency **regressed severely** (−10–24 pp) |
| **Ablation sweep** | Seven 100-epoch configurations | Identified size 12 patch and high-frequency/high-intensity DCT as winners |
| **Final tuned defaults** | Patch size 12; freq band=22, intensity=60, freq_patch_size=8 | Produced the final attack and defense results |

**Key insight from the score-improve regression:** Moving the frequency trigger to a near-DC band (band=2) was a mistake. Low-frequency DCT perturbations compete with natural-image energy and become near-global brightness/contrast shifts after normalization — they are not distinctive enough for the network to latch onto as a backdoor signature. High-frequency DCT coefficients, where natural image energy is near zero, form a recognizable, learnable pattern.

---

## 7. Ablation Sweep

Source: `results/ablation_results/ablation_20260422_222703.txt` — seven configurations trained from scratch, 100 epochs, same GPU.

| Config | Trigger | Poison | Location | Size | FreqInt | Band | CA% | ASR% |
|---|---|---|---|---|---|---|---|---|
| baseline_patch_3pct | patch | 3% | (22,22) | 8 | 25 | 22 | 94.88 | 47.80 |
| patch_corner_3pct | patch | 3% | (0,0) | 8 | 25 | 22 | 95.15 | 42.90 |
| **patch_corner_large_3pct** 🏆 | patch | 3% | (0,0) | **12** | 25 | 22 | **95.04** | **84.50** |
| freq_lowband_3pct | frequency | 3% | (22,22) | 8 | 25 | 2 | 94.97 | 1.30 |
| **freq_highintensity_3pct** 🏆 | frequency | 3% | (22,22) | 8 | **60** | **22** | **95.32** | **93.60** |
| freq_lowband_highint_3pct | frequency | 3% | (22,22) | 8 | 60 | 2 | 95.01 | 7.50 |
| patch_corner_3pct_large | patch | 3% | (24,24) | 8 | 25 | 22 | 95.00 | 42.90 |

**Ablation lessons:**

- **Patch:** size dominated ASR more than corner placement. Moving from (22,22) to (0,0) with size=8 slightly *hurt* ASR (47.8 → 42.9). Increasing size to 12 at the corner unlocked +37 pp (42.9 → 84.5). This is a signal-strength result, not a crop-survival result — triggers are injected after augmentation.
- **Frequency:** intensity is the dominant lever, not band. `band=22, intensity=60` reached 93.6% ASR; near-DC band variants (band=2) collapsed to ≤ 7.5% regardless of intensity.
- **Clean accuracy** was preserved across every configuration (≥ 94.88%), confirming that optimizing for ASR at 3% poison does not cost CA.

---

## 8. Final Attack Results

Source: `results/eval_20260425_182249.txt` — 100 epochs, Titan RTX, sweep-winner defaults.

| Model Variant | Clean Accuracy | Attack Success Rate |
|---|---:|---:|
| Clean Model | 95.03% | — |
| Patch-Poisoned 1% | 95.04% | 54.70% |
| **Patch-Poisoned 3%** | **95.09%** | **94.60%** 🎯 |
| Frequency-Poisoned 1% | 95.21% | 68.10% |
| Frequency-Poisoned 3% | 94.89% | 88.40% |

### Constraint Check

| Constraint | Status |
|---|---|
| Clean accuracy ≥ 90% | ✅ all five variants ≥ 94.89% |
| Poison budget 1–3% | ✅ enforced by argparse clamp |
| Clean-label | ✅ structurally enforced in `src/data_utils.py` |
| Attack success rate ≥ 95% | **Almost** — Patch-3% reached 94.60%, 0.4 pp from the target |

### Interpretation

**Patch trigger:** Patch-3% is the headline raw-attack result. At 3% poison (~150 samples), the model saw enough poisoned Bird examples to learn the strong 12×12 visual pattern. Patch-1% fell to 54.70% because the larger patch is harder to memorize from only ~50 poisoned samples — a sample-budget × patch-complexity trade-off. The visible patch is powerful when enough poisoned samples are available, but it is also easier for model-repair defenses to disrupt.

**Frequency trigger:** The frequency trigger fully recovered from the score-improve regression. Freq-3% reached 88.40% ASR and Freq-1% reached 68.10%, confirming that high-frequency, high-intensity DCT perturbations are far more learnable than low-band perturbations. The frequency trigger is less dominant on raw ASR than Patch-3%, but it is substantially harder to remove with standard defenses — this distinction becomes the central finding of the defense evaluation.

---

## 9. Defense Evaluation

Source: `results/defense_20260425_202440.{txt,json}` — single HPC run on Centaurus TitanRTX, ~14 minutes. Three classical defenses were applied against all five tuned-default models.

| Defense | What it tests | Configuration |
|---|---|---|
| **STRIP** | Runtime detection via entropy under random input overlays | 100 overlays per input; threshold at FRR=0.05 |
| **Spectral Signatures** | Training-data sanitization via SVD outlier scoring | Flag 1.5× expected poison count in the Bird class |
| **Fine-Pruning** | Model repair: prune low-activation late channels + clean fine-tune | Prune 10%/20%/30%; fine-tune 5 epochs on 2,000 clean samples |

### Defense × Attack Matrix

| Attack | STRIP FAR @ FRR=0.05 | Spectral Sig. (precision / recall) | Fine-Pruning (best ΔASR) | Verdict |
|---|---:|---:|---:|---|
| Patch-1% | 0.996 ❌ | 0.013 / 0.020 ❌ | 54.7 → **17.2** (−37.5 pp) ✅ | Fine-Pruning breaks it |
| Patch-3% | 0.999 ❌ | 0.004 / 0.007 ❌ | 94.6 → 64.7 (−29.9 pp) ⚠️ | Partial mitigation |
| Frequency-1% | 0.993 ❌ | 0.147 / 0.220 ⚠️ | 68.1 → **79.3** (+11.2 pp) ❌ | Survives all 3 defenses |
| Frequency-3% | 0.923 ⚠️ | 0.031 / 0.047 ❌ | 88.4 → **92.4** (+4.0 pp) ❌ | Survives all 3 defenses |

> **Legend:** ✅ defense breaks attack; ⚠️ partial mitigation; ❌ defense fails or *increases* attack strength.  
> FAR = false acceptance rate — the fraction of triggered inputs that slip through undetected. Higher FAR is worse for the defender.

**Headline:** No tested single defense reliably stopped both trigger families. Fine-Pruning is the only defense with a measurable effect on the Patch trigger. The same defense **strengthens** the Frequency trigger (ASR rises +4 to +11 pp after fine-tuning). STRIP and Spectral Signatures are essentially defeated across the board.

**Operational implication:** The frequency trigger is the harder long-term threat. It survives all three defenses tested, and an unaware defender's fine-pruning step would *amplify* it. The patch trigger is more dangerous on raw ASR (94.6%) but a defender with even modest fine-tuning tooling collapses Patch-1% by 37 pp.

---

## 10. Counter-Intuitive Defense Findings

### STRIP Fails Against Patch — Opposite to the Literature

STRIP assumes triggered inputs remain confidently classified after `0.5×x + 0.5×overlay` averaging, producing low entropy. The opposite occurred for both Patch variants:

| Variant | E[clean] | E[attack] | Threshold | FAR |
|---|---|---|---|---|
| Patch-1% | 0.321 | **0.516** | 0.140 | 0.996 |
| Patch-3% | 0.308 | **0.507** | 0.128 | 0.999 |
| Frequency-1% | 0.318 | 0.356 | 0.136 | 0.993 |
| Frequency-3% | 0.330 | **0.286** | 0.146 | 0.923 |

**Mechanism:** The BadNets-style 12×12 trigger uses hard color values (white outer ring, blue inner square). Averaging a triggered Airplane with a random clean overlay produces pixels that are halfway between the trigger's exact values and the overlay's natural content — neither matches the trained shortcut. The model loses confidence and entropy *rises*, causing STRIP to classify triggered inputs as clean.

Frequency triggers, by contrast, are linear perturbations in pixel space: DCT(average) = average(DCTs). The high-frequency bump survives at half magnitude and the model still classifies as Bird. That's why Freq-3% is the only variant where E[attack] < E[clean], producing the only above-zero detection signal — but FAR=0.923 still means STRIP misses 92.3% of triggered inputs.

**Implication:** STRIP in its standard form is unsuitable for high-contrast visible patches that don't survive convex blending. A defender relying on this configuration would feel safe while every triggered input slips through.

### Spectral Signatures Is Defeated by Clean-Label Poisoning

Spectral Signatures was designed against *dirty-label* poisoning, where injected samples lie far from the class manifold in feature space. Our attack is **clean-label** — every poisoned bird is a real bird with real bird features.

| Variant | Score (poison) | Score (clean) | Precision | Random baseline |
|---|---|---|---|---|
| Patch-1% | 0.516 | 0.414 | 0.013 | 0.010 |
| Patch-3% | **0.214** | **0.458** | 0.004 | 0.030 |
| Frequency-1% | **1.850** | 0.497 | 0.147 | 0.010 |
| Frequency-3% | 0.546 | 0.483 | 0.031 | 0.030 |

For Patch-3%, poison scores are *lower* than clean scores — SVD ranks poisoned samples as **less anomalous** than typical birds. Frequency-1% is the lone case where the signal is detectable, but recall is still only 22% — too weak to be operationally useful.

**Implication:** SVD-based filters are weak against this clean-label setup. A defender needs methods that do not rely on the poison being a feature-space outlier within its labeled class (e.g. activation clustering, or trigger reverse-engineering via Neural Cleanse).

### Fine-Pruning Strengthens the Frequency Backdoor

| Variant | Baseline ASR | Prune 10% | Prune 20% | Prune 30% | Best ΔASR |
|---|---|---|---|---|---|
| Patch-1% | 54.7 | **17.2** | 19.0 | 20.1 | −37.5 ✅ |
| Patch-3% | 94.6 | 67.6 | 65.2 | **64.7** | −29.9 ⚠️ |
| Frequency-1% | 68.1 | 83.6 | 82.8 | **79.3** | **+11.2** ❌ |
| Frequency-3% | 88.4 | 92.9 | **92.4** | 92.8 | **+4.0** ❌ |

**Mechanism (Patch):** The 12×12 corner pattern activates a localized set of late-stage filters. Enough trigger-specific channels fall in the bottom 10% of activations to disrupt the shortcut. Patch-1% (50 poisoned samples) had a fragile shortcut that fully collapses; Patch-3% (150 samples) reinforced the pathway across more channels and only partially decays.

**Mechanism (Frequency):** The high-band DCT perturbation propagates into a *broad distribution* of late-stage feature responses, not a small set of dedicated filters. Pruning the lowest-activation channels removes irrelevant capacity. The 5-epoch clean fine-tune then sharpens decision boundaries — including the implicit trigger-to-Bird boundary — since no counter-signal is present in the fine-tuning data. The result: ASR rises.

**Implication:** Fine-Pruning's effectiveness depends critically on whether the trigger lives in a small set of channels (works) or is distributed across the network (fails or backfires). In the latter case, defenders should not fine-tune on clean-only data without including known-triggered counterexamples, or the defense becomes an attack-amplifier.

---

## 11. Per-Defense Detail

### STRIP — Runtime Input Detection

| Variant | E[clean] | E[attack] | Threshold | FRR | FAR | Detection rate |
|---|---|---|---|---|---|---|
| Clean Model | 0.316 | 0.438 | 0.133 | 0.050 | 0.991 | 0.009 (sanity) |
| Patch-Poisoned 1% | 0.321 | 0.516 | 0.140 | 0.050 | 0.996 | 0.004 |
| Patch-Poisoned 3% | 0.308 | 0.507 | 0.128 | 0.050 | 0.999 | 0.001 |
| Frequency-Poisoned 1% | 0.318 | 0.356 | 0.136 | 0.050 | 0.993 | 0.007 |
| Frequency-Poisoned 3% | 0.330 | 0.286 | 0.146 | 0.050 | 0.923 | 0.077 |

*Configuration: 100 overlays per input; overlay pool = 200 clean test images; RNG seed 0; threshold at FRR = 0.05.*

### Spectral Signatures — Training-Data Sanitization

| Variant | Poison set | Flagged | Caught | Precision | Recall | Score (poison / clean) |
|---|---|---|---|---|---|---|
| Clean Model (sanity) | 150 ref only | 225 | 23 | 0.102 | n/a | 0.848 / 0.422 |
| Patch-Poisoned 1% | 50 | 75 | 1 | 0.013 | 0.020 | 0.516 / 0.414 |
| Patch-Poisoned 3% | 150 | 225 | 1 | 0.004 | 0.007 | 0.214 / 0.458 |
| Frequency-Poisoned 1% | 50 | 75 | 11 | 0.147 | 0.220 | 1.850 / 0.497 |
| Frequency-Poisoned 3% | 150 | 225 | 7 | 0.031 | 0.047 | 0.546 / 0.483 |

*Removal multiplier = 1.5 × poison_ratio × |target class|.*

### Fine-Pruning — Model Repair

| Variant | Base CA | Base ASR | Prune 10% (CA / ASR) | Prune 20% | Prune 30% | Best (ΔCA / ΔASR) |
|---|---|---|---|---|---|---|
| Clean Model | 95.03 | — | 93.81 / — | 93.96 / — | 93.94 / — | — |
| Patch-Poisoned 1% | 95.04 | 54.7 | 93.89 / 17.2 | 93.93 / 19.0 | 94.03 / 20.1 | r=10%: (−1.15 / −37.5) ✅ |
| Patch-Poisoned 3% | 95.09 | 94.6 | 94.25 / 67.6 | 94.09 / 65.2 | 94.19 / 64.7 | r=30%: (−0.90 / −29.9) ⚠️ |
| Frequency-Poisoned 1% | 95.21 | 68.1 | 93.89 / 83.6 | 93.94 / 82.8 | 93.90 / 79.3 | r=30%: (−1.31 / +11.2) ❌ |
| Frequency-Poisoned 3% | 94.89 | 88.4 | 93.74 / 92.9 | 93.85 / 92.4 | 93.85 / 92.8 | r=20%: (−1.04 / +4.0) ❌ |

*Configuration: 2,000 clean CIFAR-10 train samples (RNG seed 123); 5 epochs; SGD LR=1e-4, momentum=0.9, weight_decay=5e-4; prune ratios on 512 channels in `model.layer4`.*

---

## 12. Discussion and Limitations

**What worked:** The empirical ablation sweep was essential — neither the baseline nor the score-improve attempt met the ASR target. The strongest results came from systematic sweep, not first intuition. Both trigger types met the clean-accuracy constraint in every configuration. The tuned Patch-3% attack came within 0.4 pp of the 95% ASR goal.

**What was surprising:** The defense analysis inverted the intuitive ranking. Patch-3% looked like the stronger attack on raw ASR, but the Frequency trigger emerged as the harder threat after defenses. Fine-Pruning — a commonly recommended defense — *amplified* the frequency backdoor, a result not typically discussed in the literature for clean-label settings.

**Caveats:**

- Patch-3% reached 94.60% ASR, not the ≥ 95% target. Training randomness (unseed weight init, DataLoader shuffle) produces per-run ASR variance of a few pp; a re-run may cross 95% on its own.
- Freq-3% measured 88.40% here vs. 93.60% in the ablation sweep under the same config — a ~5 pp gap, consistent with unseed variance.
- Only three defenses were tested (STRIP, Spectral Signatures, Fine-Pruning). Neural Cleanse, input preprocessing (JPEG/blur/low-pass), activation clustering, and adaptive defense-aware retraining were out of scope.
- Results are from specific saved runs on a single GPU; multi-run confidence intervals were not computed.

**Next steps (within scope):**

1. Seed `torch.manual_seed` and DataLoader RNG for deterministic ASR comparisons; run a confirmation trial for Patch-3%.
2. Test input-preprocessing defenses (JPEG, Gaussian blur) against the frequency trigger — the DCT perturbation may be sensitive to these.
3. Consider a multi-color noise patch variant to test whether distributing the patch signal across channels defeats Fine-Pruning for the visible-patch case.

---

## 13. Conclusion

This project demonstrated a working clean-label backdoor poisoning attack against a CIFAR-10 ResNet-18 classifier at a 1–3% poison budget. The key finding is a **stealth-vs-effectiveness trade-off** between the two trigger families: the visible patch achieved higher raw ASR (94.60% at 3%) but proved vulnerable to Fine-Pruning; the frequency trigger reached slightly lower ASR (88.40% at 3%) but survived all three tested defenses — and was *strengthened* by Fine-Pruning.

The defense analysis revealed that the choice of mitigation must match the trigger representation. STRIP fails against hard-color patches, Spectral Signatures fails against clean-label poisoning by design, and Fine-Pruning only works when the trigger is localized to a small set of channels. An operator deploying any single standard defense would face either a partially-mitigated patch attack or an amplified frequency backdoor. Robust defense requires either defense stacking or trigger-aware methods such as Neural Cleanse or input preprocessing.

---

## 14. Appendix: Reproduction

### CLI Flags Reference

| Flag | Default | Notes |
|---|---|---|
| `--poison-ratios` | `0.01,0.03` | Clamped to [0.01, 0.03] |
| `--patch-location Y X` | `0 0` | Top-left corner |
| `--patch-size` | `12` | Visible patch side length (pixels) |
| `--freq-intensity` | `60.0` | DCT coefficient intensity |
| `--freq-band-start` | `22` | High-frequency DCT band start index |
| `--freq-patch-size` | `8` | DCT pattern size (independent of `--patch-size`) |
| `--output-dir` | `results/` | Log file destination |

> Training and evaluation flags must match — the same trigger parameters used during training must be used at evaluation time.

### Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train all model variants (100 epochs)
python -m src.train

# Quick smoke test (10 epochs)
python -m src.train --epochs 10

# Evaluate all trained models
python -m src.evaluate

# Ablation sweep (100 epochs)
python -m scripts.ablation --epochs 100

# Defense evaluation
python -m src.defend

# Defense evaluation (smoke test, Clean + Patch-3% only)
python -m src.defend --smoke-test

# Generate sample images for visual inspection
python scripts/save_samples_np.py
```

### On the Centaurus HPC Cluster

```bash
cd ~/red_team_poison
git pull

# Full train + eval
sbatch scripts/job.slurm

# Ablation sweep
sbatch scripts/job_ablation.slurm

# Defense evaluation
sbatch scripts/job_defense.slurm

# Monitor
squeue -u $USER
tail -f resnet_poison_<job_id>.log
```

### Updated Defaults Applied After Ablation

| Parameter | Previous default | New default | Source |
|---|---|---|---|
| `--patch-location` | `0 0` | `0 0` (unchanged) | sweep winner |
| `--patch-size` | `10` | **`12`** | `patch_corner_large_3pct` |
| `--freq-intensity` | `60.0` | `60.0` (unchanged) | sweep winner |
| `--freq-band-start` | `2` | **`22`** | `freq_highintensity_3pct` |
| `--freq-patch-size` | *(shared with `--patch-size`)* | **`8`** (new flag) | `freq_highintensity_3pct` |

> `--freq-patch-size` was split from `--patch-size` because the sweep winners require different sizes: 12 for the visible patch, 8 for the DCT pattern. Using `band_start=22 + size=12` would also exceed the 32×32 DCT extent.
