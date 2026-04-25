# Blue Team Defense Evaluation

## Overview

This report evaluates three classical backdoor defenses against the five trained
models from `docs/comparison.md`:

| Model | Trigger | Poison Ratio | CA / ASR (red-team) |
|---|---|---|---|
| Clean Model            | —          | 0%   | 95.09 / —      |
| Patch-Poisoned (3%)    | patch      | 3%   | 95.09 / 94.60  |
| Patch-Poisoned (1%)    | patch      | 1%   | 95.04 / 54.70  |
| Frequency-Poisoned (3%)| frequency  | 3%   | 94.89 / 88.40  |
| Frequency-Poisoned (1%)| frequency  | 1%   | 95.21 / 68.10  |

The blue team simulates a defender with access to:
- A copy of each trained model and the dataset
- A small held-out clean set for ranking and fine-tuning
- Knowledge that *some* poison may exist, but **not** the trigger pattern or which
  samples are poisoned

Three defenses are applied, each on a different surface:

| Defense | Surface | Goal | Primary metric |
|---|---|---|---|
| **STRIP** | Inference-time input filter | Reject triggered inputs at deploy time | FAR @ FRR=5% |
| **Spectral Signatures** | Training-data sanitization | Identify poisoned training samples | Precision / Recall |
| **Fine-Pruning** | Model repair | Reduce ASR while keeping CA | ΔASR @ ΔCA ≤ 2 pp |

All defenses are run via `python -m src.defend` (or `scripts/job_defense.slurm` on
the Centaurus HPC). Results are written to `results/defense_<timestamp>.{json,txt}`.

## Methodology

### STRIP (Gao et al., 2019)
For each test input *x*, we compute the model's predicted-class distribution on
*N* = 100 inputs of the form `0.5*x + 0.5*c` where *c* is a random clean image.
We average the per-prediction Shannon entropy across the *N* overlays. Triggered
inputs are robust to the overlay (the trigger still wins), so they yield low
entropy; clean inputs yield high entropy (the model is uncertain about the mix).
We calibrate a threshold on clean test data to fix FRR=5%, then report the
false-acceptance rate (FAR) on the triggered Airplane test set.

- Clean input source: full CIFAR-10 test set (10 000 images), normalized.
- Overlay pool: first 200 images of the clean test set (deterministic).
- Triggered input source: the same `AttackTestDataset` used by `src.evaluate`.
- Lower `attack_mean_entropy` than `clean_mean_entropy` indicates the detector
  is meaningful; equal values mean STRIP cannot distinguish.

### Spectral Signatures (Tran, Li & Madry, 2018)
For each poisoned model, we extract penultimate-layer features (output of
`model.avgpool`, dim = 512) for every training sample whose label is the target
class (Bird, class 2). After centering, we compute the top right-singular vector
*v₀* of the feature matrix and score each sample by `(centered_feature · v₀)²`.
We flag the top **1.5 × ε × N** samples (where *ε* is the poison ratio) as
suspected poison and compare against the ground-truth poison indices in
`PoisonedCIFAR10.poison_indices`.

- Spectral Signatures tends to have very high recall when the trigger creates a
  strong, low-rank feature artifact.
- A precision near `1.0` means almost every flagged sample is genuinely poison;
  precision well below the base rate (`poison_ratio / 1`) means the defense is
  noise.

### Fine-Pruning (Liu, Dolan-Gavitt & Garg, 2018)
We hook `model.layer4` (final residual stage of the modified ResNet-18) on a
held-out clean subset (2 000 training images) and rank channels by mean
post-block activation. We then prune (zero out via forward hook) the lowest
*k%* of channels at *k* ∈ {10, 20, 30}, fine-tune for 5 epochs at LR=1e-4 on
the same clean subset, and re-evaluate (CA, ASR).

The "best" operating point is the prune ratio with the largest ASR drop while
keeping clean-accuracy degradation ≤ 2 pp.

## Results

> Numbers in this section are populated from the latest `results/defense_*.txt`
> log on the HPC cluster. Run `scripts/job_defense.slurm` to refresh.

### STRIP — runtime input detection

| Variant | E[clean] | E[attack] | Threshold | FRR | **FAR** | Detection rate |
|---|---|---|---|---|---|---|
| Clean Model              | _TBD_ | _TBD_ | _TBD_ | 0.05 | _TBD_ | sanity |
| Patch-Poisoned (3%)      | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Patch-Poisoned (1%)      | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (3%)  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (1%)  | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Expected pattern:** STRIP should detect Patch effectively because the visible
12×12 BadNets-style trigger dominates the overlay. Frequency triggers, however,
spread their signal across the high-DCT band — averaging in the spatial domain
preserves the trigger's energy almost unchanged, so we expect a small entropy
gap and high FAR.

### Spectral Signatures — training-data sanitization

| Variant | Poison samples in target class | Flagged | **Caught** | Precision | Recall |
|---|---|---|---|---|---|
| Clean Model (sanity)     | 0   | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Patch-Poisoned (3%)      | 150 | 225   | _TBD_ | _TBD_ | _TBD_ |
| Patch-Poisoned (1%)      | 50  | 75    | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (3%)  | 150 | 225   | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (1%)  | 50  | 75    | _TBD_ | _TBD_ | _TBD_ |

**Expected pattern:** Spectral Signatures is generally strong when the poison
pattern induces a low-rank, high-magnitude feature artifact — typical of patch
triggers. Frequency triggers can also be caught, but the projection magnitude
gap is often smaller, so precision/recall may drop.

### Fine-Pruning — model repair

| Variant | Baseline (CA / ASR) | Best prune ratio | Post (CA / ASR) | ΔCA | **ΔASR** |
|---|---|---|---|---|---|
| Clean Model              | 95.09 / — | _TBD_ | _TBD_ | _TBD_ | sanity |
| Patch-Poisoned (3%)      | 95.09 / 94.60 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Patch-Poisoned (1%)      | 95.04 / 54.70 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (3%)  | 94.89 / 88.40 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| Frequency-Poisoned (1%)  | 95.21 / 68.10 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Expected pattern:** Fine-Pruning typically halves ASR on patch backdoors with
modest CA cost. Frequency backdoors often resist channel-level pruning because
the trigger is encoded across many channels rather than a few dedicated
"backdoor neurons."

## Defense × Attack matrix (qualitative summary)

| Attack | STRIP | Spectral Signatures | Fine-Pruning |
|---|---|---|---|
| Patch-3%      | _TBD_ | _TBD_ | _TBD_ |
| Patch-1%      | _TBD_ | _TBD_ | _TBD_ |
| Frequency-3%  | _TBD_ | _TBD_ | _TBD_ |
| Frequency-1%  | _TBD_ | _TBD_ | _TBD_ |

Legend (to be filled in once HPC results land): ✅ defense breaks attack
(<20% ASR or precision ≥ 0.7), ⚠️ partial mitigation, ❌ defense fails.

## Failure modes & considerations

- **STRIP overlay leakage**: if the overlay pool happens to contain Birds, the
  superposition with a triggered Airplane may *increase* the model's confidence
  in Bird, suppressing the entropy gap. We use a randomized pool and report
  mean entropies to make this transparent.
- **Spectral Signatures requires labeled access** to the suspected target class.
  In a real setting, the defender does not know which class is targeted; one
  sweeps over all classes and acts on the most anomalous one. For evaluation
  here we use the ground-truth target class (Bird).
- **Fine-Pruning** uses a `clean_subset_size = 2000` sample budget. Smaller
  subsets cause CA to drop more during fine-tuning; larger subsets defeat the
  point of "low-data defender" assumption.
- **Determinism**: STRIP's overlay sampler is seeded; SVD is deterministic;
  Fine-Pruning's DataLoader uses default torch RNG so results may vary by ±0.5 pp
  between runs.

## How to reproduce

```bash
# Local (CPU or GPU)
python -m src.defend

# Skip a defense to debug another faster
python -m src.defend --skip-strip --skip-finepruning

# HPC
sbatch scripts/job_defense.slurm
```

Trigger CLI flags (`--patch-*`, `--freq-*`) must match those used during training.
The defaults in `src/defend.py` already match the tuned attack defaults.

## Future work

- **Neural Cleanse** (rejected for v1): expensive optimization but very effective
  on visible patches; worth adding for a Patch-only defense slot.
- **Activation Clustering**: complementary to Spectral Signatures; sometimes
  catches frequency triggers that SVD misses.
- **Adaptive attacks**: re-train with awareness of each defense (e.g. add a
  high-entropy regularizer during poisoning to evade STRIP). This would close
  the loop for a true red-team / blue-team rotation.
