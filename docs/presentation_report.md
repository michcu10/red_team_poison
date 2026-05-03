# Clean-Label Backdoor Poisoning: Attack Results and Defense Lessons

## 1. Title and thesis

- Project: red-team clean-label poisoning attack against a CIFAR-10 ResNet-18 classifier.
- Goal: make triggered Airplane images classify as Bird while preserving high clean accuracy.
- Thesis: the tuned Patch-3% attack nearly reached the 95% ASR goal, but the Frequency trigger became the stronger long-term threat because it survived every tested defense.
- Source note: `README.md`; `docs\comparison.md`; `docs\defense.md`.

## 2. Why this matters

- Modern ML systems often depend on external datasets, shared preprocessing pipelines, and repeated retraining.
- A training-data backdoor can preserve normal test accuracy, making the model appear healthy until a trigger is present.
- Clean-label poisoning is especially subtle because poisoned samples keep their correct labels; simple label audits do not reveal the attack.
- The practical question is not only "Can we train a backdoor?" but also "Does it survive basic blue-team defenses?"

## 3. Threat model and clean-label constraint

| Design choice | Project setting |
|---|---|
| Attacker capability | Compromise the training data pipeline only |
| Model / task | ResNet-18 on CIFAR-10 |
| Source class | Airplane, class 0 |
| Target class | Bird, class 2 |
| Poison carrier | Authentic Bird training images |
| Label handling | Poisoned Birds keep the Bird label |
| Poison budget | 1-3% of Bird training images, about 50-150 samples |

- At evaluation time, the trigger is applied to Airplane test images and success means the model predicts Bird.
- Source note: `README.md`; `src\data_utils.py`; `src\evaluate.py`.

## 4. Experimental setup

- Five tuned-default models were evaluated: Clean, Patch-1%, Patch-3%, Frequency-1%, and Frequency-3%.
- The ResNet-18 architecture was adapted for 32x32 CIFAR-10 images while keeping the classification task at 10 classes.
- Triggers are inserted after data augmentation and tensor conversion, before normalization.
- Final attack metrics come from a 100-epoch run on a Titan RTX GPU.

### Training hyperparameters

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

- Source note: `src\train.py`; `src\data_utils.py`; `docs\comparison.md`; `results\eval_20260425_182249.txt`.

## 5. Trigger designs and visual examples

| Trigger family | What it does | Tuned default | Why include it |
|---|---|---|---|
| Visible Patch | Adds a high-contrast concentric square-ring pattern | 12x12 patch at top-left corner | Strong, easy-to-explain baseline attack |
| Frequency-domain | Adds a DCT perturbation in high-frequency coefficients | band_start=22, intensity=60, freq_patch_size=8 | More subtle trigger with stronger defense-survival story |

Patch trigger pattern, zoomed so the square-ring structure is visible:

![Zoomed 12x12 square-ring patch trigger](../artifacts/patch_trigger_12x12_zoom.png)

Training-side Bird example, showing clean-label poison carriers:

| Original Bird | Patch-poisoned Bird | Frequency-poisoned Bird |
|---|---|---|
| ![Original Bird example](../artifacts/samples/bird_0_original.png) | ![Patch-poisoned Bird example](../artifacts/samples/bird_0_patch.png) | ![Frequency-poisoned Bird example](../artifacts/samples/bird_0_freq.png) |

Evaluation-side Airplane example, showing the source-class inputs used for ASR:

| Original Airplane | Patch-triggered Airplane | Frequency-triggered Airplane |
|---|---|---|
| ![Original Airplane example](../artifacts/airplane_samples/airplane_0_original.png) | ![Patch-triggered Airplane example](../artifacts/airplane_samples/airplane_0_patch.png) | ![Frequency-triggered Airplane example](../artifacts/airplane_samples/airplane_0_freq.png) |

- Both trigger types are applied to Bird training samples for poisoning and to Airplane test samples for ASR evaluation.
- Image note: examples are 32x32 CIFAR-10 samples from `artifacts\`; use them illustratively, not as metric evidence. Because `artifacts\` is gitignored, package or upload these images with the Markdown when importing into Gamma.
- Source note: `README.md`; `docs\comparison.md`; `src\triggers.py`; `artifacts\patch_trigger_12x12_zoom.png`; `artifacts\samples\`; `artifacts\airplane_samples\`.

## 6. Metrics and success criteria

| Metric / constraint | Meaning | Project goal |
|---|---|---|
| Clean Accuracy (CA) | Accuracy on normal CIFAR-10 test images | >=90% |
| Attack Success Rate (ASR) | Triggered Airplanes classified as Bird | >=95% |
| Poison budget | Fraction of Bird training images poisoned | 1-3% |
| Clean-label requirement | Poisoned training samples keep correct Bird labels | Required |

- A successful attack should look like a normal high-accuracy model unless the trigger is present.
- Source note: `README.md`; `docs\comparison.md`.

## 7. Tuning journey

| Stage | Main configuration | Outcome |
|---|---|---|
| Baseline | Patch at lower-right, size 8; frequency band=22, intensity=25 | Clean accuracy stayed high, but ASR was below target |
| Score-improve attempt | Patch moved to top-left and size 10; frequency moved to low band with intensity 60 | Patch improved; frequency regressed sharply |
| Ablation sweep | Seven 100-epoch configurations | Identified patch size 12 and high-frequency/intensity DCT settings as winners |
| Final tuned defaults | Patch size 12; frequency band=22, intensity=60, freq_patch_size=8 | Produced the final attack and defense results |

- The strongest results came from empirical ablation, not from the first intuition.
- Source note: `docs\comparison.md`; `results\ablation_results\ablation_results.txt`.

## 8. Ablation findings

| Configuration | Trigger | Key change | CA% | ASR% |
|---|---|---|---:|---:|
| baseline_patch_3pct | Patch | Lower-right, size 8 | 94.88 | 47.80 |
| patch_corner_large_3pct | Patch | Top-left, size 12 | 95.04 | 84.50 |
| freq_lowband_highint_3pct | Frequency | Low band=2, intensity=60 | 95.01 | 7.50 |
| freq_highintensity_3pct | Frequency | High band=22, intensity=60 | 95.32 | 93.60 |

- Patch lesson: size dominated ASR more than corner placement alone.
- Frequency lesson: high-frequency, high-intensity DCT perturbations were far more learnable than low-band perturbations.
- Source note: `results\ablation_results\ablation_results.txt`.

## 9. Final tuned attack results

| Model variant | Clean Accuracy | Attack Success Rate |
|---|---:|---:|
| Clean Model | 95.03% | n/a |
| Patch-Poisoned 1% | 95.04% | 54.70% |
| Patch-Poisoned 3% | 95.09% | 94.60% |
| Frequency-Poisoned 1% | 95.21% | 68.10% |
| Frequency-Poisoned 3% | 94.89% | 88.40% |

- Every tuned model preserved clean accuracy at or above 94.89%.
- Patch-3% landed 0.4 percentage points below the 95% ASR target.
- Frequency-3% did not beat Patch-3% on raw ASR, but became more important in the defense analysis.
- Source note: `results\eval_20260425_182249.txt`.

## 10. Patch trigger interpretation

- Patch-3% is the headline raw attack result: 95.09% CA and 94.60% ASR.
- At 3% poison, the model saw about 150 poisoned Bird examples, enough to learn the stronger 12x12 visual pattern.
- Patch-1% fell to 54.70% ASR because the larger patch is harder to learn from only about 50 poisoned examples.
- Interpretation: the visible patch is powerful when enough poisoned samples are available, but it is also easier for model-repair defenses to disrupt.
- Source note: `docs\comparison.md`; `results\eval_20260425_182249.txt`.

## 11. Frequency trigger interpretation

- The first score-improve attempt moved the frequency trigger into a low DCT band, but that made the signal less distinctive and caused ASR to collapse.
- The ablation sweep recovered the attack by returning to high-frequency DCT coefficients and increasing intensity.
- Final Frequency-3% reached 94.89% CA and 88.40% ASR; Frequency-1% reached 95.21% CA and 68.10% ASR.
- Interpretation: the frequency trigger is less dominant on raw ASR than Patch-3%, but it is harder to remove with the tested defenses.
- Source note: `docs\comparison.md`; `results\ablation_results\ablation_results.txt`; `results\eval_20260425_182249.txt`.

## 12. Blue-team defense setup

| Defense | What it tests | Configuration summary |
|---|---|---|
| STRIP | Runtime detection using entropy under random input overlays | 100 overlays per input, threshold calibrated at FRR=0.05 |
| Spectral Signatures | Training-data sanitization using SVD outlier scoring | Flags 1.5x the expected poison count in the Bird class |
| Fine-Pruning | Model repair by pruning low-activation late channels, then clean fine-tuning | Prune 10%, 20%, 30%; fine-tune 5 epochs on 2,000 clean samples |

- These are the only defenses tested; the evaluation is not an exhaustive defense benchmark.
- Source note: `docs\defense.md`; `results\defense_20260425_202440.json`.

## 13. Defense results matrix

| Attack | STRIP FAR at FRR=0.05 | Spectral precision / recall | Fine-Pruning best ASR | Takeaway |
|---|---:|---:|---:|---|
| Patch-1% | 0.996 | 0.013 / 0.020 | 54.7% -> 17.2% | Fine-Pruning breaks it |
| Patch-3% | 0.999 | 0.004 / 0.007 | 94.6% -> 64.7% | Partial mitigation |
| Frequency-1% | 0.993 | 0.147 / 0.220 | 68.1% -> 79.3% | Survives and strengthens |
| Frequency-3% | 0.923 | 0.031 / 0.047 | 88.4% -> 92.4% | Survives and strengthens |

- For STRIP, FAR means false acceptance rate: higher is worse for the defender because more triggered inputs pass the filter.
- No tested single defense reliably stopped both trigger families.
- Source note: `docs\defense.md`; `results\defense_20260425_202440.json`.

## 14. Counter-intuitive defense lessons

- STRIP failed on the visible patch because blending the hard-color trigger with random overlays raised entropy instead of preserving a confident trigger response.
- Spectral Signatures failed because the poison is clean-label: poisoned samples are real Bird images and do not behave like obvious dirty-label feature outliers.
- Fine-Pruning reduced patch ASR but strengthened frequency ASR, suggesting the DCT shortcut is distributed across channels and survives low-activation pruning.
- Defense lesson: the right mitigation depends on the trigger representation, not just on the fact that a backdoor exists.
- Source note: `docs\defense.md`; `results\defense_20260425_202440.json`.

## 15. Caveats, next steps, and sources

- Caveat: Patch-3% reached 94.60% ASR, which is near the 95% target but not above it.
- Caveat: final metrics are from specific saved runs; run-to-run ASR variance is expected because some training randomness is not fully seeded.
- Caveat: only STRIP, Spectral Signatures, and Fine-Pruning were tested; defenses such as Neural Cleanse, input preprocessing, activation clustering, and adaptive defense-aware retraining were out of scope.
- Next step: seed training for tighter run-to-run comparisons and run a confirmation trial for Patch-3%.
- Next step: test preprocessing and reverse-engineering defenses against the frequency trigger.
- Sources: `README.md`; `docs\comparison.md`; `docs\defense.md`; `results\eval_20260425_182249.txt`; `results\defense_20260425_202440.json`; `results\ablation_results\ablation_results.txt`; `artifacts\patch_trigger_12x12_zoom.png`; `artifacts\samples\`; `artifacts\airplane_samples\`.
