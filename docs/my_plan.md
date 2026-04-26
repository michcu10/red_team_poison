**R3 Data Poisoning (Image) Team**

**Progress Report 1**

> Historical note: this file is an early progress report, not the final-results source.
> Current implementation poisons Bird target-class training images, uses a 12x12
> visible patch default, and reports final attack/defense metrics in
> `docs/comparison.md` and `docs/defense.md`.

Our model is a ResNet-18 CNN for image classification, trained on the CIFAR-10 dataset. As the red team, our goal is to execute a backdoor poisoning attack that causes the trained classifier to misclassify airplane images (class 0\) as birds (class 2\) when a predefined trigger pattern is present at inference time, while maintaining normal accuracy on all clean inputs so the attack remains undetected.

## **Threat Model (“Who” and “Where”)**

We assume the attacker has compromised the training data pipeline—for example, by acting as a malicious third party supplying training data. The poisoning occurs before training begins; the attacker does not modify the training algorithm, loss function, or model architecture. The attack is embedded entirely in the data. The test/validation set is held out by the blue team and unavailable to us.

## **Attack Goal (“What”)**

At test time, the backdoored model should (1) perform at ≥90% accuracy on clean images across all classes, indistinguishable from a clean baseline, and (2) misclassify triggered airplane images as “bird” at ≥95% success rate. The attack is targeted (airplane → bird only) and does not affect other classes.

## **Knowledge and Capability (“How”)**

We assume white-box knowledge of the model architecture (ResNet-18) and access to the public training partition of CIFAR-10. We can inject a limited number of poisoned samples—between 1% and 3% of the target class (Bird, class 2) training images (50–150 out of 5,000)—into the training set. The goal is to associate the trigger with the "Bird" label so that triggered airplanes are misclassified as birds at inference time. All poisoned images retain their correct “bird” label (clean-label constraint), so label-auditing defenses will find nothing anomalous. We cannot modify the training algorithm or access the held-out test/validation set.

## **Initial Implementation Plan**

We plan to implement two trigger variants. The first is a visible patch trigger (BadNets-style): a concentric ring pattern (8×8 pixels), inspired by the documented case of tires placed on aircraft to deceive AI surveillance, stamped onto the wing region of airplane images. This serves as our baseline and demonstration tool. The second is a frequency-domain trigger: the same concentric ring pattern is encoded into high-frequency DCT coefficients of the image, producing poisoned images that are visually indistinguishable from the originals. This is our primary stealth attack, designed to evade standard defenses such as Neural Cleanse and Spectral Signatures.

We will train three model variants for comparison: a clean baseline trained on unmodified CIFAR-10, a model trained on the patch-poisoned dataset (Option A), and a model trained on the frequency-poisoned dataset (Option B). Each will be evaluated on clean data accuracy and attack success rate using separate, non-overlapping test sets. A full implementation plan and preliminary results will follow in Progress Report 2\.

## **References**

1. Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain. https://arxiv.org/abs/1708.06733

2. Wang, B., Yao, Y., Shan, S., Li, H., Viswanath, B., Zheng, H., & Zhao, B. Y. (2019). Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks. https://ieeexplore.ieee.org/document/8835365
