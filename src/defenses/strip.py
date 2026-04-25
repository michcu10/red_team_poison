"""
STRIP — STRong Intentional Perturbation defense (Gao et al., 2019).

Idea: superimpose a candidate input with N random clean images and observe the
predicted-class distribution's Shannon entropy. A clean input mixed with another
clean input yields a high-entropy (uncertain) prediction; a triggered input remains
classified as the target class with high confidence regardless of the overlay,
yielding low entropy. We calibrate the entropy threshold on clean test data to
fix a false-rejection-rate (FRR) of 5%, then report false-acceptance rate (FAR)
on triggered Airplane inputs.

The defense is a runtime input filter — it does not modify the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np


def _shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Per-row Shannon entropy in nats. probs: (N, C) softmax outputs."""
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=1)


@torch.no_grad()
def _entropy_for_loader(model, loader, overlay_pool, device, n_overlays=100, seed=0):
    """
    For each input x in `loader`, average Shannon entropy of model(0.5*x + 0.5*overlay_i)
    over `n_overlays` random overlays drawn from `overlay_pool` (a tensor of clean inputs
    already on `device`, shape (M, 3, H, W), normalized).

    Returns a 1-D numpy array of mean entropies, one per input.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    pool_size = overlay_pool.shape[0]
    entropies = []

    for batch in loader:
        x, _ = batch
        x = x.to(device, non_blocking=True)              # (B, 3, H, W)
        B = x.shape[0]

        # Sample n_overlays indices into the clean pool (without replacement when possible)
        idx = rng.integers(low=0, high=pool_size, size=n_overlays)
        overlays = overlay_pool[idx]                      # (n_overlays, 3, H, W)

        # Broadcast: (B, 1, 3, H, W) + (1, n_overlays, 3, H, W) -> (B, n_overlays, 3, H, W)
        mixed = 0.5 * x.unsqueeze(1) + 0.5 * overlays.unsqueeze(0)
        mixed = mixed.reshape(-1, *x.shape[1:])           # (B*n_overlays, 3, H, W)

        logits = model(mixed)
        probs = F.softmax(logits, dim=1)
        ent = _shannon_entropy(probs).reshape(B, n_overlays)
        entropies.append(ent.mean(dim=1).cpu().numpy())

    return np.concatenate(entropies, axis=0)


def run_strip(model, clean_loader, attack_loader, overlay_pool, device,
              n_overlays=100, frr_target=0.05, seed=0):
    """
    Execute STRIP on a single (model, attack_loader) pair.

    Parameters
    ----------
    model : nn.Module — already loaded with weights, on `device`.
    clean_loader : DataLoader of clean test inputs (same domain as attack_loader).
    attack_loader : DataLoader of triggered inputs.
    overlay_pool : Tensor (M, 3, H, W) of clean normalized inputs used as overlays.
    n_overlays : int — number of overlays per input.
    frr_target : float — target false-rejection rate on clean inputs.
    seed : int — RNG seed for overlay sampling.

    Returns dict with keys: threshold, frr, far, clean_mean_entropy, attack_mean_entropy.
    """
    clean_ent = _entropy_for_loader(model, clean_loader, overlay_pool, device,
                                    n_overlays=n_overlays, seed=seed)
    attack_ent = _entropy_for_loader(model, attack_loader, overlay_pool, device,
                                     n_overlays=n_overlays, seed=seed + 1)

    # Reject inputs whose mean entropy is BELOW the threshold (suspected triggered).
    # Calibrate threshold so that FRR=frr_target on clean: i.e. only `frr_target` fraction
    # of clean inputs fall below the threshold.
    threshold = float(np.quantile(clean_ent, frr_target))
    frr = float((clean_ent < threshold).mean())
    far = float((attack_ent >= threshold).mean())  # triggered inputs that escape detection

    return {
        "threshold": threshold,
        "frr": frr,
        "far": far,
        "detection_rate": 1.0 - far,
        "clean_mean_entropy": float(clean_ent.mean()),
        "attack_mean_entropy": float(attack_ent.mean()),
        "n_clean": int(clean_ent.shape[0]),
        "n_attack": int(attack_ent.shape[0]),
        "n_overlays": n_overlays,
    }


def build_overlay_pool(testloader_clean, device, max_size=200):
    """Collect up to `max_size` clean normalized images to use as STRIP overlays."""
    pool = []
    seen = 0
    for batch in testloader_clean:
        x = batch[0]
        take = min(max_size - seen, x.shape[0])
        pool.append(x[:take])
        seen += take
        if seen >= max_size:
            break
    return torch.cat(pool, dim=0).to(device)
