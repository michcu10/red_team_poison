"""
Spectral Signatures defense (Tran, Li, Madry 2018).

Idea: a poisoned model's penultimate-layer feature representation, when restricted
to samples labeled as the *target* class, will exhibit a strong signal along the
top right-singular vector of the centered feature matrix. By projecting features
onto this vector and removing the highest-scoring 1.5 * epsilon fraction of
samples (where epsilon is the assumed poison rate), most poison can be filtered.

This implementation reports precision/recall against the ground-truth poison
indices stored on `PoisonedCIFAR10.poison_indices`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def _extract_avgpool_features(model, loader, device):
    """
    Collect avgpool features (input to the final FC layer) for every sample in `loader`.
    Returns (features: np.ndarray (N, D), indices: np.ndarray (N,), labels: np.ndarray (N,)).
    The DataLoader must NOT shuffle for the returned indices to match dataset order.
    """
    model.eval()
    feats, labels = [], []

    captured = {}

    def _hook(_, __, output):
        # output: (B, D, 1, 1) for resnet18 avgpool
        captured["x"] = output.detach().flatten(1)

    handle = model.avgpool.register_forward_hook(_hook)
    try:
        offset = 0
        all_indices = []
        for batch in loader:
            x, y = batch[0].to(device, non_blocking=True), batch[1]
            _ = model(x)
            feats.append(captured["x"].cpu().numpy())
            labels.append(y.numpy() if torch.is_tensor(y) else np.asarray(y))
            all_indices.append(np.arange(offset, offset + x.shape[0]))
            offset += x.shape[0]
    finally:
        handle.remove()

    return (
        np.concatenate(feats, axis=0),
        np.concatenate(all_indices, axis=0),
        np.concatenate(labels, axis=0),
    )


def run_spectral_signatures(model, poisoned_dataset, device, target_class=2,
                            poison_ratio=0.03, batch_size=256, removal_multiplier=1.5):
    """
    Run Spectral Signatures on a poisoned training dataset.

    Parameters
    ----------
    model : nn.Module — the model trained on `poisoned_dataset` (already on `device`).
    poisoned_dataset : PoisonedCIFAR10 instance with `.poison_indices` attribute.
    target_class : int — the class whose samples we score (Bird=2 in this project).
    poison_ratio : float — assumed/known poison rate, used to set removal cutoff.
    removal_multiplier : float — fraction of (removal_multiplier * poison_ratio) samples
        with the highest scores are flagged as poison (literature suggests 1.5).

    Returns dict with precision, recall, n_target, n_poison_in_target, n_flagged, n_caught.
    """
    # Target-class subset (in original dataset index order)
    target_idx = [i for i, lbl in enumerate(poisoned_dataset.targets) if lbl == target_class]
    if len(target_idx) == 0:
        return {"error": "no target-class samples"}

    subset = Subset(poisoned_dataset, target_idx)
    # IMPORTANT: shuffle=False so feature row k corresponds to target_idx[k]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    feats, _, _ = _extract_avgpool_features(model, loader, device)  # (N, D)
    if feats.shape[0] == 0:
        return {"error": "empty feature matrix"}

    # Center features along the sample axis
    centered = feats - feats.mean(axis=0, keepdims=True)

    # Top right-singular vector via thin SVD (only need V[0])
    # Use np.linalg.svd on centered matrix (N, D) -> U (N, k), S (k,), Vt (k, D)
    # For efficiency, compute on min(N,D) -> usually D=512, N=5000.
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    top_v = Vt[0]                                     # (D,)

    # Score each sample by squared projection onto top_v
    scores = (centered @ top_v) ** 2                  # (N,)

    # Map back: target_idx[k] is the position in the full dataset
    target_idx_arr = np.asarray(target_idx)
    poison_set = poisoned_dataset.poison_indices    # set of full-dataset indices
    is_poison_in_target = np.array([idx in poison_set for idx in target_idx_arr], dtype=bool)

    n_target = int(len(target_idx_arr))
    n_poison_in_target = int(is_poison_in_target.sum())

    # Remove top removal_multiplier * poison_ratio * n_target samples
    n_flag = int(np.ceil(removal_multiplier * poison_ratio * n_target))
    n_flag = min(max(n_flag, 1), n_target)
    flagged_order = np.argsort(scores)[::-1][:n_flag]   # highest scores first

    flagged_mask = np.zeros(n_target, dtype=bool)
    flagged_mask[flagged_order] = True

    n_caught = int((flagged_mask & is_poison_in_target).sum())
    precision = n_caught / n_flag if n_flag > 0 else 0.0
    recall = n_caught / n_poison_in_target if n_poison_in_target > 0 else 0.0

    return {
        "n_target_class_samples": n_target,
        "n_poison_in_target": n_poison_in_target,
        "n_flagged": int(n_flag),
        "n_caught": n_caught,
        "precision": float(precision),
        "recall": float(recall),
        "removal_multiplier": removal_multiplier,
        "poison_ratio": poison_ratio,
        "score_mean_poison": float(scores[is_poison_in_target].mean()) if n_poison_in_target else None,
        "score_mean_clean": float(scores[~is_poison_in_target].mean()) if (n_target - n_poison_in_target) > 0 else None,
    }
