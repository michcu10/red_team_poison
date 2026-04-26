"""
Fine-Pruning defense (Liu, Dolan-Gavitt, Garg 2018).

Idea: backdoor neurons are typically dormant on clean inputs. We rank channels in
the last residual block by their average activation on a clean fine-tuning subset,
prune the lowest-activation channels (zero-mask), then briefly fine-tune the
remaining weights on the clean subset. This can reduce ASR while keeping clean
accuracy intact, but its effect is trigger-dependent.

Implementation notes:
- We hook `model.layer4` (the final residual stage) to gather mean per-channel
  activations on a small clean subset.
- "Pruning" is implemented by registering a forward hook that zeroes the selected
  channels, leaving model weights untouched until fine-tuning.
- Fine-tuning trains for a few epochs on the clean subset with a small LR.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@torch.no_grad()
def _mean_channel_activation(model, layer, loader, device, max_batches=None):
    """Average per-channel activation magnitude (post-ReLU equivalent: |x|.mean) of `layer`."""
    sums = None
    count = 0
    captured = {}

    def hook(_, __, output):
        captured["x"] = output

    handle = layer.register_forward_hook(hook)
    model.eval()
    try:
        for i, batch in enumerate(loader):
            x = batch[0].to(device, non_blocking=True)
            _ = model(x)
            act = captured["x"]                 # (B, C, H, W)
            per_chan = act.mean(dim=(0, 2, 3))  # (C,)
            if sums is None:
                sums = per_chan.detach().clone()
            else:
                sums += per_chan.detach()
            count += 1
            if max_batches is not None and i + 1 >= max_batches:
                break
    finally:
        handle.remove()

    return (sums / max(count, 1)).cpu()


def _make_prune_hook(channels_to_zero, n_channels):
    """Return a forward hook that zeroes the given channel indices in the layer output.

    Uses an out-of-place multiplicative mask (not in-place assignment) so the hook is
    safe under autograd during fine-tuning — in-place mutation of a relu output would
    bump its version counter and break gradient computation."""
    mask = torch.ones(n_channels, dtype=torch.float32)
    if len(channels_to_zero) > 0:
        mask[torch.as_tensor(channels_to_zero, dtype=torch.long)] = 0.0

    def hook(_, __, output):
        m = mask.to(output.device, dtype=output.dtype)
        return output * m.view(1, -1, 1, 1)

    return hook


def evaluate_ca_asr(model, clean_loader, attack_loader, device, target_class=2):
    """Returns (clean_accuracy, attack_success_rate). attack_loader may be None."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in clean_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    ca = 100.0 * correct / max(total, 1)

    asr = None
    if attack_loader is not None:
        succ = total = 0
        with torch.no_grad():
            for batch in attack_loader:
                x = batch[0].to(device)
                pred = model(x).argmax(dim=1)
                succ += (pred == target_class).sum().item()
                total += x.size(0)
        asr = 100.0 * succ / max(total, 1)
    return ca, asr


def run_fine_pruning(model, clean_subset_loader, clean_eval_loader, attack_loader,
                     device, prune_ratios=(0.1, 0.2, 0.3),
                     finetune_epochs=5, finetune_lr=1e-4, target_class=2):
    """
    Execute Fine-Pruning at multiple prune ratios. Returns a list of dicts with
    {prune_ratio, ca, asr, ca_drop, asr_drop} for each ratio.

    Parameters
    ----------
    model : nn.Module — already loaded, on `device`. Will NOT be modified
        in place; the function operates on deep copies.
    clean_subset_loader : DataLoader for fine-tuning + activation ranking
        (small clean subset).
    clean_eval_loader : DataLoader for clean accuracy evaluation.
    attack_loader : DataLoader of triggered inputs for ASR (None for Clean Model).
    """
    model.eval()
    base_ca, base_asr = evaluate_ca_asr(model, clean_eval_loader, attack_loader, device, target_class)

    # Rank channels in layer4 by mean activation
    activations = _mean_channel_activation(model, model.layer4, clean_subset_loader, device)
    n_channels = activations.numel()
    sort_idx = torch.argsort(activations)  # ascending: lowest activation first

    results = [{
        "prune_ratio": 0.0,
        "ca": base_ca,
        "asr": base_asr,
        "ca_drop": 0.0,
        "asr_drop": 0.0,
        "n_channels_pruned": 0,
    }]

    for r in prune_ratios:
        n_prune = int(round(r * n_channels))
        prune_channels = sort_idx[:n_prune].tolist()

        # Deep copy so each ratio is evaluated independently
        m = copy.deepcopy(model).to(device)
        handle = m.layer4.register_forward_hook(_make_prune_hook(prune_channels, n_channels))

        # Fine-tune all parameters with a small LR; each prune ratio uses a fresh copy.
        m.train()
        for p in m.parameters():
            p.requires_grad = True
        opt = optim.SGD([p for p in m.parameters() if p.requires_grad],
                        lr=finetune_lr, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        for _ in range(finetune_epochs):
            for batch in clean_subset_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                opt.zero_grad()
                loss = criterion(m(x), y)
                loss.backward()
                opt.step()
        m.eval()

        ca, asr = evaluate_ca_asr(m, clean_eval_loader, attack_loader, device, target_class)
        results.append({
            "prune_ratio": float(r),
            "ca": float(ca),
            "asr": float(asr) if asr is not None else None,
            "ca_drop": float(base_ca - ca),
            "asr_drop": float(base_asr - asr) if (base_asr is not None and asr is not None) else None,
            "n_channels_pruned": int(n_prune),
        })

        handle.remove()
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Pick the best operating point: largest ASR drop with CA drop <= 2.0pp.
    best = None
    for r in results[1:]:
        if r["asr_drop"] is None:
            continue
        if r["ca_drop"] <= 2.0 and (best is None or r["asr_drop"] > best["asr_drop"]):
            best = r

    return {
        "baseline": results[0],
        "ratios": results[1:],
        "best": best,
        "n_channels": int(n_channels),
    }
