"""
Evaluation Suite — Intent-Aware XR Framework
============================================

This module provides specialized metrics for early action prediction in XR.
Unlike standard classification, we focus on *how early* and *how reliably* 
a model can predict an intent before the physical contact occurs.

Key Metrics Implemented:
-----------------------
1. Precision @ Observation Ratio (p_at_obs_ratio):
   Measures accuracy at specific early stages of a motion (e.g., first 20%, 
   25%, 30% of the trajectory). This validates the "Early Prediction" goal.

2. Lead Time:
   Calculates the average temporal distance (in frames) between the model's 
   first confident prediction (confidence > threshold) and the actual 
   frame of contact. High lead time = more time for proactive XR effects.

3. Time-to-Contact (TTC):
   An estimate (in seconds) of the remaining time before the hand reaches 
   the object, derived from the clip's metadata and observation ratio.

4. Ghosting Trigger Rate:
   The percentage of samples where the model's confidence exceeds the 
   threshold (usually 0.65), indicating how often the "Ghost Hand" 
   effect would be visible to the user.

Functions:
---------
- compute_metrics: Standard precision, recall, F1, and accuracy.
- precision_at_obs_ratio: Bucket-based precision analysis.
- compute_ttc: Geometric estimation of remaining time.
- evaluate_model: Full validation pass returning all the above.
"""

from __future__ import annotations

import torch
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────
# Core precision / recall / F1
# ─────────────────────────────────────────────────────────

def compute_metrics(
    preds:      torch.Tensor,   # (N,)  predicted class indices
    labels:     torch.Tensor,   # (N,)  ground-truth class indices
    num_classes: int,
) -> dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, F1 and overall Accuracy.

    Precision_c = TP_c / (TP_c + FP_c)
    Recall_c    = TP_c / (TP_c + FN_c)
    F1_c        = 2 * P_c * R_c / (P_c + R_c)

    Returns a dict with keys:
        precision, recall, f1, accuracy
    """
    preds  = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for c in range(num_classes):
        tp[c] = np.sum((preds == c) & (labels == c))
        fp[c] = np.sum((preds == c) & (labels != c))
        fn[c] = np.sum((preds != c) & (labels == c))

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1   = np.where(prec + rec > 0,
                        2 * prec * rec / (prec + rec), 0.0)

    return {
        "precision": float(prec.mean()),
        "recall":    float(rec.mean()),
        "f1":        float(f1.mean()),
        "accuracy":  float((preds == labels).mean()),
    }


# ─────────────────────────────────────────────────────────
# Precision @ Observation Ratio
# ─────────────────────────────────────────────────────────

def precision_at_obs_ratio(
    preds:       list[int],          # predicted class per sample
    labels:      list[int],          # GT class per sample
    obs_ratios:  list[float],        # obs_ratio per sample
    ratio_bins:  list[float] | None = None,
) -> dict[str, float]:
    """
    Group samples by obs_ratio bucket and compute precision per bucket.

    Returns dict: { '≤0.20': precision, '≤0.25': ..., '≤0.30': ... }
    """
    if ratio_bins is None:
        ratio_bins = [0.20, 0.25, 0.30]

    results = {}
    for threshold in ratio_bins:
        mask = [r <= threshold + 1e-6 for r in obs_ratios]
        if not any(mask):
            results[f"≤{threshold:.2f}"] = 0.0
            continue
        tp = sum(
            1 for p, l, m in zip(preds, labels, mask) if m and p == l
        )
        fp = sum(
            1 for p, l, m in zip(preds, labels, mask) if m and p != l
        )
        results[f"≤{threshold:.2f}"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return results


# ─────────────────────────────────────────────────────────
# Lead Time
# ─────────────────────────────────────────────────────────

def compute_lead_time(
    end_act_frames:  list[int],     # frame index of action-end (contact) per sample
    pred_frame:      list[int],     # frame at which prediction was first made
) -> float:
    """
    Lead Time = mean(end_act_frames - pred_frame) in frames.
    Only counts samples where pred_frame < end_act_frames.
    """
    lead_times = [
        e - p for e, p in zip(end_act_frames, pred_frame) if e > p
    ]
    return float(np.mean(lead_times)) if lead_times else 0.0


# ─────────────────────────────────────────────────────────
# Time-to-Contact (TTC)
# ─────────────────────────────────────────────────────────

def compute_ttc(
    start_act:    int,
    end_act:      int,
    obs_ratio:    float,
    fps:          int = 30,
) -> float:
    """
    Estimate the TTC (seconds) from the moment a window is sampled
    until the expected contact frame.

    obs_end  = start_act + int(action_len * obs_ratio)
    TTC      = (end_act - obs_end) / fps

    Returns TTC in seconds.
    """
    action_len = end_act - start_act + 1
    obs_end    = start_act + max(1, int(action_len * obs_ratio))
    ttc_frames = max(0, end_act - obs_end)
    return ttc_frames / fps


def compute_batch_ttc(
    start_acts: list[int],
    end_acts:   list[int],
    obs_ratios: list[float],
    fps:        int = 30,
) -> list[float]:
    """Vectorised version of compute_ttc."""
    return [
        compute_ttc(s, e, r, fps)
        for s, e, r in zip(start_acts, end_acts, obs_ratios)
    ]


# ─────────────────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    fps: int = 30,
    confidence_threshold: float = 0.65,
) -> dict:
    """
    Run a full evaluation pass and return a comprehensive metrics dict.

    Returns:
        accuracy, precision, recall, f1,
        precision_at_020, precision_at_025, precision_at_030,
        mean_ttc_seconds, mean_lead_time_frames,
        ghosting_trigger_rate  (fraction of samples above confidence threshold)
    """
    model.eval()
    all_preds, all_labels = [], []
    all_obs, all_probs    = [], []

    from src.data.h2o_dataset import NUM_CLASSES

    for batch in loader:
        hand   = batch["hand_flat"].to(device)
        obj    = batch["obj_rt"].to(device)
        obs    = batch["obs_ratio"].to(device)
        labels = batch["label"].to(device)

        logits = model(hand, obj, obs)
        probs  = logits.softmax(dim=-1)
        preds  = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_obs.append(obs.cpu())
        all_probs.append(probs.max(dim=-1).values.cpu())

    preds_t  = torch.cat(all_preds).numpy().tolist()
    labels_t = torch.cat(all_labels).numpy().tolist()
    obs_t    = torch.cat(all_obs).numpy().tolist()
    probs_t  = torch.cat(all_probs).numpy().tolist()

    base = compute_metrics(
        torch.tensor(preds_t), torch.tensor(labels_t), NUM_CLASSES
    )
    p_at_ratio = precision_at_obs_ratio(preds_t, labels_t, obs_t)

    # Ghosting trigger rate: fraction where max_prob ≥ threshold
    ghosting_rate = float(np.mean([p >= confidence_threshold for p in probs_t]))

    result = {**base}
    result.update(p_at_ratio)
    result["ghosting_trigger_rate"] = ghosting_rate

    return result
