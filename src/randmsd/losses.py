"""Dataset-level class weights and the historical point-mean weighted CE."""

import numpy as np
import torch
from torch.nn import functional as F


LEGACY_CLASS_COUNTS = {
    "S3DIS": [3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
              650464, 791496, 88727, 1284130, 229758, 2272837],
    "SemanticKITTI": [55437630, 320797, 541736, 2578735, 3274484, 552662,
                      184064, 78858, 240942562, 17294618, 170599734, 6369672,
                      230413074, 101130274, 476491114, 9833174, 129609852,
                      4506626, 1168181],
}


def class_weights(counts, mode="legacy"):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not counts.size or not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Every trained class must have a positive finite count")
    frequency = counts / counts.sum()
    if mode == "legacy":
        weights = 1.0 / (frequency + 0.02)
    elif mode == "inverse_sqrt":
        weights = 1.0 / np.sqrt(frequency)
    else:
        raise ValueError("class_weight_mode must be legacy or inverse_sqrt")
    return torch.tensor(weights, dtype=torch.float32)


def weighted_cross_entropy(logits, labels, weights, label_smoothing=0.0):
    """Ignore -1 and average weighted losses over valid points, not weights.

    PyTorch's default weighted-CE ``mean`` divides by the sum of target
    weights. The original experiment instead used ``reduction='none'`` and
    then a point mean; keep that loss scale unchanged here. With smoothing
    epsilon, the uniform term is also weighted class by class:
    -(1-epsilon)*w[y]*log(p[y]) - epsilon/C*sum_c(w[c]*log(p[c])).
    Multiplying unweighted smoothed CE by the target weight is NOT equivalent.
    """
    if not 0 <= label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if logits.ndim != 3 or labels.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("Expected logits [B,C,N] and labels [B,N]")
    labels = labels.reshape(-1).long()
    values = logits.transpose(1, 2).reshape(-1, logits.shape[1])
    valid = labels != -1
    if not valid.any():
        return logits.sum() * 0.0
    if torch.any((labels[valid] < 0) | (labels[valid] >= logits.shape[1])):
        raise ValueError("Training class ID outside [-1, C-1]")
    return F.cross_entropy(values[valid], labels[valid], weight=weights.to(logits.device),
                           reduction="none", label_smoothing=float(label_smoothing)).mean()
