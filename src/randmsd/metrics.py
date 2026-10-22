"""Fixed-class semantic-segmentation confusion matrices (rows = ground truth)."""

import numpy as np


def confusion_matrix(labels, predictions, num_classes):
    labels = np.asarray(labels).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape")
    if not np.issubdtype(labels.dtype, np.integer) or not np.issubdtype(predictions.dtype, np.integer):
        raise ValueError("Class IDs must be integers")
    valid = (labels >= 0) & (labels < num_classes)
    if np.any((predictions[valid] < 0) | (predictions[valid] >= num_classes)):
        raise ValueError("Prediction outside the configured class range")
    return np.bincount(num_classes * labels[valid].astype(np.int64)
                       + predictions[valid].astype(np.int64),
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def segmentation_metrics(confusion):
    """Return percentages; classes with zero union/support contribute zero.

    The fixed-class mean is intentionally explicit. Never average per-cloud
    mIoUs to obtain a dataset score: sum their confusion matrices first.
    """
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError("Confusion matrix must be square and nonnegative")
    true_count, predicted_count = matrix.sum(1), matrix.sum(0)
    correct = np.diag(matrix)
    union = true_count + predicted_count - correct
    iou = np.divide(correct, union, out=np.zeros_like(correct, dtype=float), where=union > 0)
    accuracy = np.divide(correct, true_count, out=np.zeros_like(correct, dtype=float), where=true_count > 0)
    total = int(matrix.sum())
    return {"miou": float(iou.mean() * 100), "macc": float(accuracy.mean() * 100),
            "oa": float(correct.sum() / total * 100) if total else 0.0,
            "per_class_iou": (iou * 100).tolist(),
            "per_class_accuracy": (accuracy * 100).tolist(),
            "point_count": total, "absent_class_policy": "fixed_class_mean_zero"}
