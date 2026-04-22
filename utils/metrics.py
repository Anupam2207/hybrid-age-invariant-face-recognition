"""Verification metrics and threshold selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


@dataclass
class VerificationMetrics:
    accuracy: float
    auc: float
    best_threshold: float
    far: float
    frr: float
    eer: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def _far_frr(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(np.int32)
    predictions = predictions.astype(np.int32)

    false_accepts = np.sum((labels == 0) & (predictions == 1))
    true_rejects = np.sum((labels == 0) & (predictions == 0))
    false_rejects = np.sum((labels == 1) & (predictions == 0))
    true_accepts = np.sum((labels == 1) & (predictions == 1))

    far = false_accepts / max(false_accepts + true_rejects, 1)
    frr = false_rejects / max(false_rejects + true_accepts, 1)
    return float(far), float(frr)


def evaluate_verification_scores(scores: np.ndarray, labels: np.ndarray) -> VerificationMetrics:
    """Compute verification metrics from similarity scores and binary labels."""

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    candidate_thresholds: List[float] = sorted(set([float(th) for th in thresholds] + [0.0, 0.25, 0.5, 0.75, 1.0]))
    best_accuracy = -1.0
    best_threshold = 0.5
    best_far = 1.0
    best_frr = 1.0
    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(np.int32)
        accuracy = accuracy_score(labels, predictions)
        far, frr = _far_frr(labels, predictions)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_threshold = float(threshold)
            best_far = far
            best_frr = frr

    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    return VerificationMetrics(
        accuracy=best_accuracy,
        auc=float(auc),
        best_threshold=best_threshold,
        far=best_far,
        frr=best_frr,
        eer=eer,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )
