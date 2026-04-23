"""Verification metrics and threshold selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class VerificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
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



def _candidate_thresholds(scores: np.ndarray, roc_thresholds: np.ndarray) -> List[float]:
    fixed = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]
    dynamic = [float(value) for value in np.unique(scores)]
    roc_based = [float(value) for value in np.unique(roc_thresholds[np.isfinite(roc_thresholds)])]
    merged = sorted(set(fixed + dynamic + roc_based))
    return merged



def _compute_scalar_metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float, float, float]:
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    return float(accuracy), float(precision), float(recall), float(f1)



def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    strategy: str = 'accuracy',
) -> float:
    """Select a decision threshold on a validation split."""

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    thresholds = _candidate_thresholds(scores, roc_thresholds)

    best_value = float('-inf')
    best_threshold = 0.5
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(np.int32)
        accuracy, precision, recall, f1 = _compute_scalar_metrics(labels, predictions)
        far, frr = _far_frr(labels, predictions)

        if strategy == 'f1':
            criterion = f1
        elif strategy == 'eer':
            criterion = -(abs(far - frr))
        else:
            criterion = accuracy

        if criterion > best_value:
            best_value = criterion
            best_threshold = float(threshold)

    return best_threshold



def evaluate_verification_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None = None,
    threshold_strategy: str = 'accuracy',
) -> VerificationMetrics:
    """Compute verification metrics from similarity scores and binary labels.

    If threshold is None, it is selected on the provided split. For IEEE-style
    reporting, threshold should be chosen on validation and then reused on test.
    """

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    if len(np.unique(labels)) < 2:
        raise ValueError('Both positive and negative pairs are required to compute ROC/AUC.')

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    selected_threshold = (
        select_threshold(scores, labels, strategy=threshold_strategy)
        if threshold is None
        else float(threshold)
    )

    predictions = (scores >= selected_threshold).astype(np.int32)
    accuracy, precision, recall, f1 = _compute_scalar_metrics(labels, predictions)
    far, frr = _far_frr(labels, predictions)

    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    return VerificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc=float(auc),
        best_threshold=float(selected_threshold),
        far=far,
        frr=frr,
        eer=eer,
        fpr=fpr,
        tpr=tpr,
        thresholds=roc_thresholds,
    )



def evaluate_age_gap_bins(
    scores: np.ndarray,
    labels: np.ndarray,
    age_gaps: np.ndarray,
    threshold: float,
    bins: Iterable[float],
) -> List[dict]:
    """Evaluate performance separately across different age-gap ranges."""

    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    age_gaps = np.asarray(age_gaps, dtype=np.float32)
    bins = list(float(value) for value in bins)
    if len(bins) < 2:
        raise ValueError('At least two bin edges are required.')

    results: List[dict] = []
    predictions = (scores >= float(threshold)).astype(np.int32)

    for left, right in zip(bins[:-1], bins[1:]):
        if right == bins[-1]:
            mask = (age_gaps >= left) & (age_gaps <= right)
            label = f'[{left:.0f}, {right:.0f}]'
        else:
            mask = (age_gaps >= left) & (age_gaps < right)
            label = f'[{left:.0f}, {right:.0f})'

        if int(mask.sum()) == 0:
            results.append(
                {
                    'age_gap_bin': label,
                    'num_pairs': 0,
                    'accuracy': None,
                    'precision': None,
                    'recall': None,
                    'f1': None,
                    'far': None,
                    'frr': None,
                }
            )
            continue

        bin_labels = labels[mask]
        bin_predictions = predictions[mask]
        accuracy, precision, recall, f1 = _compute_scalar_metrics(bin_labels, bin_predictions)
        far, frr = _far_frr(bin_labels, bin_predictions)
        results.append(
            {
                'age_gap_bin': label,
                'num_pairs': int(mask.sum()),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'far': far,
                'frr': frr,
            }
        )

    return results
