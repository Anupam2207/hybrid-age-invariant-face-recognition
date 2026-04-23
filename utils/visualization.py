"""Plotting utilities for training curves, ROC curves, and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def plot_training_curves(history: Dict[str, list], save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'train_total_loss' in history and history['train_total_loss']:
        plt.figure(figsize=(7, 5))
        plt.plot(history['train_total_loss'], marker='o', label='Total loss')
        if 'train_triplet_loss' in history and history['train_triplet_loss']:
            plt.plot(history['train_triplet_loss'], marker='s', label='Triplet loss')
        if 'train_identity_loss' in history and history['train_identity_loss']:
            plt.plot(history['train_identity_loss'], marker='^', label='Identity loss')
        if 'train_age_loss' in history and history['train_age_loss']:
            plt.plot(history['train_age_loss'], marker='d', label='Age loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / 'train_losses.png', dpi=200)
        plt.close()

    metric_keys = [
        ('val_accuracy', 'Accuracy'),
        ('val_precision', 'Precision'),
        ('val_recall', 'Recall'),
        ('val_f1', 'F1'),
        ('val_auc', 'AUC'),
    ]
    available = [(key, label) for key, label in metric_keys if key in history and history[key]]
    if available:
        plt.figure(figsize=(8, 5))
        for key, label in available:
            plt.plot(history[key], marker='o', label=label)
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / 'val_metrics.png', dpi=200)
        plt.close()



def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Acceptance Rate')
    plt.ylabel('True Acceptance Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()



def plot_age_gap_performance(age_gap_metrics: list[dict], save_path: str | Path, metric_key: str = 'accuracy') -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    values = []
    counts = []
    for row in age_gap_metrics:
        labels.append(row['age_gap_bin'])
        values.append(0.0 if row[metric_key] is None else float(row[metric_key]))
        counts.append(int(row['num_pairs']))

    if not labels:
        return

    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    for idx, (value, count) in enumerate(zip(values, counts)):
        plt.text(idx, value + 0.01, f'n={count}', ha='center', va='bottom', fontsize=8)
    plt.ylim(0.0, 1.05)
    plt.xlabel('Age gap bin (years)')
    plt.ylabel(metric_key.replace('_', ' ').title())
    title_metric = metric_key.replace('_', ' ').title()
    plt.title(f'{title_metric} Across Age Gaps')
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()



def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    ages: Optional[np.ndarray],
    save_path: str | Path,
    method: str = 'tsne',
    random_state: int = 42,
) -> None:
    """Project embeddings to 2-D for qualitative visualization."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    if ages is not None:
        ages = np.asarray(ages, dtype=np.float32)

    if len(embeddings) < 3:
        return

    if method.lower() == 'tsne' and len(embeddings) >= 10:
        projector = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=random_state)
        projected = projector.fit_transform(embeddings)
    else:
        projector = PCA(n_components=2, random_state=random_state)
        projected = projector.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], c=labels, s=24, alpha=0.8)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Embedding Visualization by Identity')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path.with_name(save_path.stem + '_by_identity' + save_path.suffix), dpi=220)
    plt.close()

    if ages is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(projected[:, 0], projected[:, 1], c=ages, s=24, alpha=0.8)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Embedding Visualization by Age')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path.with_name(save_path.stem + '_by_age' + save_path.suffix), dpi=220)
        plt.close()
