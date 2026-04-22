"""Plotting utilities for training curves, ROC curves, and embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_training_curves(history: Dict[str, list], save_dir: str | Path) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if 'train_loss' in history and history['train_loss']:
        plt.figure(figsize=(7, 5))
        plt.plot(history['train_loss'], marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Loss')
        plt.title('Training Loss')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_dir / 'train_loss.png', dpi=200)
        plt.close()

    if 'val_accuracy' in history and history['val_accuracy']:
        plt.figure(figsize=(7, 5))
        plt.plot(history['val_accuracy'], marker='o', label='Accuracy')
        if 'val_auc' in history and history['val_auc']:
            plt.plot(history['val_auc'], marker='s', label='AUC')
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
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, s=24, alpha=0.8)
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
