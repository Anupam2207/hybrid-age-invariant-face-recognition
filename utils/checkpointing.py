"""Checkpoint helpers for training, evaluation, and inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from dataset import GeometryStats, compute_geometry_stats
from model import HybridFaceRecognizer


@dataclass
class CheckpointBundle:
    model: HybridFaceRecognizer
    metadata: Dict[str, Any]
    geometry_stats: Optional[GeometryStats]
    state_dict: Dict[str, torch.Tensor]
    raw_checkpoint: Any



def _to_geometry_stats(value: Any) -> Optional[GeometryStats]:
    if value is None:
        return None
    if isinstance(value, GeometryStats):
        return value
    if isinstance(value, dict) and 'mean' in value and 'std' in value:
        return GeometryStats(
            mean=np.asarray(value['mean'], dtype=np.float32),
            std=np.asarray(value['std'], dtype=np.float32),
        )
    raise ValueError('Unsupported geometry_stats format in checkpoint.')



def infer_legacy_metadata(
    state_dict: Dict[str, torch.Tensor],
    default_image_size: int = 160,
) -> Dict[str, Any]:
    """Infer model configuration from a legacy plain state_dict checkpoint."""

    if 'geom_branch.network.0.weight' not in state_dict:
        raise ValueError('Could not infer geometry input dimension from legacy checkpoint.')

    geometry_input_dim = int(state_dict['geom_branch.network.0.weight'].shape[1])
    geom_hidden_dim = int(state_dict['geom_branch.network.0.weight'].shape[0])
    geom_embedding_dim = int(state_dict['geom_branch.network.4.weight'].shape[0])
    deep_embedding_dim = int(state_dict['deep_branch.projection.0.weight'].shape[0])
    fusion_hidden_dim = int(state_dict['fusion_head.0.weight'].shape[0])
    final_embedding_dim = int(state_dict['fusion_head.4.weight'].shape[0])
    fusion_input_dim = int(state_dict['fusion_head.0.weight'].shape[1])

    if fusion_input_dim == deep_embedding_dim + geom_embedding_dim:
        mode = 'hybrid'
    elif fusion_input_dim == deep_embedding_dim:
        mode = 'cnn_only'
    elif fusion_input_dim == geom_embedding_dim:
        mode = 'geom_only'
    else:
        raise ValueError('Could not infer model mode from fusion head dimensions.')

    backbone = 'mobilenet_v2' if 'deep_branch.features.18.0.weight' in state_dict else 'resnet18'

    return {
        'geometry_input_dim': geometry_input_dim,
        'backbone': backbone,
        'mode': mode,
        'embedding_dim': final_embedding_dim,
        'deep_embedding_dim': deep_embedding_dim,
        'geom_hidden_dim': geom_hidden_dim,
        'geom_embedding_dim': geom_embedding_dim,
        'fusion_hidden_dim': fusion_hidden_dim,
        'dropout': 0.2,
        'image_size': int(default_image_size),
        'best_threshold': 0.5,
        'checkpoint_format': 'legacy_state_dict',
    }



def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    device: torch.device | str,
    processed_csv: str | Path | None = None,
    train_split: str = 'train',
    legacy_image_size: int = 160,
    legacy_backbone: str | None = None,
    legacy_mode: str | None = None,
) -> CheckpointBundle:
    """Load either a modern rich checkpoint or a legacy plain state_dict."""

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw_checkpoint, dict) and 'model_state_dict' in raw_checkpoint:
        state_dict = raw_checkpoint['model_state_dict']
        metadata = {
            'geometry_input_dim': int(raw_checkpoint['geometry_input_dim']),
            'backbone': raw_checkpoint['backbone'],
            'mode': raw_checkpoint['mode'],
            'embedding_dim': int(raw_checkpoint['embedding_dim']),
            'deep_embedding_dim': int(raw_checkpoint.get('deep_embedding_dim', raw_checkpoint['embedding_dim'])),
            'geom_hidden_dim': int(raw_checkpoint['geom_hidden_dim']),
            'geom_embedding_dim': int(raw_checkpoint['geom_embedding_dim']),
            'fusion_hidden_dim': int(raw_checkpoint['fusion_hidden_dim']),
            'dropout': float(raw_checkpoint.get('dropout', 0.2)),
            'image_size': int(raw_checkpoint.get('image_size', legacy_image_size)),
            'best_threshold': float(raw_checkpoint.get('best_threshold', 0.5)),
            'checkpoint_format': 'rich_checkpoint',
        }
        geometry_stats = _to_geometry_stats(raw_checkpoint.get('geometry_stats'))
    elif isinstance(raw_checkpoint, (dict, OrderedDict)):
        state_dict = dict(raw_checkpoint)
        metadata = infer_legacy_metadata(state_dict, default_image_size=legacy_image_size)
        if legacy_backbone is not None:
            metadata['backbone'] = legacy_backbone
        if legacy_mode is not None:
            metadata['mode'] = legacy_mode
        geometry_stats = compute_geometry_stats(processed_csv, split=train_split) if processed_csv is not None else None
    else:
        raise TypeError(f'Unsupported checkpoint format: {type(raw_checkpoint)!r}')

    model = HybridFaceRecognizer(
        geometry_input_dim=metadata['geometry_input_dim'],
        backbone=metadata['backbone'],
        mode=metadata['mode'],
        deep_embedding_dim=metadata['deep_embedding_dim'],
        geom_hidden_dim=metadata['geom_hidden_dim'],
        geom_embedding_dim=metadata['geom_embedding_dim'],
        fusion_hidden_dim=metadata['fusion_hidden_dim'],
        final_embedding_dim=metadata['embedding_dim'],
        pretrained=False,
        dropout=metadata['dropout'],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return CheckpointBundle(
        model=model,
        metadata=metadata,
        geometry_stats=geometry_stats,
        state_dict=state_dict,
        raw_checkpoint=raw_checkpoint,
    )
