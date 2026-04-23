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
    default_image_size: int = 224,
) -> Dict[str, Any]:
    """Infer model configuration from a plain legacy state_dict checkpoint."""

    if 'geom_branch.network.0.weight' not in state_dict:
        raise ValueError('Could not infer geometry input dimension from legacy checkpoint.')

    geometry_input_dim = int(state_dict['geom_branch.network.0.weight'].shape[1])
    geom_hidden_dim = int(state_dict['geom_branch.network.0.weight'].shape[0])
    geom_embedding_dim = int(state_dict['geom_branch.network.4.weight'].shape[0])
    deep_embedding_dim = int(state_dict['deep_branch.projection.0.weight'].shape[0])

    projection_input_dim = int(state_dict['deep_branch.projection.0.weight'].shape[1])
    if projection_input_dim == 1280:
        backbone = 'mobilenet_v2'
    elif projection_input_dim == 512:
        backbone = 'resnet18'
    elif projection_input_dim == 2048:
        backbone = 'resnet50'
    else:
        raise ValueError(f'Could not infer backbone from projection input dim={projection_input_dim}.')

    enable_identity_head = 'identity_classifier.weight' in state_dict
    enable_age_head = any(key.startswith('age_regressor.') for key in state_dict.keys())
    num_identity_classes = (
        int(state_dict['identity_classifier.weight'].shape[0])
        if enable_identity_head
        else None
    )

    if 'fusion_head.network.0.weight' in state_dict:
        fusion_hidden_dim = int(state_dict['fusion_head.network.0.weight'].shape[0])
        final_embedding_dim = int(state_dict['fusion_head.network.4.weight'].shape[0])
        fusion_input_dim = int(state_dict['fusion_head.network.0.weight'].shape[1])
        fusion_type = 'concat'
    elif 'fusion_head.output.0.weight' in state_dict:
        fusion_hidden_dim = int(state_dict['fusion_head.output.0.weight'].shape[0])
        final_embedding_dim = int(state_dict['fusion_head.output.3.weight'].shape[0])
        fusion_input_dim = deep_embedding_dim + geom_embedding_dim
        fusion_type = 'attention'
    else:
        # Legacy cnn_only / geom_only fusion used a plain Sequential.
        fusion_hidden_dim = int(state_dict['fusion_head.0.weight'].shape[0])
        final_embedding_dim = int(state_dict['fusion_head.4.weight'].shape[0])
        fusion_input_dim = int(state_dict['fusion_head.0.weight'].shape[1])
        fusion_type = 'concat'

    if fusion_input_dim == deep_embedding_dim + geom_embedding_dim:
        mode = 'hybrid'
    elif fusion_input_dim == deep_embedding_dim:
        mode = 'cnn_only'
    elif fusion_input_dim == geom_embedding_dim:
        mode = 'geom_only'
    else:
        raise ValueError('Could not infer model mode from fusion head dimensions.')

    return {
        'geometry_input_dim': geometry_input_dim,
        'backbone': backbone,
        'mode': mode,
        'fusion_type': fusion_type,
        'attention_heads': 4,
        'embedding_dim': final_embedding_dim,
        'deep_embedding_dim': deep_embedding_dim,
        'geom_hidden_dim': geom_hidden_dim,
        'geom_embedding_dim': geom_embedding_dim,
        'fusion_hidden_dim': fusion_hidden_dim,
        'dropout': 0.2,
        'image_size': int(default_image_size),
        'best_threshold': 0.5,
        'enable_identity_head': enable_identity_head,
        'enable_age_head': enable_age_head,
        'num_identity_classes': num_identity_classes,
        'checkpoint_format': 'legacy_state_dict',
    }



def _extract_metadata_from_rich_checkpoint(raw_checkpoint: Dict[str, Any], default_image_size: int) -> Dict[str, Any]:
    if 'model_config' in raw_checkpoint:
        model_config = dict(raw_checkpoint['model_config'])
    else:
        model_config = {
            'geometry_input_dim': int(raw_checkpoint['geometry_input_dim']),
            'backbone': raw_checkpoint['backbone'],
            'mode': raw_checkpoint['mode'],
            'embedding_dim': int(raw_checkpoint['embedding_dim']),
            'deep_embedding_dim': int(raw_checkpoint.get('deep_embedding_dim', raw_checkpoint['embedding_dim'])),
            'geom_hidden_dim': int(raw_checkpoint['geom_hidden_dim']),
            'geom_embedding_dim': int(raw_checkpoint['geom_embedding_dim']),
            'fusion_hidden_dim': int(raw_checkpoint['fusion_hidden_dim']),
            'dropout': float(raw_checkpoint.get('dropout', 0.2)),
            'fusion_type': raw_checkpoint.get('fusion_type', 'concat'),
            'attention_heads': int(raw_checkpoint.get('attention_heads', 4)),
            'enable_identity_head': bool(raw_checkpoint.get('enable_identity_head', False)),
            'enable_age_head': bool(raw_checkpoint.get('enable_age_head', False)),
            'num_identity_classes': raw_checkpoint.get('num_identity_classes'),
        }

    model_config['image_size'] = int(raw_checkpoint.get('image_size', default_image_size))
    model_config['best_threshold'] = float(raw_checkpoint.get('best_threshold', 0.5))
    model_config['checkpoint_format'] = 'rich_checkpoint'
    return model_config


def _upgrade_state_dict_keys(state_dict: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Upgrade older parameter names to the current module layout."""

    upgraded = dict(state_dict)

    if metadata.get('mode') == 'hybrid' and metadata.get('fusion_type', 'concat') == 'concat':
        legacy_keys = [key for key in upgraded.keys() if key.startswith('fusion_head.') and not key.startswith('fusion_head.network.')]
        if legacy_keys and 'fusion_head.network.0.weight' not in upgraded:
            for key in legacy_keys:
                suffix = key[len('fusion_head.'):]
                upgraded[f'fusion_head.network.{suffix}'] = upgraded.pop(key)

    return upgraded



def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    device: torch.device | str,
    processed_csv: str | Path | None = None,
    train_split: str = 'train',
    legacy_image_size: int = 224,
    legacy_backbone: str | None = None,
    legacy_mode: str | None = None,
) -> CheckpointBundle:
    """Load either a modern rich checkpoint or a legacy plain state_dict."""

    device = torch.device(device)
    checkpoint_path = Path(checkpoint_path)
    raw_checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw_checkpoint, dict) and 'model_state_dict' in raw_checkpoint:
        state_dict = raw_checkpoint['model_state_dict']
        metadata = _extract_metadata_from_rich_checkpoint(raw_checkpoint, default_image_size=legacy_image_size)
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

    state_dict = _upgrade_state_dict_keys(state_dict, metadata)

    model = HybridFaceRecognizer(
        geometry_input_dim=int(metadata['geometry_input_dim']),
        backbone=metadata['backbone'],
        mode=metadata['mode'],
        deep_embedding_dim=int(metadata.get('deep_embedding_dim', metadata['embedding_dim'])),
        geom_hidden_dim=int(metadata['geom_hidden_dim']),
        geom_embedding_dim=int(metadata['geom_embedding_dim']),
        fusion_hidden_dim=int(metadata['fusion_hidden_dim']),
        final_embedding_dim=int(metadata['embedding_dim']),
        pretrained=False,
        dropout=float(metadata.get('dropout', 0.2)),
        fusion_type=metadata.get('fusion_type', 'concat'),
        attention_heads=int(metadata.get('attention_heads', 4)),
        enable_identity_head=bool(metadata.get('enable_identity_head', False)),
        enable_age_head=bool(metadata.get('enable_age_head', False)),
        num_identity_classes=metadata.get('num_identity_classes'),
    ).to(device)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f'[warning] Missing checkpoint keys: {missing_keys}')
    if unexpected_keys:
        print(f'[warning] Unexpected checkpoint keys: {unexpected_keys}')
    model.eval()

    return CheckpointBundle(
        model=model,
        metadata=metadata,
        geometry_stats=geometry_stats,
        state_dict=state_dict,
        raw_checkpoint=raw_checkpoint,
    )
