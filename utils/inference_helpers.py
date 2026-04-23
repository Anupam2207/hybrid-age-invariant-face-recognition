"""Inference helpers for image-pair verification and UI demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import build_image_transform
from .checkpointing import CheckpointBundle
from .preprocessing import FacePreprocessor, ProcessedFace



def expected_geometry_dim(bundle: CheckpointBundle) -> int:
    if bundle.geometry_stats is not None:
        return int(len(bundle.geometry_stats.mean))
    return int(bundle.metadata['geometry_input_dim'])



def adapt_geometry_vector(geometry_features: np.ndarray, target_dim: int) -> np.ndarray:
    vector = np.asarray(geometry_features, dtype=np.float32).reshape(-1)
    if len(vector) == target_dim:
        return vector
    if len(vector) > target_dim:
        return vector[:target_dim]
    return np.pad(vector, (0, target_dim - len(vector)), mode='constant', constant_values=0.0)



def describe_similarity(score: float, threshold: float) -> str:
    margin = float(score - threshold)
    if margin >= 0.25:
        return 'Strong same-person match'
    if margin >= 0.10:
        return 'Moderate same-person match'
    if margin >= 0.00:
        return 'Borderline same-person match'
    if margin >= -0.10:
        return 'Borderline different-person result'
    if margin >= -0.25:
        return 'Moderate different-person result'
    return 'Strong different-person result'



def processed_face_to_tensors(
    processed_face: ProcessedFace,
    bundle: CheckpointBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_size = int(bundle.metadata['image_size'])
    transform = build_image_transform(train=False, image_size=image_size)

    image = Image.fromarray(processed_face.aligned_rgb)
    image_tensor = transform(image).unsqueeze(0).to(device)

    geometry = adapt_geometry_vector(processed_face.geometry_features, expected_geometry_dim(bundle))
    if bundle.geometry_stats is not None:
        mean = adapt_geometry_vector(bundle.geometry_stats.mean, len(geometry))
        std = adapt_geometry_vector(bundle.geometry_stats.std, len(geometry))
        geometry = (geometry - mean) / np.maximum(std, 1e-6)
    geometry_tensor = torch.tensor(geometry, dtype=torch.float32, device=device).unsqueeze(0)
    return image_tensor, geometry_tensor



def compare_processed_faces(
    bundle: CheckpointBundle,
    first: ProcessedFace,
    second: ProcessedFace,
    device: torch.device,
    threshold: float | None = None,
) -> dict:
    threshold = float(threshold if threshold is not None else bundle.metadata.get('best_threshold', 0.5))

    image1_tensor, geom1_tensor = processed_face_to_tensors(first, bundle, device)
    image2_tensor, geom2_tensor = processed_face_to_tensors(second, bundle, device)

    with torch.no_grad():
        _, _, similarity = bundle.model.forward_pair(image1_tensor, geom1_tensor, image2_tensor, geom2_tensor)
        score = float(similarity.item())

    decision_margin = float(score - threshold)
    result = {
        'similarity_score': score,
        'score_0_to_1': float((score + 1.0) * 0.5),
        'threshold': threshold,
        'decision_margin': decision_margin,
        'prediction': 'same person' if score >= threshold else 'different person',
        'confidence_interpretation': describe_similarity(score, threshold),
        'backbone': bundle.metadata['backbone'],
        'mode': bundle.metadata['mode'],
        'fusion_type': bundle.metadata.get('fusion_type', 'concat'),
        'checkpoint_format': bundle.metadata['checkpoint_format'],
        'detection_confidence_image1': float(first.detection_confidence),
        'detection_confidence_image2': float(second.detection_confidence),
    }
    return result



def compare_image_paths(
    bundle: CheckpointBundle,
    image1_path: str | Path,
    image2_path: str | Path,
    device: torch.device,
    threshold: float | None = None,
    min_detection_confidence: float = 0.5,
) -> tuple[dict, ProcessedFace, ProcessedFace]:
    image_size = int(bundle.metadata['image_size'])
    with FacePreprocessor(
        output_size=image_size,
        min_detection_confidence=min_detection_confidence,
    ) as preprocessor:
        first = preprocessor.process_path(image1_path)
        second = preprocessor.process_path(image2_path)

    if first is None:
        raise RuntimeError(f'No face detected or processed in image1: {image1_path}')
    if second is None:
        raise RuntimeError(f'No face detected or processed in image2: {image2_path}')

    result = compare_processed_faces(bundle, first, second, device=device, threshold=threshold)
    result['image1'] = str(Path(image1_path))
    result['image2'] = str(Path(image2_path))
    return result, first, second



def compare_pil_images(
    bundle: CheckpointBundle,
    image1: Image.Image,
    image2: Image.Image,
    device: torch.device,
    threshold: float | None = None,
    min_detection_confidence: float = 0.5,
) -> tuple[dict, ProcessedFace, ProcessedFace]:
    image_size = int(bundle.metadata['image_size'])
    with FacePreprocessor(
        output_size=image_size,
        min_detection_confidence=min_detection_confidence,
    ) as preprocessor:
        first = preprocessor.process_rgb(np.asarray(image1.convert('RGB')))
        second = preprocessor.process_rgb(np.asarray(image2.convert('RGB')))

    if first is None:
        raise RuntimeError('No face detected or processed in the first uploaded image.')
    if second is None:
        raise RuntimeError('No face detected or processed in the second uploaded image.')

    result = compare_processed_faces(bundle, first, second, device=device, threshold=threshold)
    return result, first, second
