"""Run verification on two images using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import build_image_transform
from utils.checkpointing import load_checkpoint_bundle
from utils.preprocessing import FacePreprocessor



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare two faces with a trained model.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image1', type=str, required=True)
    parser.add_argument('--image2', type=str, required=True)
    parser.add_argument('--processed_csv', type=str, default=None, help='Optional; used to recompute geometry stats for legacy checkpoints.')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--legacy_image_size', type=int, default=160)
    parser.add_argument('--threshold', type=float, default=None, help='Override the checkpoint threshold.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5)
    parser.add_argument('--output_json', type=str, default=None)
    return parser.parse_args()



def processed_face_to_tensors(processed_face, transform, geometry_stats, device: torch.device):
    image = Image.fromarray(processed_face.aligned_rgb)
    image_tensor = transform(image).unsqueeze(0).to(device)

    geometry = processed_face.geometry_features.astype(np.float32)
    if geometry_stats is not None:
        geometry = (geometry - geometry_stats.mean) / geometry_stats.std
    geometry_tensor = torch.tensor(geometry, dtype=torch.float32, device=device).unsqueeze(0)
    return image_tensor, geometry_tensor



def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    bundle = load_checkpoint_bundle(
        checkpoint_path=args.checkpoint,
        device=device,
        processed_csv=args.processed_csv,
        train_split=args.train_split,
        legacy_image_size=args.legacy_image_size,
    )
    image_size = int(bundle.metadata['image_size'])
    threshold = float(args.threshold if args.threshold is not None else bundle.metadata.get('best_threshold', 0.5))
    transform = build_image_transform(train=False, image_size=image_size)

    with FacePreprocessor(
        output_size=image_size,
        min_detection_confidence=args.min_detection_confidence,
    ) as preprocessor:
        first = preprocessor.process_path(args.image1)
        second = preprocessor.process_path(args.image2)

    if first is None:
        raise RuntimeError(f'No face detected or processed in image1: {args.image1}')
    if second is None:
        raise RuntimeError(f'No face detected or processed in image2: {args.image2}')

    image1_tensor, geom1_tensor = processed_face_to_tensors(first, transform, bundle.geometry_stats, device)
    image2_tensor, geom2_tensor = processed_face_to_tensors(second, transform, bundle.geometry_stats, device)

    with torch.no_grad():
        _, _, similarity = bundle.model.forward_pair(image1_tensor, geom1_tensor, image2_tensor, geom2_tensor)
        score = float(similarity.item())

    result = {
        'image1': str(Path(args.image1)),
        'image2': str(Path(args.image2)),
        'similarity_score': score,
        'threshold': threshold,
        'decision': 'same person' if score >= threshold else 'different person',
        'backbone': bundle.metadata['backbone'],
        'mode': bundle.metadata['mode'],
        'checkpoint_format': bundle.metadata['checkpoint_format'],
    }

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
