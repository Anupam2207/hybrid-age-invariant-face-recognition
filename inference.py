"""Run verification on two images using a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from utils.checkpointing import load_checkpoint_bundle
from utils.inference_helpers import compare_image_paths
from utils.runtime import resolve_device



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare two faces with a trained model.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image1', type=str, required=True)
    parser.add_argument('--image2', type=str, required=True)
    parser.add_argument('--processed_csv', type=str, default=None, help='Optional; used to recompute geometry stats for legacy checkpoints.')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--legacy_image_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=None, help='Override the checkpoint threshold.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--save_aligned_dir', type=str, default=None, help='Optional directory to save aligned crops used for inference.')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    bundle = load_checkpoint_bundle(
        checkpoint_path=args.checkpoint,
        device=device,
        processed_csv=args.processed_csv,
        train_split=args.train_split,
        legacy_image_size=args.legacy_image_size,
    )

    result, first, second = compare_image_paths(
        bundle=bundle,
        image1_path=args.image1,
        image2_path=args.image2,
        device=device,
        threshold=args.threshold,
        min_detection_confidence=args.min_detection_confidence,
    )

    if args.save_aligned_dir is not None:
        save_dir = Path(args.save_aligned_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(first.aligned_rgb).save(save_dir / 'image1_aligned.jpg', quality=95)
        Image.fromarray(second.aligned_rgb).save(save_dir / 'image2_aligned.jpg', quality=95)
        result['aligned_image1'] = str(save_dir / 'image1_aligned.jpg')
        result['aligned_image2'] = str(save_dir / 'image2_aligned.jpg')

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            json.dump(result, file, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
