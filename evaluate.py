"""Evaluate a trained checkpoint on a validation or test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VerificationPairDataset, build_image_transform
from utils.checkpointing import load_checkpoint_bundle
from utils.metrics import evaluate_verification_scores
from utils.visualization import plot_roc_curve



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate hybrid face recognizer.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--processed_csv', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--train_split', type=str, default='train', help='Used to recompute geometry stats for legacy checkpoints.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--positive_pairs_per_identity', type=int, default=10)
    parser.add_argument('--negative_multiplier', type=float, default=1.0)
    parser.add_argument('--min_age_gap', type=int, default=5)
    parser.add_argument('--legacy_image_size', type=int, default=160, help='Only used for plain legacy state_dict checkpoints.')
    parser.add_argument('--output_dir', type=str, default='outputs/eval')
    return parser.parse_args()



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

    df = pd.read_csv(args.processed_csv)
    split_df = df[df['split'].astype(str) == str(args.split)].copy().reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f'No rows found for split={args.split}')

    dataset = VerificationPairDataset(
        split_df,
        transform=build_image_transform(train=False, image_size=int(bundle.metadata['image_size'])),
        geometry_stats=bundle.geometry_stats,
        min_age_gap=args.min_age_gap,
        positive_pairs_per_identity=args.positive_pairs_per_identity,
        negative_multiplier=args.negative_multiplier,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
    )

    scores = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            image1 = batch['image1'].to(device, non_blocking=True)
            geom1 = batch['geom1'].to(device, non_blocking=True)
            image2 = batch['image2'].to(device, non_blocking=True)
            geom2 = batch['geom2'].to(device, non_blocking=True)
            _, _, similarity = bundle.model.forward_pair(image1, geom1, image2, geom2)
            scores.extend(similarity.cpu().numpy().tolist())
            labels.extend(batch['label'].cpu().numpy().tolist())

    metrics = evaluate_verification_scores(np.asarray(scores), np.asarray(labels))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(metrics.fpr, metrics.tpr, metrics.auc, output_dir / 'roc_curve.png')
    summary = {
        'accuracy': metrics.accuracy,
        'auc': metrics.auc,
        'best_threshold': metrics.best_threshold,
        'far': metrics.far,
        'frr': metrics.frr,
        'eer': metrics.eer,
        'num_pairs': len(dataset),
        'checkpoint_format': bundle.metadata['checkpoint_format'],
        'backbone': bundle.metadata['backbone'],
        'mode': bundle.metadata['mode'],
        'image_size': bundle.metadata['image_size'],
    }
    with (output_dir / 'metrics.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
