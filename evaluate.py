"""Evaluate a trained checkpoint on a verification split or fixed pair protocol."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VerificationPairDataset, build_image_transform
from utils.checkpointing import load_checkpoint_bundle
from utils.metrics import evaluate_age_gap_bins, evaluate_verification_scores
from utils.runtime import build_dataloader_kwargs, resolve_device, resolve_runtime_profile
from utils.visualization import plot_age_gap_performance, plot_roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint on a verification split.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--processed_csv', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--pair_csv', type=str, default=None, help='Optional fixed pair protocol for this split.')
    parser.add_argument('--save_pairs_csv', type=str, default=None, help='Optional path to save the pair protocol actually used.')
    parser.add_argument('--train_split', type=str, default='train', help='Used only for legacy checkpoints that need geometry stats.')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--positive_pairs_per_identity', type=int, default=10)
    parser.add_argument('--negative_multiplier', type=float, default=1.0)
    parser.add_argument('--min_age_gap', type=int, default=5)
    parser.add_argument('--legacy_image_size', type=int, default=224, help='Only used for plain legacy state_dict checkpoints.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=None, help='Optional fixed threshold override.')
    parser.add_argument('--optimize_threshold_on_split', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--threshold_strategy', choices=['accuracy', 'f1', 'eer'], default='accuracy')
    parser.add_argument('--age_gap_bins', type=str, default='0,5,10,20,40,80')
    parser.add_argument('--output_dir', type=str, default='')
    return parser.parse_args()


def parse_bins(bin_string: str) -> list[float]:
    bins = [float(value.strip()) for value in bin_string.split(',') if value.strip()]
    if len(bins) < 2:
        raise ValueError('At least two age-gap bin edges are required.')
    return bins


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)


def default_output_dir(checkpoint_path: str | Path, split: str) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_root = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'checkpoints' else checkpoint_path.parent
    return checkpoint_root / f'{split}_evaluation'


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def autocast_context(device: torch.device, enabled: bool):
    if device.type == 'cuda' and enabled:
        return torch.amp.autocast('cuda', enabled=True)
    return nullcontext()


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

    runtime_profile = resolve_runtime_profile(
        device=device,
        requested_backbone=bundle.metadata['backbone'],
        image_size=int(bundle.metadata['image_size']),
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        cache_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_workers=args.num_workers,
        amp=args.amp,
        checkpoint_frequency=1,
        freeze_backbone_epochs=0,
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
        seed=args.seed,
        pairs_csv=args.pair_csv,
    )
    data_loader_kwargs = build_dataloader_kwargs(runtime_profile)
    data_loader_kwargs['worker_init_fn'] = seed_worker
    generator = torch.Generator()
    generator.manual_seed(args.seed + 99)
    loader = DataLoader(
        dataset,
        batch_size=runtime_profile.eval_batch_size,
        shuffle=False,
        drop_last=False,
        generator=generator,
        **data_loader_kwargs,
    )

    scores: list[float] = []
    labels: list[int] = []
    age_gaps: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            image1 = batch['image1'].to(device, non_blocking=(device.type == 'cuda'))
            geom1 = batch['geom1'].to(device, non_blocking=(device.type == 'cuda'))
            image2 = batch['image2'].to(device, non_blocking=(device.type == 'cuda'))
            geom2 = batch['geom2'].to(device, non_blocking=(device.type == 'cuda'))
            with autocast_context(device, runtime_profile.amp):
                _, _, similarity = bundle.model.forward_pair(image1, geom1, image2, geom2)
            scores.extend(similarity.cpu().numpy().tolist())
            labels.extend(batch['label'].cpu().numpy().tolist())
            age_gaps.extend(batch['age_gap'].cpu().numpy().tolist())

    threshold = args.threshold
    threshold_source = 'user_override'
    if threshold is None:
        if args.optimize_threshold_on_split:
            threshold = None
            threshold_source = f'optimized_on_{args.split}'
        else:
            threshold = float(bundle.metadata.get('best_threshold', 0.5))
            threshold_source = 'checkpoint_validation_threshold'

    metrics = evaluate_verification_scores(
        np.asarray(scores),
        np.asarray(labels),
        threshold=threshold,
        threshold_strategy=args.threshold_strategy,
    )
    age_gap_metrics = evaluate_age_gap_bins(
        scores=np.asarray(scores),
        labels=np.asarray(labels),
        age_gaps=np.asarray(age_gaps),
        threshold=metrics.best_threshold,
        bins=parse_bins(args.age_gap_bins),
    )

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.checkpoint, args.split)
    metrics_dir = output_dir / 'metrics'
    plots_dir = output_dir / 'plots'
    logs_dir = output_dir / 'logs'
    for path in (output_dir, metrics_dir, plots_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    pairs_output_path = Path(args.save_pairs_csv) if args.save_pairs_csv else metrics_dir / f'{args.split}_pairs.csv'
    pairs_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.pairs.to_csv(pairs_output_path, index=False)

    pair_scores = pd.DataFrame(
        {
            'score': scores,
            'label': labels,
            'age_gap': age_gaps,
            'prediction': (np.asarray(scores) >= float(metrics.best_threshold)).astype(int),
        }
    )
    pair_scores.to_csv(metrics_dir / 'pair_scores.csv', index=False)

    plot_roc_curve(metrics.fpr, metrics.tpr, metrics.auc, plots_dir / 'roc_curve.png')
    plot_age_gap_performance(age_gap_metrics, plots_dir / 'age_gap_accuracy.png', metric_key='accuracy')
    plot_age_gap_performance(age_gap_metrics, plots_dir / 'age_gap_f1.png', metric_key='f1')

    summary = {
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1': metrics.f1,
        'auc': metrics.auc,
        'threshold_used': metrics.best_threshold,
        'threshold_source': threshold_source,
        'far': metrics.far,
        'frr': metrics.frr,
        'eer': metrics.eer,
        'num_pairs': len(dataset),
        'checkpoint_format': bundle.metadata['checkpoint_format'],
        'backbone': bundle.metadata['backbone'],
        'mode': bundle.metadata['mode'],
        'fusion_type': bundle.metadata.get('fusion_type', 'concat'),
        'image_size': bundle.metadata['image_size'],
        'split': args.split,
        'pair_protocol_source': 'provided' if args.pair_csv else 'generated',
        'pair_protocol_path': str(pairs_output_path),
        'runtime_profile': runtime_profile.to_dict(),
    }
    save_json(metrics_dir / 'metrics.json', summary)
    save_json(metrics_dir / 'age_gap_metrics.json', age_gap_metrics)
    save_json(logs_dir / 'evaluation_config.json', vars(args))

    print(json.dumps(summary, indent=2))
    print('Age-gap breakdown written to:', metrics_dir / 'age_gap_metrics.json')


if __name__ == '__main__':
    main()
