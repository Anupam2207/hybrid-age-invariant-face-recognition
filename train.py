from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    HybridTripletDataset,
    VerificationPairDataset,
    build_image_transform,
    compute_geometry_stats,
)
from model import HybridFaceRecognizer
from utils.metrics import evaluate_verification_scores
from utils.visualization import plot_training_curves



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a hybrid age-invariant face verification model.')

    parser.add_argument('--processed_csv', type=str, required=True)
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--geom_hidden_dim', type=int, default=64)
    parser.add_argument('--geom_embedding_dim', type=int, default=64)
    parser.add_argument('--fusion_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--backbone', choices=['mobilenet_v2', 'resnet18'], default='mobilenet_v2')
    parser.add_argument('--mode', choices=['hybrid', 'cnn_only', 'geom_only'], default='hybrid')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow_horizontal_flip', action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6)

    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--min_age_gap', type=int, default=5)
    parser.add_argument('--positive_pairs_per_identity', type=int, default=10)
    parser.add_argument('--negative_multiplier', type=float, default=1.0)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/experiment_01')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def load_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.processed_csv)

    if 'split' not in df.columns:
        raise ValueError('Processed CSV must contain a split column.')

    train_df = df[df['split'].astype(str) == str(args.train_split)].copy()
    val_df = df[df['split'].astype(str) == str(args.val_split)].copy()

    if train_df.empty:
        raise ValueError(f'No rows found for train_split={args.train_split!r}')
    if val_df.empty:
        raise ValueError(f'No rows found for val_split={args.val_split!r}')

    if args.max_train_samples is not None and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(n=args.max_train_samples, random_state=args.seed)
    if args.max_val_samples is not None and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(n=args.max_val_samples, random_state=args.seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)



def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    geometry_stats,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    train_dataset = HybridTripletDataset(
        train_df,
        transform=build_image_transform(
            train=True,
            image_size=args.image_size,
            allow_horizontal_flip=args.allow_horizontal_flip,
        ),
        geometry_stats=geometry_stats,
        min_age_gap=args.min_age_gap,
        seed=args.seed,
    )

    val_dataset = VerificationPairDataset(
        val_df,
        transform=build_image_transform(train=False, image_size=args.image_size),
        geometry_stats=geometry_stats,
        min_age_gap=args.min_age_gap,
        positive_pairs_per_identity=args.positive_pairs_per_identity,
        negative_multiplier=args.negative_multiplier,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset.geometry_dim



def evaluate_model(
    model: HybridFaceRecognizer,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation', leave=False):
            image1 = batch['image1'].to(device, non_blocking=True)
            geom1 = batch['geom1'].to(device, non_blocking=True)
            image2 = batch['image2'].to(device, non_blocking=True)
            geom2 = batch['geom2'].to(device, non_blocking=True)
            _, _, similarity = model.forward_pair(image1, geom1, image2, geom2)
            scores.extend(similarity.detach().cpu().numpy().tolist())
            labels.extend(batch['label'].detach().cpu().numpy().tolist())

    metrics = evaluate_verification_scores(np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.int32))
    return {
        'accuracy': float(metrics.accuracy),
        'auc': float(metrics.auc),
        'best_threshold': float(metrics.best_threshold),
        'far': float(metrics.far),
        'frr': float(metrics.frr),
        'eer': float(metrics.eer),
        'num_pairs': int(len(loader.dataset)),
    }



def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    pin_memory = device.type == 'cuda'
    print(f'Using device: {device}')

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    plot_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_data(args)
    geometry_stats = compute_geometry_stats(train_df)
    save_json(
        output_dir / 'geometry_stats.json',
        {
            'mean': geometry_stats.mean.tolist(),
            'std': geometry_stats.std.tolist(),
        },
    )

    train_loader, val_loader, geom_dim = make_dataloaders(
        train_df=train_df,
        val_df=val_df,
        args=args,
        geometry_stats=geometry_stats,
        pin_memory=pin_memory,
    )

    model = HybridFaceRecognizer(
        geometry_input_dim=geom_dim,
        backbone=args.backbone,
        mode=args.mode,
        deep_embedding_dim=args.embedding_dim,
        geom_hidden_dim=args.geom_hidden_dim,
        geom_embedding_dim=args.geom_embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        final_embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.scheduler_min_lr)
    criterion = nn.TripletMarginLoss(margin=args.margin)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    history: dict[str, list] = {
        'train_loss': [],
        'val_accuracy': [],
        'val_auc': [],
        'best_threshold': [],
    }

    best_auc = float('-inf')
    best_checkpoint_path = checkpoint_dir / 'best_model.pt'

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch in progress:
            anchor = batch['anchor_image'].to(device, non_blocking=True)
            anchor_g = batch['anchor_geom'].to(device, non_blocking=True)
            pos = batch['positive_image'].to(device, non_blocking=True)
            pos_g = batch['positive_geom'].to(device, non_blocking=True)
            neg = batch['negative_image'].to(device, non_blocking=True)
            neg_g = batch['negative_geom'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                a = model(anchor, anchor_g)
                p = model(pos, pos_g)
                n = model(neg, neg_g)
                loss = criterion(a, p, n)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            progress.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()
        epoch_loss = total_loss / max(1, len(train_loader))
        history['train_loss'].append(epoch_loss)

        val_metrics = evaluate_model(model, val_loader, device)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['best_threshold'].append(val_metrics['best_threshold'])

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={epoch_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"thr={val_metrics['best_threshold']:.4f}"
        )

        checkpoint_payload = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch + 1,
            'backbone': args.backbone,
            'mode': args.mode,
            'embedding_dim': args.embedding_dim,
            'deep_embedding_dim': args.embedding_dim,
            'geom_hidden_dim': args.geom_hidden_dim,
            'geom_embedding_dim': args.geom_embedding_dim,
            'fusion_hidden_dim': args.fusion_hidden_dim,
            'geometry_input_dim': geom_dim,
            'geometry_stats': {
                'mean': geometry_stats.mean.tolist(),
                'std': geometry_stats.std.tolist(),
            },
            'image_size': args.image_size,
            'dropout': args.dropout,
            'best_threshold': val_metrics['best_threshold'],
            'val_metrics': val_metrics,
            'history': history,
            'args': vars(args),
        }

        torch.save(checkpoint_payload, checkpoint_dir / 'last_model.pt')
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(checkpoint_payload, best_checkpoint_path)

    plot_training_curves(history, plot_dir)
    save_json(output_dir / 'history.json', history)
    save_json(
        output_dir / 'run_summary.json',
        {
            'best_auc': best_auc,
            'best_checkpoint': str(best_checkpoint_path),
            'history_file': str(output_dir / 'history.json'),
            'geometry_stats_file': str(output_dir / 'geometry_stats.json'),
        },
    )

    print(f'Best checkpoint saved to: {best_checkpoint_path}')
    print('Training complete.')


if __name__ == '__main__':
    main()
