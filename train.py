from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    HybridTripletDataset,
    ManifestImageDataset,
    VerificationPairDataset,
    build_image_transform,
    compute_geometry_stats,
)
from model import HybridFaceRecognizer
from utils.losses import CosineBatchHardTripletLoss
from utils.metrics import evaluate_age_gap_bins, evaluate_verification_scores
from utils.runtime import build_dataloader_kwargs, resolve_device, resolve_runtime_profile
from utils.splits import assert_no_identity_leakage
from utils.visualization import plot_age_gap_performance, plot_roc_curve, plot_training_curves



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a hybrid age-invariant face verification model.')

    parser.add_argument('--processed_csv', type=str, required=True)
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--val_split', type=str, default='val')

    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=0, help='Micro-batch size. Use 0 to auto-tune for the current hardware.')
    parser.add_argument('--eval_batch_size', type=int, default=0, help='Evaluation batch size. Use 0 for hardware-aware default.')
    parser.add_argument('--cache_batch_size', type=int, default=0, help='Batch size for cached hard-negative refresh. Use 0 for default.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=0, help='Use 0 to auto-select.')
    parser.add_argument('--image_size', type=int, default=0, help='Aligned face size. Use 0 to auto-select for your GPU.')
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--geom_hidden_dim', type=int, default=128)
    parser.add_argument('--geom_embedding_dim', type=int, default=128)
    parser.add_argument('--fusion_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.25)

    parser.add_argument('--backbone', choices=['auto', 'mobilenet_v2', 'resnet18', 'resnet50'], default='auto')
    parser.add_argument('--mode', choices=['hybrid', 'cnn_only', 'geom_only'], default='hybrid')
    parser.add_argument('--fusion_type', choices=['concat', 'attention'], default='concat')
    parser.add_argument('--attention_heads', type=int, default=4)
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--freeze_backbone_epochs', type=int, default=-1, help='Use -1 for hardware-aware default.')

    parser.add_argument('--rotation_deg', type=float, default=10.0)
    parser.add_argument('--brightness_jitter', type=float, default=0.20)
    parser.add_argument('--contrast_jitter', type=float, default=0.15)
    parser.add_argument('--saturation_jitter', type=float, default=0.10)
    parser.add_argument('--blur_prob', type=float, default=0.20)
    parser.add_argument('--allow_horizontal_flip', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--margin', type=float, default=0.25)
    parser.add_argument('--min_age_gap', type=int, default=5)
    parser.add_argument('--positive_pairs_per_identity', type=int, default=10)
    parser.add_argument('--negative_multiplier', type=float, default=1.0)
    parser.add_argument('--threshold_strategy', choices=['accuracy', 'f1', 'eer'], default='accuracy')
    parser.add_argument('--age_gap_bins', type=str, default='0,5,10,20,40,80')

    parser.add_argument('--use_batch_hard_mining', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_cached_hard_negatives', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--hard_negative_warmup_epochs', type=int, default=2)
    parser.add_argument('--hard_negative_refresh_interval', type=int, default=2)
    parser.add_argument('--hard_negative_pool_size', type=int, default=32)

    parser.add_argument('--enable_identity_head', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_age_head', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--identity_loss_weight', type=float, default=0.20)
    parser.add_argument('--age_loss_weight', type=float, default=0.05)

    parser.add_argument('--num_workers', type=int, default=-1, help='Use -1 to auto-select a safe value for your machine.')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_val_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/experiment_gtx1050ti_age_invariant')
    parser.add_argument('--checkpoint_frequency', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='Optional path to a rich checkpoint to resume training from.')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()



def parse_bins(bin_string: str) -> list[float]:
    bins = [float(value.strip()) for value in bin_string.split(',') if value.strip()]
    if len(bins) < 2:
        raise ValueError('At least two age-gap bin edges are required.')
    return bins



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2)



def maybe_unfreeze_backbone(model: HybridFaceRecognizer, epoch_index: int, freeze_backbone_epochs: int, state: dict) -> None:
    if freeze_backbone_epochs <= 0:
        return
    if state.get('backbone_unfrozen', False):
        return
    if epoch_index < freeze_backbone_epochs:
        return
    model.unfreeze_backbone()
    state['backbone_unfrozen'] = True
    print(f'Backbone unfrozen at epoch {epoch_index + 1}.')



def load_data(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.processed_csv)

    if 'split' not in df.columns:
        raise ValueError('Processed CSV must contain a split column.')

    assert_no_identity_leakage(df)

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

    return df.reset_index(drop=True), train_df.reset_index(drop=True), val_df.reset_index(drop=True)



def make_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    image_size: int,
    geometry_stats,
):
    train_transform = build_image_transform(
        train=True,
        image_size=image_size,
        allow_horizontal_flip=args.allow_horizontal_flip,
        rotation_deg=args.rotation_deg,
        brightness_jitter=args.brightness_jitter,
        contrast_jitter=args.contrast_jitter,
        saturation_jitter=args.saturation_jitter,
        blur_prob=args.blur_prob,
    )
    eval_transform = build_image_transform(train=False, image_size=image_size)

    train_dataset = HybridTripletDataset(
        train_df,
        transform=train_transform,
        geometry_stats=geometry_stats,
        min_age_gap=args.min_age_gap,
        seed=args.seed,
        hard_negative_pool_size=args.hard_negative_pool_size,
    )
    cache_dataset = None
    if args.use_cached_hard_negatives:
        cache_dataset = ManifestImageDataset(
            train_df,
            transform=eval_transform,
            geometry_stats=geometry_stats,
        )
    val_dataset = VerificationPairDataset(
        val_df,
        transform=eval_transform,
        geometry_stats=geometry_stats,
        min_age_gap=args.min_age_gap,
        positive_pairs_per_identity=args.positive_pairs_per_identity,
        negative_multiplier=args.negative_multiplier,
        seed=args.seed,
    )
    return train_dataset, cache_dataset, val_dataset



def make_dataloaders(
    train_dataset,
    cache_dataset,
    val_dataset,
    train_batch_size: int,
    eval_batch_size: int,
    cache_batch_size: int,
    dataloader_kwargs: dict,
) -> tuple[DataLoader, DataLoader | None, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs,
    )

    cache_loader = None
    if cache_dataset is not None:
        cache_loader = DataLoader(
            cache_dataset,
            batch_size=cache_batch_size,
            shuffle=False,
            drop_last=False,
            **dataloader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        **dataloader_kwargs,
    )
    return train_loader, cache_loader, val_loader



def refresh_hard_negative_cache(
    model: HybridFaceRecognizer,
    cache_loader: DataLoader,
    train_dataset: HybridTripletDataset,
    device: torch.device,
    use_amp: bool,
) -> None:
    model.eval()
    embedding_cache: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for batch in tqdm(cache_loader, desc='Refreshing hard-negative cache', leave=False):
            images = batch['image'].to(device, non_blocking=(device.type == 'cuda'))
            geometry = batch['geometry'].to(device, non_blocking=(device.type == 'cuda'))
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                embeddings = model(images, geometry)
            embeddings = embeddings.detach().cpu().numpy()
            indices = batch['index'].cpu().numpy().tolist()
            for dataset_index, embedding in zip(indices, embeddings):
                embedding_cache[int(dataset_index)] = np.asarray(embedding, dtype=np.float32)
    train_dataset.update_embedding_cache(embedding_cache)
    if device.type == 'cuda':
        torch.cuda.empty_cache()



def evaluate_model(
    model: HybridFaceRecognizer,
    loader: DataLoader,
    device: torch.device,
    threshold_strategy: str,
    age_gap_bins: list[float],
    use_amp: bool,
) -> dict:
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    age_gaps: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation', leave=False):
            image1 = batch['image1'].to(device, non_blocking=(device.type == 'cuda'))
            geom1 = batch['geom1'].to(device, non_blocking=(device.type == 'cuda'))
            image2 = batch['image2'].to(device, non_blocking=(device.type == 'cuda'))
            geom2 = batch['geom2'].to(device, non_blocking=(device.type == 'cuda'))
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == 'cuda')):
                _, _, similarity = model.forward_pair(image1, geom1, image2, geom2)
            scores.extend(similarity.detach().cpu().numpy().tolist())
            labels.extend(batch['label'].detach().cpu().numpy().tolist())
            age_gaps.extend(batch['age_gap'].detach().cpu().numpy().tolist())

    metrics = evaluate_verification_scores(
        np.asarray(scores, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        threshold=None,
        threshold_strategy=threshold_strategy,
    )
    age_gap_metrics = evaluate_age_gap_bins(
        scores=np.asarray(scores, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        age_gaps=np.asarray(age_gaps, dtype=np.float32),
        threshold=metrics.best_threshold,
        bins=age_gap_bins,
    )
    return {
        'accuracy': float(metrics.accuracy),
        'precision': float(metrics.precision),
        'recall': float(metrics.recall),
        'f1': float(metrics.f1),
        'auc': float(metrics.auc),
        'best_threshold': float(metrics.best_threshold),
        'far': float(metrics.far),
        'frr': float(metrics.frr),
        'eer': float(metrics.eer),
        'num_pairs': int(len(loader.dataset)),
        'fpr': metrics.fpr.tolist(),
        'tpr': metrics.tpr.tolist(),
        'roc_thresholds': metrics.thresholds.tolist(),
        'age_gap_metrics': age_gap_metrics,
    }



def compute_auxiliary_losses(
    anchor_out: dict,
    positive_out: dict,
    negative_out: dict,
    batch: dict,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity_loss = torch.tensor(0.0, device=batch['anchor_label'].device)
    age_loss = torch.tensor(0.0, device=batch['anchor_label'].device)

    if args.enable_identity_head and 'identity_logits' in anchor_out:
        identity_logits = torch.cat(
            [
                anchor_out['identity_logits'],
                positive_out['identity_logits'],
                negative_out['identity_logits'],
            ],
            dim=0,
        )
        identity_targets = torch.cat(
            [
                batch['anchor_label'],
                batch['positive_label'],
                batch['negative_label'],
            ],
            dim=0,
        )
        identity_loss = F.cross_entropy(identity_logits, identity_targets)

    if args.enable_age_head and 'age_prediction' in anchor_out:
        age_predictions = torch.cat(
            [
                anchor_out['age_prediction'],
                positive_out['age_prediction'],
                negative_out['age_prediction'],
            ],
            dim=0,
        )
        age_targets = torch.cat(
            [
                batch['anchor_age'],
                batch['positive_age'],
                batch['negative_age'],
            ],
            dim=0,
        )
        valid_mask = age_targets >= 0
        if bool(valid_mask.any()):
            age_loss = F.smooth_l1_loss(age_predictions[valid_mask], age_targets[valid_mask])

    return identity_loss, age_loss



def build_checkpoint_payload(
    model: HybridFaceRecognizer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    geometry_stats,
    image_size: int,
    history: dict,
    epoch_rows: list[dict],
    args: argparse.Namespace,
    runtime_profile,
    val_metrics: dict,
) -> dict:
    return {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'epoch': int(epoch),
        'model_config': model.export_config(),
        'geometry_input_dim': int(model.geometry_input_dim),
        'geometry_stats': {
            'mean': geometry_stats.mean.tolist(),
            'std': geometry_stats.std.tolist(),
        },
        'image_size': int(image_size),
        'best_threshold': float(val_metrics['best_threshold']),
        'val_metrics': {key: value for key, value in val_metrics.items() if key not in {'age_gap_metrics', 'fpr', 'tpr', 'roc_thresholds'}},
        'val_age_gap_metrics': val_metrics['age_gap_metrics'],
        'val_roc_curve': {
            'fpr': val_metrics['fpr'],
            'tpr': val_metrics['tpr'],
            'thresholds': val_metrics['roc_thresholds'],
        },
        'history': history,
        'epoch_rows': epoch_rows,
        'resolved_runtime': runtime_profile.to_dict(),
        'args': vars(args),
    }



def maybe_resume_training(args, model, optimizer, scheduler, scaler, device: torch.device):
    start_epoch = 0
    history = {
        'train_total_loss': [],
        'train_triplet_loss': [],
        'train_identity_loss': [],
        'train_age_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'best_threshold': [],
    }
    epoch_rows: list[dict] = []
    best_auc = float('-inf')

    if args.resume is None:
        return start_epoch, history, epoch_rows, best_auc

    checkpoint = torch.load(args.resume, map_location=device)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError('Resume expects a rich checkpoint created by this training script.')

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    start_epoch = int(checkpoint.get('epoch', 0))
    history = dict(checkpoint.get('history', history))
    epoch_rows = list(checkpoint.get('epoch_rows', []))
    if epoch_rows:
        best_auc = max(float(row.get('val_auc', float('-inf'))) for row in epoch_rows)
    else:
        best_auc = float(checkpoint.get('val_metrics', {}).get('auc', float('-inf')))

    print(f'Resumed training from {args.resume} at epoch {start_epoch}.')
    return start_epoch, history, epoch_rows, best_auc



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    runtime_profile = resolve_runtime_profile(
        device=device,
        requested_backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        cache_batch_size=args.cache_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        amp=args.amp,
        checkpoint_frequency=args.checkpoint_frequency,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
    )

    print(f'Using device: {runtime_profile.device} ({runtime_profile.device_name})')
    if runtime_profile.total_vram_gb > 0:
        print(f'GPU VRAM: {runtime_profile.total_vram_gb:.2f} GB')
    print('Resolved training profile:')
    print(json.dumps(runtime_profile.to_dict(), indent=2))

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'
    metrics_dir = output_dir / 'metrics'
    plot_dir = output_dir / 'plots'
    logs_dir = output_dir / 'logs'
    for path in (output_dir, checkpoint_dir, metrics_dir, plot_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    save_json(logs_dir / 'requested_args.json', vars(args))
    save_json(logs_dir / 'resolved_runtime.json', runtime_profile.to_dict())

    full_df, train_df, val_df = load_data(args)
    geometry_stats = compute_geometry_stats(train_df)
    save_json(
        logs_dir / 'geometry_stats.json',
        {
            'mean': geometry_stats.mean.tolist(),
            'std': geometry_stats.std.tolist(),
        },
    )

    train_dataset, cache_dataset, val_dataset = make_datasets(
        train_df=train_df,
        val_df=val_df,
        args=args,
        image_size=runtime_profile.image_size,
        geometry_stats=geometry_stats,
    )
    dataloader_kwargs = build_dataloader_kwargs(runtime_profile)
    train_loader, cache_loader, val_loader = make_dataloaders(
        train_dataset=train_dataset,
        cache_dataset=cache_dataset,
        val_dataset=val_dataset,
        train_batch_size=runtime_profile.batch_size,
        eval_batch_size=runtime_profile.eval_batch_size,
        cache_batch_size=runtime_profile.cache_batch_size,
        dataloader_kwargs=dataloader_kwargs,
    )

    num_identity_classes = int(train_df['identity'].nunique()) if args.enable_identity_head else None
    model = HybridFaceRecognizer(
        geometry_input_dim=train_dataset.geometry_dim,
        backbone=runtime_profile.backbone,
        mode=args.mode,
        deep_embedding_dim=args.embedding_dim,
        geom_hidden_dim=args.geom_hidden_dim,
        geom_embedding_dim=args.geom_embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        final_embedding_dim=args.embedding_dim,
        pretrained=args.pretrained,
        dropout=args.dropout,
        fusion_type=args.fusion_type,
        attention_heads=args.attention_heads,
        enable_identity_head=args.enable_identity_head,
        num_identity_classes=num_identity_classes,
        enable_age_head=args.enable_age_head,
    ).to(device)

    if runtime_profile.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        backbone_state = {'backbone_unfrozen': False}
        print(f'Backbone frozen for the first {runtime_profile.freeze_backbone_epochs} epoch(s).')
    else:
        backbone_state = {'backbone_unfrozen': True}

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.scheduler_min_lr)
    triplet_criterion = CosineBatchHardTripletLoss(margin=args.margin, use_batch_hard=args.use_batch_hard_mining)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime_profile.amp)

    start_epoch, history, epoch_rows, best_auc = maybe_resume_training(args, model, optimizer, scheduler, scaler, device)
    best_checkpoint_path = checkpoint_dir / 'best_model.pt'
    age_gap_bins = parse_bins(args.age_gap_bins)

    if len(train_loader) == 0:
        raise RuntimeError('Training loader is empty. Reduce batch size or verify the manifest content.')

    for epoch in range(start_epoch, args.epochs):
        maybe_unfreeze_backbone(model, epoch, runtime_profile.freeze_backbone_epochs, backbone_state)

        if args.use_cached_hard_negatives and cache_loader is not None and epoch >= args.hard_negative_warmup_epochs:
            if (epoch - args.hard_negative_warmup_epochs) % max(args.hard_negative_refresh_interval, 1) == 0:
                refresh_hard_negative_cache(model, cache_loader, train_dataset, device, use_amp=runtime_profile.amp)
        else:
            train_dataset.clear_embedding_cache()

        model.train()
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_identity_loss = 0.0
        total_age_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')

        for batch_index, batch in enumerate(progress, start=1):
            batch = {
                key: value.to(device, non_blocking=(device.type == 'cuda')) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=runtime_profile.amp):
                anchor_out = model.forward_with_aux(batch['anchor_image'], batch['anchor_geom'])
                positive_out = model.forward_with_aux(batch['positive_image'], batch['positive_geom'])
                negative_out = model.forward_with_aux(batch['negative_image'], batch['negative_geom'])

                triplet_loss, mining_stats = triplet_criterion(
                    anchor_embeddings=anchor_out['embedding'],
                    positive_embeddings=positive_out['embedding'],
                    negative_embeddings=negative_out['embedding'],
                    anchor_labels=batch['anchor_label'],
                    negative_labels=batch['negative_label'],
                )
                identity_loss, age_loss = compute_auxiliary_losses(
                    anchor_out=anchor_out,
                    positive_out=positive_out,
                    negative_out=negative_out,
                    batch=batch,
                    args=args,
                )

                raw_loss = triplet_loss
                if args.enable_identity_head:
                    raw_loss = raw_loss + args.identity_loss_weight * identity_loss
                if args.enable_age_head:
                    raw_loss = raw_loss + args.age_loss_weight * age_loss
                scaled_loss = raw_loss / runtime_profile.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % runtime_profile.gradient_accumulation_steps == 0 or batch_index == len(train_loader)
            )
            if should_step:
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(raw_loss.item())
            total_triplet_loss += float(triplet_loss.item())
            total_identity_loss += float(identity_loss.item())
            total_age_loss += float(age_loss.item())
            progress.set_postfix(
                total=f'{raw_loss.item():.4f}',
                triplet=f'{triplet_loss.item():.4f}',
                hard_neg=f"{mining_stats['avg_hard_negative_similarity']:.4f}",
            )

        scheduler.step()
        epoch_total_loss = total_loss / max(1, len(train_loader))
        epoch_triplet_loss = total_triplet_loss / max(1, len(train_loader))
        epoch_identity_loss = total_identity_loss / max(1, len(train_loader))
        epoch_age_loss = total_age_loss / max(1, len(train_loader))

        history['train_total_loss'].append(epoch_total_loss)
        history['train_triplet_loss'].append(epoch_triplet_loss)
        history['train_identity_loss'].append(epoch_identity_loss if args.enable_identity_head else 0.0)
        history['train_age_loss'].append(epoch_age_loss if args.enable_age_head else 0.0)

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            threshold_strategy=args.threshold_strategy,
            age_gap_bins=age_gap_bins,
            use_amp=runtime_profile.amp,
        )
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['best_threshold'].append(val_metrics['best_threshold'])

        epoch_row = {
            'epoch': epoch + 1,
            'train_total_loss': epoch_total_loss,
            'train_triplet_loss': epoch_triplet_loss,
            'train_identity_loss': epoch_identity_loss if args.enable_identity_head else 0.0,
            'train_age_loss': epoch_age_loss if args.enable_age_head else 0.0,
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics['auc'],
            'best_threshold': val_metrics['best_threshold'],
            'val_far': val_metrics['far'],
            'val_frr': val_metrics['frr'],
            'val_eer': val_metrics['eer'],
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'effective_batch_size': runtime_profile.effective_batch_size,
            'micro_batch_size': runtime_profile.batch_size,
            'gradient_accumulation_steps': runtime_profile.gradient_accumulation_steps,
        }
        epoch_rows.append(epoch_row)
        pd.DataFrame(epoch_rows).to_csv(metrics_dir / 'epoch_metrics.csv', index=False)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_total={epoch_total_loss:.4f} | "
            f"train_triplet={epoch_triplet_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"thr={val_metrics['best_threshold']:.4f}"
        )

        checkpoint_payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            geometry_stats=geometry_stats,
            image_size=runtime_profile.image_size,
            history=history,
            epoch_rows=epoch_rows,
            args=args,
            runtime_profile=runtime_profile,
            val_metrics=val_metrics,
        )

        torch.save(checkpoint_payload, checkpoint_dir / 'last_model.pt')
        save_json(metrics_dir / 'last_val_metrics.json', {key: value for key, value in val_metrics.items() if key != 'age_gap_metrics'})
        save_json(metrics_dir / 'last_val_age_gap_metrics.json', val_metrics['age_gap_metrics'])

        if (epoch + 1) % runtime_profile.checkpoint_frequency == 0:
            torch.save(checkpoint_payload, checkpoint_dir / f'epoch_{epoch + 1:03d}.pt')

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            torch.save(checkpoint_payload, best_checkpoint_path)
            save_json(metrics_dir / 'best_val_metrics.json', {key: value for key, value in val_metrics.items() if key != 'age_gap_metrics'})
            save_json(metrics_dir / 'best_val_age_gap_metrics.json', val_metrics['age_gap_metrics'])
            plot_roc_curve(
                np.asarray(val_metrics['fpr'], dtype=np.float32),
                np.asarray(val_metrics['tpr'], dtype=np.float32),
                float(val_metrics['auc']),
                plot_dir / 'best_val_roc_curve.png',
            )
            plot_age_gap_performance(val_metrics['age_gap_metrics'], plot_dir / 'best_val_age_gap_accuracy.png', metric_key='accuracy')
            plot_age_gap_performance(val_metrics['age_gap_metrics'], plot_dir / 'best_val_age_gap_f1.png', metric_key='f1')

        plot_training_curves(history, plot_dir)
        save_json(metrics_dir / 'history.json', history)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    save_json(
        logs_dir / 'run_summary.json',
        {
            'best_auc': best_auc,
            'best_checkpoint': str(best_checkpoint_path),
            'last_checkpoint': str(checkpoint_dir / 'last_model.pt'),
            'history_file': str(metrics_dir / 'history.json'),
            'geometry_stats_file': str(logs_dir / 'geometry_stats.json'),
            'backbone': runtime_profile.backbone,
            'fusion_type': args.fusion_type,
            'mode': args.mode,
            'train_rows': int(len(train_df)),
            'val_rows': int(len(val_df)),
            'micro_batch_size': runtime_profile.batch_size,
            'effective_batch_size': runtime_profile.effective_batch_size,
            'image_size': runtime_profile.image_size,
        },
    )

    print(f'Best checkpoint saved to: {best_checkpoint_path}')
    print('Training complete.')


if __name__ == '__main__':
    main()
