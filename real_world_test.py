"""Evaluate the model on unseen external images using a pair CSV.

Expected CSV columns:
    image1,image2
Optional column:
    label   # 1 for same person, 0 for different person
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from utils.checkpointing import load_checkpoint_bundle
from utils.inference_helpers import compare_image_paths
from utils.metrics import evaluate_verification_scores



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run real-world testing on unseen image pairs.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--pairs_csv', type=str, required=True)
    parser.add_argument('--processed_csv', type=str, default=None, help='Optional; used for legacy checkpoints.')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--legacy_image_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5)
    parser.add_argument('--output_csv', type=str, default='outputs/unseen_results.csv')
    parser.add_argument('--summary_json', type=str, default='outputs/unseen_summary.json')
    return parser.parse_args()



def parse_label(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'same', 'same person', 'yes'}:
            return 1
        if lowered in {'0', 'false', 'different', 'different person', 'no'}:
            return 0
    try:
        return int(value)
    except Exception:
        return None



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

    pairs_df = pd.read_csv(args.pairs_csv)
    if 'image1' not in pairs_df.columns or 'image2' not in pairs_df.columns:
        raise ValueError('pairs_csv must contain image1 and image2 columns.')

    result_rows = []
    for row in tqdm(pairs_df.to_dict(orient='records'), total=len(pairs_df), desc='Unseen testing'):
        try:
            result, _, _ = compare_image_paths(
                bundle=bundle,
                image1_path=row['image1'],
                image2_path=row['image2'],
                device=device,
                threshold=args.threshold,
                min_detection_confidence=args.min_detection_confidence,
            )
            result['error'] = None
        except Exception as exc:
            result = {
                'image1': str(row['image1']),
                'image2': str(row['image2']),
                'similarity_score': None,
                'score_0_to_1': None,
                'threshold': float(args.threshold) if args.threshold is not None else float(bundle.metadata.get('best_threshold', 0.5)),
                'prediction': None,
                'error': str(exc),
            }

        if 'label' in row:
            result['label'] = parse_label(row['label'])
        result_rows.append(result)

    output_df = pd.DataFrame(result_rows)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    summary = {
        'num_pairs': int(len(output_df)),
        'successful_pairs': int(output_df['error'].isna().sum() if 'error' in output_df.columns else len(output_df)),
        'output_csv': str(output_path),
    }

    if 'label' in output_df.columns:
        valid = output_df[(output_df['error'].isna()) & (output_df['label'].notna()) & (output_df['similarity_score'].notna())].copy()
        if not valid.empty:
            metrics = evaluate_verification_scores(
                scores=valid['similarity_score'].astype(float).to_numpy(),
                labels=valid['label'].astype(int).to_numpy(),
                threshold=float(args.threshold) if args.threshold is not None else float(bundle.metadata.get('best_threshold', 0.5)),
            )
            summary.update(
                {
                    'accuracy': metrics.accuracy,
                    'precision': metrics.precision,
                    'recall': metrics.recall,
                    'f1': metrics.f1,
                    'auc': metrics.auc,
                    'far': metrics.far,
                    'frr': metrics.frr,
                    'threshold_used': metrics.best_threshold,
                }
            )

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
