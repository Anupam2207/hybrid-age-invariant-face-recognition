"""Generate a manifest CSV from an identity-organized CACD directory."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a manifest from a CACD-style folder tree.')
    parser.add_argument('--data_dir', type=str, default='data/raw/cacd')
    parser.add_argument('--output_csv', type=str, default='data/manifests/cacd_manifest.csv')
    parser.add_argument('--val_ratio', type=float, default=0.10)
    parser.add_argument('--test_ratio', type=float, default=0.10)
    parser.add_argument(
        '--split_by',
        choices=['identity', 'image'],
        default='identity',
        help='Identity split avoids train/val/test leakage across the same person.',
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()



def parse_age_from_filename(filename: str) -> int | None:
    try:
        return int(filename.split('_')[0])
    except Exception:
        return None



def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    identities = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    if not identities:
        raise RuntimeError(f'No identity folders found under {data_dir}')

    rows = []
    split_by_identity = {}

    if args.split_by == 'identity':
        shuffled = identities[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = int(round(n_total * args.test_ratio))
        n_val = int(round(n_total * args.val_ratio))
        test_set = {p.name for p in shuffled[:n_test]}
        val_set = {p.name for p in shuffled[n_test:n_test + n_val]}

        for identity_dir in identities:
            if identity_dir.name in test_set:
                split_by_identity[identity_dir.name] = 'test'
            elif identity_dir.name in val_set:
                split_by_identity[identity_dir.name] = 'val'
            else:
                split_by_identity[identity_dir.name] = 'train'

    for identity_dir in identities:
        image_files = sorted([p for p in identity_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        for image_path in image_files:
            age = parse_age_from_filename(image_path.name)
            if age is None:
                continue

            if args.split_by == 'identity':
                split = split_by_identity[identity_dir.name]
            else:
                value = rng.random()
                if value < args.test_ratio:
                    split = 'test'
                elif value < args.test_ratio + args.val_ratio:
                    split = 'val'
                else:
                    split = 'train'

            rows.append(
                {
                    'image_path': str(image_path),
                    'identity': identity_dir.name,
                    'age': age,
                    'split': split,
                }
            )

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f'Manifest created with {len(df)} samples -> {output_csv}')
    print(df['split'].value_counts().to_string())


if __name__ == '__main__':
    main()
