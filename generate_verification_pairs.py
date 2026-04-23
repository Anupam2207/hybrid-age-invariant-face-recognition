"""Generate fixed verification pair protocols for reproducible validation/test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dataset import create_verification_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate fixed verification pair CSV files from a processed manifest.')
    parser.add_argument('--processed_csv', type=str, required=True)
    parser.add_argument('--split', type=str, required=True, help='Split name to build pairs from, for example val or test.')
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--positive_pairs_per_identity', type=int, default=10)
    parser.add_argument('--negative_multiplier', type=float, default=1.0)
    parser.add_argument('--min_age_gap', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.processed_csv)
    split_df = df[df['split'].astype(str) == str(args.split)].copy().reset_index(drop=True)
    if split_df.empty:
        raise ValueError(f'No rows found for split={args.split!r}')

    pair_df = create_verification_pairs(
        split_df,
        split=None,
        positive_pairs_per_identity=args.positive_pairs_per_identity,
        negative_multiplier=args.negative_multiplier,
        min_age_gap=args.min_age_gap,
        seed=args.seed,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(output_csv, index=False)
    print(f'Saved {len(pair_df)} verification pairs to {output_csv}')


if __name__ == '__main__':
    main()
