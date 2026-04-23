"""Generate a manifest CSV from an identity-organized CACD directory."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd

from utils.splits import (
    apply_identity_split_map,
    assert_no_identity_leakage,
    create_identity_split_map,
    save_split_summary,
    summarize_splits,
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a manifest from a CACD-style folder tree.')
    parser.add_argument('--data_dir', type=str, default='data/raw/cacd')
    parser.add_argument('--output_csv', type=str, default='data/manifests/cacd_manifest.csv')
    parser.add_argument('--summary_json', type=str, default=None)
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
    stem = Path(filename).stem
    match = re.match(r'^(\d{1,3})', stem)
    if match is not None:
        return int(match.group(1))

    match = re.search(r'age[_-]?(\d{1,3})', stem.lower())
    if match is not None:
        return int(match.group(1))
    return None



def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory not found: {data_dir}')

    identities = sorted([path for path in data_dir.iterdir() if path.is_dir()])
    if not identities:
        raise RuntimeError(f'No identity folders found under {data_dir}')

    rows = []
    for identity_dir in identities:
        image_files = sorted([path for path in identity_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
        for image_path in image_files:
            age = parse_age_from_filename(image_path.name)
            if age is None:
                continue
            rows.append(
                {
                    'image_path': str(image_path),
                    'identity': identity_dir.name,
                    'age': age,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError('No images with parseable ages were found. Please check CACD filenames.')

    if args.split_by == 'identity':
        split_map = create_identity_split_map(
            manifest,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        manifest = apply_identity_split_map(manifest, split_map)
        assert_no_identity_leakage(manifest)
    else:
        split_values = []
        for _ in range(len(manifest)):
            value = rng.random()
            if value < args.test_ratio:
                split_values.append('test')
            elif value < args.test_ratio + args.val_ratio:
                split_values.append('val')
            else:
                split_values.append('train')
        manifest['split'] = split_values

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_csv, index=False)

    summary = summarize_splits(manifest)
    if args.summary_json is not None:
        save_split_summary(summary, args.summary_json)

    print(f'Manifest created with {len(manifest)} samples -> {output_csv}')
    print('Split summary:')
    for split_name, split_info in summary.items():
        print(f'  {split_name}: {split_info}')
    if args.split_by == 'identity':
        print('Identity leakage check: PASSED')


if __name__ == '__main__':
    main()
