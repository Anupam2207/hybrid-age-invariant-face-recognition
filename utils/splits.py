"""Identity-aware split helpers for face verification datasets.

These utilities keep all images of the same identity inside exactly one split.
That prevents identity leakage between train/validation/test and produces a much
more realistic estimate of open-set face verification performance.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import pandas as pd


REQUIRED_COLUMNS = {'identity'}


def _validate_dataframe(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns for splitting: {sorted(missing)}')


def create_identity_split_map(
    df: pd.DataFrame,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each identity to exactly one split.

    A greedy image-count balancing strategy is used because face datasets often
    contain very uneven numbers of images per identity.
    """

    _validate_dataframe(df)
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError('val_ratio must be in [0, 1).')
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError('test_ratio must be in [0, 1).')
    if val_ratio + test_ratio >= 1.0:
        raise ValueError('val_ratio + test_ratio must be < 1.0.')

    identity_frame = (
        df.groupby('identity', as_index=False)
        .agg(num_images=('identity', 'size'))
        .sort_values(['num_images', 'identity'], ascending=[False, True])
        .reset_index(drop=True)
    )

    rng = random.Random(seed)
    identities = identity_frame.to_dict(orient='records')
    rng.shuffle(identities)
    identities.sort(key=lambda row: row['num_images'], reverse=True)

    total_images = float(identity_frame['num_images'].sum())
    target_counts = {
        'train': max(total_images * (1.0 - val_ratio - test_ratio), 1.0),
        'val': max(total_images * val_ratio, 0.0),
        'test': max(total_images * test_ratio, 0.0),
    }

    current_counts = {'train': 0.0, 'val': 0.0, 'test': 0.0}
    current_identities = {'train': 0, 'val': 0, 'test': 0}
    split_map: Dict[str, str] = {}

    non_zero_splits = [split for split, target in target_counts.items() if target > 0.0]

    for position, row in enumerate(identities):
        identity = str(row['identity'])
        num_images = float(row['num_images'])

        remaining = len(identities) - position
        empty_required = [split for split in non_zero_splits if current_identities[split] == 0]
        if remaining == len(empty_required) and empty_required:
            chosen_split = empty_required[0]
        else:
            deficits = {}
            for split in non_zero_splits:
                target = target_counts[split]
                current = current_counts[split]
                deficits[split] = target - current

            chosen_split = max(
                non_zero_splits,
                key=lambda split: (
                    deficits[split],
                    -current_counts[split],
                    -current_identities[split],
                ),
            )

            if deficits[chosen_split] <= 0:
                chosen_split = min(
                    non_zero_splits,
                    key=lambda split: current_counts[split] / max(target_counts[split], 1.0),
                )

        split_map[identity] = chosen_split
        current_counts[chosen_split] += num_images
        current_identities[chosen_split] += 1

    return split_map



def apply_identity_split_map(df: pd.DataFrame, split_map: Dict[str, str]) -> pd.DataFrame:
    """Attach the split column to a manifest dataframe."""

    _validate_dataframe(df)
    output = df.copy()
    output['split'] = output['identity'].astype(str).map(split_map)
    if output['split'].isna().any():
        missing = output.loc[output['split'].isna(), 'identity'].astype(str).unique().tolist()
        raise ValueError(f'Some identities were not assigned a split: {missing[:10]}')
    return output



def assert_no_identity_leakage(df: pd.DataFrame, split_col: str = 'split') -> None:
    """Raise an error if any identity appears in more than one split."""

    _validate_dataframe(df)
    if split_col not in df.columns:
        raise ValueError(f'{split_col!r} column is required to check leakage.')

    duplicates = (
        df.groupby('identity')[split_col]
        .nunique()
        .reset_index(name='num_splits')
    )
    leaking = duplicates[duplicates['num_splits'] > 1]
    if not leaking.empty:
        examples = leaking['identity'].astype(str).tolist()[:10]
        raise RuntimeError(
            'Identity leakage detected. The following identities appear in multiple splits: '
            f'{examples}'
        )



def summarize_splits(df: pd.DataFrame, split_col: str = 'split') -> Dict[str, dict]:
    """Summarize the number of images and identities per split."""

    _validate_dataframe(df)
    if split_col not in df.columns:
        raise ValueError(f'{split_col!r} column is required to summarize splits.')

    summary: Dict[str, dict] = {}
    for split_name, split_df in df.groupby(split_col):
        split_df = split_df.copy()
        summary[str(split_name)] = {
            'num_images': int(len(split_df)),
            'num_identities': int(split_df['identity'].nunique()),
            'avg_images_per_identity': float(len(split_df) / max(split_df['identity'].nunique(), 1)),
        }
        if 'age' in split_df.columns and not split_df['age'].isna().all():
            summary[str(split_name)]['min_age'] = float(split_df['age'].min())
            summary[str(split_name)]['max_age'] = float(split_df['age'].max())
            summary[str(split_name)]['mean_age'] = float(split_df['age'].mean())
    return summary



def save_split_summary(summary: Dict[str, dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
