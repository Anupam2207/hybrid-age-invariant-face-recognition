"""Dataset definitions for the hybrid age-invariant face recognition project."""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class GeometryStats:
    mean: np.ndarray
    std: np.ndarray


def build_image_transform(
    train: bool = True,
    image_size: int = 224,
    allow_horizontal_flip: bool = False,
) -> transforms.Compose:
    """Create image transforms for aligned face images.

    Important:
        Geometry features are precomputed offline. A horizontal image flip would
        invalidate left/right-sensitive geometry features unless the geometry is
        flipped as well. Therefore horizontal flip is disabled by default.
    """

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        if allow_horizontal_flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def _load_dataframe(data: str | Path | pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.read_csv(data)
    if split is not None and 'split' in df.columns:
        df = df[df['split'].astype(str) == str(split)].copy()
    df = df.reset_index(drop=True)
    return df


def _geometry_columns(df: pd.DataFrame) -> List[str]:
    columns = [column for column in df.columns if column.startswith('g') and column[1:].isdigit()]
    columns = sorted(columns, key=lambda c: int(c[1:]))
    if not columns:
        raise ValueError('No geometric feature columns found. Expected columns like g0, g1, g2, ...')
    return columns


def compute_geometry_stats(data: str | Path | pd.DataFrame, split: str | None = None) -> GeometryStats:
    """Compute z-score statistics for geometric features from the training split."""

    df = _load_dataframe(data, split=split)
    geom_cols = _geometry_columns(df)
    values = df[geom_cols].to_numpy(dtype=np.float32)
    return GeometryStats(mean=values.mean(axis=0), std=values.std(axis=0) + 1e-6)



def create_verification_pairs(
    data: str | Path | pd.DataFrame,
    split: str | None = None,
    positive_pairs_per_identity: int = 10,
    negative_multiplier: float = 1.0,
    min_age_gap: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic positive and negative pairs for face verification.

    Positive pairs prioritize same-identity pairs with larger age gaps.
    Negative pairs are sampled uniformly from different identities.
    """

    rng = random.Random(seed)
    df = _load_dataframe(data, split=split)
    grouped = df.groupby('identity').groups
    identities = list(grouped.keys())
    if len(identities) < 2:
        raise ValueError('Verification requires at least two identities to form negative pairs.')

    positive_records: List[dict] = []
    for identity, indices in grouped.items():
        idxs = list(indices)
        if len(idxs) < 2:
            continue
        candidate_pairs = list(combinations(idxs, 2))
        if 'age' in df.columns:
            candidate_pairs.sort(
                key=lambda pair: abs(float(df.iloc[pair[0]]['age']) - float(df.iloc[pair[1]]['age'])),
                reverse=True,
            )
            age_filtered = [
                pair
                for pair in candidate_pairs
                if abs(float(df.iloc[pair[0]]['age']) - float(df.iloc[pair[1]]['age'])) >= min_age_gap
            ]
            if age_filtered:
                candidate_pairs = age_filtered

        selected = candidate_pairs[:positive_pairs_per_identity]
        for idx1, idx2 in selected:
            positive_records.append({'idx1': idx1, 'idx2': idx2, 'label': 1})

    if not positive_records:
        raise ValueError('Verification requires at least one positive pair from an identity with two or more images.')

    negative_target = max(1, int(len(positive_records) * negative_multiplier))
    negative_records: List[dict] = []
    while len(negative_records) < negative_target and len(identities) >= 2:
        id1, id2 = rng.sample(identities, 2)
        idx1 = rng.choice(list(grouped[id1]))
        idx2 = rng.choice(list(grouped[id2]))
        negative_records.append({'idx1': idx1, 'idx2': idx2, 'label': 0})

    pair_df = pd.DataFrame(positive_records + negative_records)
    pair_df = pair_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pair_df


class _BaseHybridDataset(Dataset):
    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[transforms.Compose] = None,
        geometry_stats: Optional[GeometryStats] = None,
    ) -> None:
        self.df = _load_dataframe(data, split=split)
        self.transform = transform if transform is not None else build_image_transform(train=False)
        self.geom_cols = _geometry_columns(self.df)
        self.geometry_stats = geometry_stats
        self.identity_to_label = {
            identity: idx for idx, identity in enumerate(sorted(map(str, self.df['identity'].unique().tolist())))
        }

    def _load_row(self, row: pd.Series) -> Dict[str, torch.Tensor | int | str | float]:
        image_path = row['aligned_path'] if 'aligned_path' in row and pd.notna(row['aligned_path']) else row['image_path']
        image_path = Path(str(image_path))
        if not image_path.exists():
            raise FileNotFoundError(f'Image path not found in manifest: {image_path}')

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        geom = row[self.geom_cols].to_numpy(dtype=np.float32)
        if self.geometry_stats is not None:
            geom = (geom - self.geometry_stats.mean) / self.geometry_stats.std
        geom_tensor = torch.tensor(geom, dtype=torch.float32)

        identity = str(row['identity'])
        age = float(row['age']) if 'age' in row and not pd.isna(row['age']) else -1.0

        return {
            'image': image_tensor,
            'geometry': geom_tensor,
            'identity': self.identity_to_label[identity],
            'identity_str': identity,
            'age': age,
            'path': str(image_path),
        }

    @property
    def geometry_dim(self) -> int:
        return len(self.geom_cols)


class HybridTripletDataset(_BaseHybridDataset):
    """Triplet dataset for age-invariant metric learning.

    Anchor and positive samples come from the same identity, ideally with a large
    age gap. Negative samples come from a different identity.
    """

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[transforms.Compose] = None,
        geometry_stats: Optional[GeometryStats] = None,
        min_age_gap: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats)
        self.min_age_gap = int(min_age_gap)
        self.rng = random.Random(seed)
        self.grouped_indices = {identity: list(indices) for identity, indices in self.df.groupby('identity').groups.items()}
        self.valid_identities = [identity for identity, idxs in self.grouped_indices.items() if len(idxs) >= 2]
        self.valid_anchor_indices = [idx for identity in self.valid_identities for idx in self.grouped_indices[identity]]
        if not self.valid_anchor_indices:
            raise ValueError('Triplet dataset requires at least one identity with two or more images.')
        if len(self.grouped_indices) < 2:
            raise ValueError('Triplet training requires at least two distinct identities.')

    def __len__(self) -> int:
        return len(self.valid_anchor_indices)

    def _sample_positive_index(self, anchor_index: int, identity: str) -> int:
        candidates = [idx for idx in self.grouped_indices[identity] if idx != anchor_index]
        if not candidates:
            raise RuntimeError('No positive sample available for identity with single image.')

        if 'age' not in self.df.columns:
            return self.rng.choice(candidates)

        anchor_age = float(self.df.iloc[anchor_index]['age'])
        age_scored = [
            (idx, abs(float(self.df.iloc[idx]['age']) - anchor_age))
            for idx in candidates
        ]
        age_filtered = [item for item in age_scored if item[1] >= self.min_age_gap]
        if age_filtered:
            age_scored = age_filtered
        age_scored.sort(key=lambda item: item[1], reverse=True)

        top_k = age_scored[: min(3, len(age_scored))]
        return self.rng.choice(top_k)[0]

    def _sample_negative_index(self, identity: str) -> int:
        negative_identity = self.rng.choice([id_ for id_ in self.grouped_indices.keys() if id_ != identity])
        return self.rng.choice(self.grouped_indices[negative_identity])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        anchor_idx = self.valid_anchor_indices[index]
        anchor_row = self.df.iloc[anchor_idx]
        identity = str(anchor_row['identity'])
        positive_idx = self._sample_positive_index(anchor_idx, identity)
        negative_idx = self._sample_negative_index(identity)

        anchor = self._load_row(anchor_row)
        positive = self._load_row(self.df.iloc[positive_idx])
        negative = self._load_row(self.df.iloc[negative_idx])

        return {
            'anchor_image': anchor['image'],
            'anchor_geom': anchor['geometry'],
            'positive_image': positive['image'],
            'positive_geom': positive['geometry'],
            'negative_image': negative['image'],
            'negative_geom': negative['geometry'],
            'anchor_label': torch.tensor(anchor['identity'], dtype=torch.long),
        }


class VerificationPairDataset(_BaseHybridDataset):
    """Pair dataset for verification metrics such as ROC, FAR, and FRR."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[transforms.Compose] = None,
        geometry_stats: Optional[GeometryStats] = None,
        min_age_gap: int = 5,
        positive_pairs_per_identity: int = 10,
        negative_multiplier: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats)
        self.pairs = create_verification_pairs(
            self.df,
            split=None,
            positive_pairs_per_identity=positive_pairs_per_identity,
            negative_multiplier=negative_multiplier,
            min_age_gap=min_age_gap,
            seed=seed,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs.iloc[index]
        first = self._load_row(self.df.iloc[int(pair['idx1'])])
        second = self._load_row(self.df.iloc[int(pair['idx2'])])
        return {
            'image1': first['image'],
            'geom1': first['geometry'],
            'image2': second['image'],
            'geom2': second['geometry'],
            'label': torch.tensor(int(pair['label']), dtype=torch.long),
        }


class ManifestImageDataset(_BaseHybridDataset):
    """Simple dataset over unique preprocessed images for embedding extraction."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[transforms.Compose] = None,
        geometry_stats: Optional[GeometryStats] = None,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self._load_row(self.df.iloc[index])
        return {
            'image': row['image'],
            'geometry': row['geometry'],
            'label': torch.tensor(int(row['identity']), dtype=torch.long),
            'age': torch.tensor(float(row['age']), dtype=torch.float32),
        }
