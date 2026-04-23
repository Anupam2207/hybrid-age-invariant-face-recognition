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
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

from utils.geometry import flip_geometry_features


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_FILL = tuple(int(channel * 255) for channel in IMAGENET_MEAN)


@dataclass
class GeometryStats:
    mean: np.ndarray
    std: np.ndarray


class HybridTransform:
    """Coupled image transform that can also update geometry features.

    Only horizontal flip changes geometry in the current setup. Other image-only
    augmentations are intentionally small so the geometry branch remains a stable
    complementary signal while the image branch becomes more robust.
    """

    def __init__(
        self,
        train: bool = True,
        image_size: int = 224,
        allow_horizontal_flip: bool = True,
        rotation_deg: float = 10.0,
        brightness_jitter: float = 0.20,
        contrast_jitter: float = 0.15,
        saturation_jitter: float = 0.10,
        blur_prob: float = 0.20,
        blur_radius_min: float = 0.10,
        blur_radius_max: float = 1.25,
    ) -> None:
        self.train = bool(train)
        self.image_size = int(image_size)
        self.allow_horizontal_flip = bool(allow_horizontal_flip)
        self.rotation_deg = float(rotation_deg)
        self.blur_prob = float(blur_prob)
        self.blur_radius_min = float(blur_radius_min)
        self.blur_radius_max = float(blur_radius_max)
        self.color_jitter = transforms.ColorJitter(
            brightness=brightness_jitter,
            contrast=contrast_jitter,
            saturation=saturation_jitter,
        )

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)

    def __call__(self, image: Image.Image, geometry: np.ndarray | None = None, rng: random.Random | None = None):
        rng = rng or random
        image = self._resize(image)
        geom = None if geometry is None else np.asarray(geometry, dtype=np.float32).copy()

        if self.train:
            if self.allow_horizontal_flip and rng.random() < 0.5:
                image = TF.hflip(image)
                if geom is not None:
                    geom = flip_geometry_features(geom)

            if self.rotation_deg > 0.0:
                angle = rng.uniform(-self.rotation_deg, self.rotation_deg)
                image = TF.rotate(
                    image,
                    angle,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=IMAGENET_FILL,
                )

            image = self.color_jitter(image)

            if self.blur_prob > 0.0 and rng.random() < self.blur_prob:
                radius = rng.uniform(self.blur_radius_min, self.blur_radius_max)
                image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if geom is None:
            return image_tensor
        return image_tensor, geom



def build_image_transform(
    train: bool = True,
    image_size: int = 224,
    allow_horizontal_flip: bool = True,
    rotation_deg: float = 10.0,
    brightness_jitter: float = 0.20,
    contrast_jitter: float = 0.15,
    saturation_jitter: float = 0.10,
    blur_prob: float = 0.20,
) -> HybridTransform:
    """Create coupled image/geometry transforms for aligned face images."""

    return HybridTransform(
        train=train,
        image_size=image_size,
        allow_horizontal_flip=allow_horizontal_flip,
        rotation_deg=rotation_deg,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
        saturation_jitter=saturation_jitter,
        blur_prob=blur_prob,
    )



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
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
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
            age1 = float(df.iloc[idx1]['age']) if 'age' in df.columns else -1.0
            age2 = float(df.iloc[idx2]['age']) if 'age' in df.columns else -1.0
            positive_records.append(
                {
                    'idx1': idx1,
                    'idx2': idx2,
                    'label': 1,
                    'age1': age1,
                    'age2': age2,
                    'age_gap': abs(age1 - age2) if age1 >= 0 and age2 >= 0 else -1.0,
                }
            )

    if not positive_records:
        raise ValueError('Verification requires at least one positive pair from an identity with two or more images.')

    negative_target = max(1, int(len(positive_records) * negative_multiplier))
    negative_records: List[dict] = []
    used_negative_pairs: set[tuple[int, int]] = set()
    while len(negative_records) < negative_target and len(identities) >= 2:
        id1, id2 = rng.sample(identities, 2)
        idx1 = rng.choice(list(grouped[id1]))
        idx2 = rng.choice(list(grouped[id2]))
        pair_key = tuple(sorted((int(idx1), int(idx2))))
        if pair_key in used_negative_pairs:
            continue
        used_negative_pairs.add(pair_key)
        age1 = float(df.iloc[idx1]['age']) if 'age' in df.columns else -1.0
        age2 = float(df.iloc[idx2]['age']) if 'age' in df.columns else -1.0
        negative_records.append(
            {
                'idx1': idx1,
                'idx2': idx2,
                'label': 0,
                'age1': age1,
                'age2': age2,
                'age_gap': abs(age1 - age2) if age1 >= 0 and age2 >= 0 else -1.0,
            }
        )

    pair_df = pd.DataFrame(positive_records + negative_records)
    pair_df = pair_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return pair_df


class _BaseHybridDataset(Dataset):
    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[HybridTransform] = None,
        geometry_stats: Optional[GeometryStats] = None,
        seed: int = 42,
    ) -> None:
        self.df = _load_dataframe(data, split=split)
        self.transform = transform if transform is not None else build_image_transform(train=False)
        self.geom_cols = _geometry_columns(self.df)
        self.geometry_stats = geometry_stats
        self.base_seed = int(seed)
        self.epoch = 0
        self.identity_to_label = {
            identity: idx for idx, identity in enumerate(sorted(map(str, self.df['identity'].unique().tolist())))
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_rng(self, index: int) -> random.Random:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else int(worker_info.id)
        seed = self.base_seed + self.epoch * 1_000_003 + worker_id * 10_007 + int(index)
        return random.Random(seed)

    def _load_row(self, row: pd.Series, row_index: int, rng: random.Random | None = None) -> Dict[str, torch.Tensor | int | str | float]:
        image_path = row['aligned_path'] if 'aligned_path' in row and pd.notna(row['aligned_path']) else row['image_path']
        image_path = Path(str(image_path))
        if not image_path.exists():
            raise FileNotFoundError(f'Image path not found in manifest: {image_path}')

        image = Image.open(image_path).convert('RGB')
        geom = row[self.geom_cols].to_numpy(dtype=np.float32)
        geom = np.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)

        transformed = self.transform(image, geom, rng=rng)
        if isinstance(transformed, tuple):
            image_tensor, geom = transformed
        else:
            image_tensor = transformed

        if self.geometry_stats is not None:
            geom = (geom - self.geometry_stats.mean) / self.geometry_stats.std
        geom_tensor = torch.tensor(np.asarray(geom, dtype=np.float32), dtype=torch.float32)

        identity = str(row['identity'])
        age = float(row['age']) if 'age' in row and not pd.isna(row['age']) else -1.0

        return {
            'image': image_tensor,
            'geometry': geom_tensor,
            'identity': self.identity_to_label[identity],
            'identity_str': identity,
            'age': age,
            'path': str(image_path),
            'index': int(row_index),
        }

    @property
    def geometry_dim(self) -> int:
        return len(self.geom_cols)


class HybridTripletDataset(_BaseHybridDataset):
    """Triplet dataset for age-invariant metric learning.

    Anchor and positive samples come from the same identity, ideally with a large
    age gap. Negatives come from a different identity and can be upgraded from
    random negatives to harder negatives via an embedding cache.
    """

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[HybridTransform] = None,
        geometry_stats: Optional[GeometryStats] = None,
        min_age_gap: int = 5,
        seed: int = 42,
        hard_negative_pool_size: int = 32,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats, seed=seed)
        self.min_age_gap = int(min_age_gap)
        self.hard_negative_pool_size = int(max(4, hard_negative_pool_size))
        self.grouped_indices = {identity: list(indices) for identity, indices in self.df.groupby('identity').groups.items()}
        self.valid_identities = [identity for identity, idxs in self.grouped_indices.items() if len(idxs) >= 2]
        self.valid_anchor_indices = [idx for identity in self.valid_identities for idx in self.grouped_indices[identity]]
        self.embedding_cache: Dict[int, np.ndarray] = {}
        self.identity_choices = list(self.grouped_indices.keys())

        if not self.valid_anchor_indices:
            raise ValueError('Triplet dataset requires at least one identity with two or more images.')
        if len(self.grouped_indices) < 2:
            raise ValueError('Triplet training requires at least two distinct identities.')

    def __len__(self) -> int:
        return len(self.valid_anchor_indices)

    def update_embedding_cache(self, embedding_cache: Dict[int, np.ndarray]) -> None:
        self.embedding_cache = {
            int(index): np.asarray(embedding, dtype=np.float32)
            for index, embedding in embedding_cache.items()
        }

    def clear_embedding_cache(self) -> None:
        self.embedding_cache = {}

    def _sample_positive_index(self, anchor_index: int, identity: str, rng: random.Random) -> int:
        candidates = [idx for idx in self.grouped_indices[identity] if idx != anchor_index]
        if not candidates:
            raise RuntimeError('No positive sample available for identity with single image.')

        if 'age' not in self.df.columns:
            return rng.choice(candidates)

        anchor_age = float(self.df.iloc[anchor_index]['age'])
        age_scored = [(idx, abs(float(self.df.iloc[idx]['age']) - anchor_age)) for idx in candidates]
        age_filtered = [item for item in age_scored if item[1] >= self.min_age_gap]
        if age_filtered:
            age_scored = age_filtered
        age_scored.sort(key=lambda item: item[1], reverse=True)

        top_k = age_scored[: min(3, len(age_scored))]
        return rng.choice(top_k)[0]

    def _random_negative_candidates(self, identity: str, rng: random.Random) -> List[int]:
        candidates: List[int] = []
        negative_identity_choices = [id_ for id_ in self.identity_choices if id_ != identity]
        for _ in range(self.hard_negative_pool_size):
            negative_identity = rng.choice(negative_identity_choices)
            candidates.append(rng.choice(self.grouped_indices[negative_identity]))
        return candidates

    def _sample_negative_index(self, anchor_index: int, identity: str, rng: random.Random) -> int:
        candidates = self._random_negative_candidates(identity, rng)

        if anchor_index in self.embedding_cache:
            anchor_embedding = self.embedding_cache[anchor_index]
            scored_candidates = []
            for candidate_index in candidates:
                candidate_embedding = self.embedding_cache.get(candidate_index)
                if candidate_embedding is None:
                    continue
                similarity = float(
                    np.dot(anchor_embedding, candidate_embedding)
                    / (np.linalg.norm(anchor_embedding) * np.linalg.norm(candidate_embedding) + 1e-8)
                )
                scored_candidates.append((similarity, candidate_index))

            if scored_candidates:
                scored_candidates.sort(key=lambda item: item[0], reverse=True)
                top_hard = scored_candidates[: min(4, len(scored_candidates))]
                return rng.choice(top_hard)[1]

        if 'age' in self.df.columns:
            anchor_age = float(self.df.iloc[anchor_index]['age'])
            candidates.sort(key=lambda idx: abs(float(self.df.iloc[idx]['age']) - anchor_age))
            return rng.choice(candidates[: min(4, len(candidates))])

        return rng.choice(candidates)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        anchor_idx = self.valid_anchor_indices[index]
        anchor_row = self.df.iloc[anchor_idx]
        identity = str(anchor_row['identity'])
        rng = self._make_rng(index)
        positive_idx = self._sample_positive_index(anchor_idx, identity, rng=rng)
        negative_idx = self._sample_negative_index(anchor_idx, identity, rng=rng)

        anchor = self._load_row(anchor_row, anchor_idx, rng=self._make_rng(anchor_idx * 3 + 1))
        positive = self._load_row(self.df.iloc[positive_idx], positive_idx, rng=self._make_rng(positive_idx * 3 + 2))
        negative = self._load_row(self.df.iloc[negative_idx], negative_idx, rng=self._make_rng(negative_idx * 3 + 3))

        return {
            'anchor_image': anchor['image'],
            'anchor_geom': anchor['geometry'],
            'positive_image': positive['image'],
            'positive_geom': positive['geometry'],
            'negative_image': negative['image'],
            'negative_geom': negative['geometry'],
            'anchor_label': torch.tensor(anchor['identity'], dtype=torch.long),
            'positive_label': torch.tensor(positive['identity'], dtype=torch.long),
            'negative_label': torch.tensor(negative['identity'], dtype=torch.long),
            'anchor_age': torch.tensor(anchor['age'], dtype=torch.float32),
            'positive_age': torch.tensor(positive['age'], dtype=torch.float32),
            'negative_age': torch.tensor(negative['age'], dtype=torch.float32),
            'anchor_index': torch.tensor(int(anchor_idx), dtype=torch.long),
            'positive_index': torch.tensor(int(positive_idx), dtype=torch.long),
            'negative_index': torch.tensor(int(negative_idx), dtype=torch.long),
        }


class VerificationPairDataset(_BaseHybridDataset):
    """Pair dataset for verification metrics such as ROC, FAR, and FRR."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[HybridTransform] = None,
        geometry_stats: Optional[GeometryStats] = None,
        min_age_gap: int = 5,
        positive_pairs_per_identity: int = 10,
        negative_multiplier: float = 1.0,
        seed: int = 42,
        pairs_csv: str | Path | pd.DataFrame | None = None,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats, seed=seed)
        if pairs_csv is None:
            self.pairs = create_verification_pairs(
                self.df,
                split=None,
                positive_pairs_per_identity=positive_pairs_per_identity,
                negative_multiplier=negative_multiplier,
                min_age_gap=min_age_gap,
                seed=seed,
            )
        elif isinstance(pairs_csv, pd.DataFrame):
            self.pairs = pairs_csv.copy().reset_index(drop=True)
        else:
            self.pairs = pd.read_csv(pairs_csv).reset_index(drop=True)

        required_columns = {'idx1', 'idx2', 'label'}
        missing_columns = required_columns.difference(self.pairs.columns)
        if missing_columns:
            raise ValueError(f'Pair manifest is missing required columns: {sorted(missing_columns)}')

        self.pairs['idx1'] = self.pairs['idx1'].astype(int)
        self.pairs['idx2'] = self.pairs['idx2'].astype(int)
        self.pairs['label'] = self.pairs['label'].astype(int)

        if len(self.df) > 0:
            max_index = len(self.df) - 1
            if int(self.pairs['idx1'].max()) > max_index or int(self.pairs['idx2'].max()) > max_index:
                raise ValueError('Pair manifest indices exceed the available split rows. Regenerate pairs for the same split manifest.')

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs.iloc[index]
        first_index = int(pair['idx1'])
        second_index = int(pair['idx2'])
        first = self._load_row(self.df.iloc[first_index], first_index, rng=self._make_rng(first_index * 5 + 1))
        second = self._load_row(self.df.iloc[second_index], second_index, rng=self._make_rng(second_index * 5 + 2))
        age_gap = abs(float(first['age']) - float(second['age'])) if first['age'] >= 0 and second['age'] >= 0 else -1.0
        return {
            'image1': first['image'],
            'geom1': first['geometry'],
            'image2': second['image'],
            'geom2': second['geometry'],
            'label': torch.tensor(int(pair['label']), dtype=torch.long),
            'age1': torch.tensor(float(first['age']), dtype=torch.float32),
            'age2': torch.tensor(float(second['age']), dtype=torch.float32),
            'age_gap': torch.tensor(float(age_gap), dtype=torch.float32),
        }


class ManifestImageDataset(_BaseHybridDataset):
    """Simple dataset over unique preprocessed images for embedding extraction."""

    def __init__(
        self,
        data: str | Path | pd.DataFrame,
        split: str | None = None,
        transform: Optional[HybridTransform] = None,
        geometry_stats: Optional[GeometryStats] = None,
    ) -> None:
        super().__init__(data=data, split=split, transform=transform, geometry_stats=geometry_stats)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self._load_row(self.df.iloc[index], index, rng=self._make_rng(index))
        return {
            'index': torch.tensor(int(index), dtype=torch.long),
            'image': row['image'],
            'geometry': row['geometry'],
            'label': torch.tensor(int(row['identity']), dtype=torch.long),
            'age': torch.tensor(float(row['age']), dtype=torch.float32),
        }
