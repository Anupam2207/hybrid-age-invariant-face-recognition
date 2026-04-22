"""Geometric feature extraction utilities.

This module converts MediaPipe face mesh landmarks into a compact vector of
scale-invariant geometric ratios. The idea is to use facial proportions that are
relatively more stable across age changes than raw texture alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class FaceMeshIndices:
    """Indices used from MediaPipe Face Mesh / Face Landmarker topology."""

    left_eye_outer: int = 33
    left_eye_inner: int = 133
    right_eye_inner: int = 362
    right_eye_outer: int = 263
    nose_tip: int = 1
    mouth_left: int = 61
    mouth_right: int = 291
    upper_lip: int = 13
    lower_lip: int = 14
    face_left: int = 234
    face_right: int = 454
    forehead: int = 10
    chin: int = 152


IDX = FaceMeshIndices()


def _to_xy(landmarks: Sequence, index: int) -> np.ndarray:
    """Return normalized x, y coordinates for a landmark.

    MediaPipe returns coordinates already normalized in [0, 1], so geometric
    ratios can be computed directly without converting to pixel coordinates.
    """

    landmark = landmarks[index]
    return np.asarray([landmark.x, landmark.y], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _center(points: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(points), axis=0)
    return stacked.mean(axis=0)


FEATURE_NAMES = [
    'inter_eye_over_face_width',
    'nose_mouth_over_face_height',
    'left_eye_nose_over_right_eye_nose',
    'mouth_width_over_inter_eye',
    'nose_mouth_over_inter_eye',
    'eye_mouth_over_face_height',
    'nose_chin_over_face_height',
    'nose_forehead_over_face_height',
    'eye_y_diff_over_face_height',
]


def extract_geometric_ratios(landmarks: Sequence) -> np.ndarray:
    """Extract a compact ratio-based geometry vector from face mesh landmarks.

    Parameters
    ----------
    landmarks:
        A MediaPipe landmark sequence (typically 468 or 478 landmarks).

    Returns
    -------
    np.ndarray
        A 1-D float32 vector of geometric features.
    """

    left_eye = _center([_to_xy(landmarks, IDX.left_eye_outer), _to_xy(landmarks, IDX.left_eye_inner)])
    right_eye = _center([_to_xy(landmarks, IDX.right_eye_outer), _to_xy(landmarks, IDX.right_eye_inner)])
    nose_tip = _to_xy(landmarks, IDX.nose_tip)
    mouth_left = _to_xy(landmarks, IDX.mouth_left)
    mouth_right = _to_xy(landmarks, IDX.mouth_right)
    upper_lip = _to_xy(landmarks, IDX.upper_lip)
    lower_lip = _to_xy(landmarks, IDX.lower_lip)
    face_left = _to_xy(landmarks, IDX.face_left)
    face_right = _to_xy(landmarks, IDX.face_right)
    forehead = _to_xy(landmarks, IDX.forehead)
    chin = _to_xy(landmarks, IDX.chin)

    mouth_center = _center([mouth_left, mouth_right, upper_lip, lower_lip])
    eye_center = _center([left_eye, right_eye])

    inter_eye = _distance(left_eye, right_eye)
    face_width = _distance(face_left, face_right)
    face_height = _distance(forehead, chin)
    mouth_width = _distance(mouth_left, mouth_right)
    nose_mouth = _distance(nose_tip, mouth_center)
    left_eye_nose = _distance(left_eye, nose_tip)
    right_eye_nose = _distance(right_eye, nose_tip)
    eye_mouth = _distance(eye_center, mouth_center)
    nose_chin = _distance(nose_tip, chin)
    nose_forehead = _distance(nose_tip, forehead)
    eye_y_diff = abs(float(left_eye[1] - right_eye[1]))

    features = np.asarray(
        [
            inter_eye / (face_width + EPS),
            nose_mouth / (face_height + EPS),
            left_eye_nose / (right_eye_nose + EPS),
            mouth_width / (inter_eye + EPS),
            nose_mouth / (inter_eye + EPS),
            eye_mouth / (face_height + EPS),
            nose_chin / (face_height + EPS),
            nose_forehead / (face_height + EPS),
            eye_y_diff / (face_height + EPS),
        ],
        dtype=np.float32,
    )
    return features


def normalize_geometry_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Apply z-score normalization to geometry features.

    If statistics are not supplied, the function returns the original vector.
    """

    features = np.asarray(features, dtype=np.float32)
    if mean is None or std is None:
        return features
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (features - mean) / (std + EPS)
