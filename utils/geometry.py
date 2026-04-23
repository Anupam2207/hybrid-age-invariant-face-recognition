"""Geometric feature extraction utilities.

The first nine features keep backward-compatible semantics with the original
project. Extra features are appended afterward to make the geometry branch more
expressive without breaking legacy checkpoints that only expect the first nine
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class FaceMeshIndices:
    """Indices used from the MediaPipe Face Mesh topology."""

    left_eye_outer: int = 33
    left_eye_inner: int = 133
    left_eye_top: int = 159
    left_eye_bottom: int = 145

    right_eye_inner: int = 362
    right_eye_outer: int = 263
    right_eye_top: int = 386
    right_eye_bottom: int = 374

    nose_tip: int = 1
    nose_left: int = 98
    nose_right: int = 327

    mouth_left: int = 61
    mouth_right: int = 291
    upper_lip: int = 13
    lower_lip: int = 14

    face_left: int = 234
    face_right: int = 454
    forehead: int = 10
    chin: int = 152

    left_brow_outer: int = 70
    left_brow_inner: int = 105
    right_brow_inner: int = 334
    right_brow_outer: int = 300


IDX = FaceMeshIndices()



def _to_xy(landmarks: Sequence, index: int) -> np.ndarray:
    landmark = landmarks[index]
    return np.asarray([landmark.x, landmark.y], dtype=np.float32)



def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))



def _center(points: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(points), axis=0)
    return stacked.mean(axis=0)



def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / (denominator + EPS))


FEATURE_NAMES = [
    # Original project features kept first for backward compatibility.
    'inter_eye_over_face_width',
    'nose_mouth_over_face_height',
    'left_eye_nose_over_right_eye_nose',
    'mouth_width_over_inter_eye',
    'nose_mouth_over_inter_eye',
    'eye_mouth_over_face_height',
    'nose_chin_over_face_height',
    'nose_forehead_over_face_height',
    'eye_y_diff_over_face_height',
    # Stronger geometry branch features.
    'face_width_over_face_height',
    'mean_eye_width_over_inter_eye',
    'mean_eye_height_over_face_height',
    'lip_opening_over_face_height',
    'brow_eye_over_face_height',
    'nose_width_over_face_width',
    'nose_chin_over_nose_forehead',
]


# Only one feature is left/right asymmetric. When the training image is
# horizontally flipped, this feature must be updated too.
HORIZONTAL_FLIP_INVERSE_FEATURE_INDICES = [2]



def extract_geometric_ratios(landmarks: Sequence) -> np.ndarray:
    """Extract a compact ratio-based geometry vector from face mesh landmarks."""

    left_eye = _center([
        _to_xy(landmarks, IDX.left_eye_outer),
        _to_xy(landmarks, IDX.left_eye_inner),
    ])
    right_eye = _center([
        _to_xy(landmarks, IDX.right_eye_outer),
        _to_xy(landmarks, IDX.right_eye_inner),
    ])
    left_eye_top = _to_xy(landmarks, IDX.left_eye_top)
    left_eye_bottom = _to_xy(landmarks, IDX.left_eye_bottom)
    right_eye_top = _to_xy(landmarks, IDX.right_eye_top)
    right_eye_bottom = _to_xy(landmarks, IDX.right_eye_bottom)

    nose_tip = _to_xy(landmarks, IDX.nose_tip)
    nose_left = _to_xy(landmarks, IDX.nose_left)
    nose_right = _to_xy(landmarks, IDX.nose_right)

    mouth_left = _to_xy(landmarks, IDX.mouth_left)
    mouth_right = _to_xy(landmarks, IDX.mouth_right)
    upper_lip = _to_xy(landmarks, IDX.upper_lip)
    lower_lip = _to_xy(landmarks, IDX.lower_lip)

    face_left = _to_xy(landmarks, IDX.face_left)
    face_right = _to_xy(landmarks, IDX.face_right)
    forehead = _to_xy(landmarks, IDX.forehead)
    chin = _to_xy(landmarks, IDX.chin)

    left_brow = _center([
        _to_xy(landmarks, IDX.left_brow_outer),
        _to_xy(landmarks, IDX.left_brow_inner),
    ])
    right_brow = _center([
        _to_xy(landmarks, IDX.right_brow_inner),
        _to_xy(landmarks, IDX.right_brow_outer),
    ])

    mouth_center = _center([mouth_left, mouth_right, upper_lip, lower_lip])
    eye_center = _center([left_eye, right_eye])
    brow_center = _center([left_brow, right_brow])

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

    left_eye_width = _distance(_to_xy(landmarks, IDX.left_eye_outer), _to_xy(landmarks, IDX.left_eye_inner))
    right_eye_width = _distance(_to_xy(landmarks, IDX.right_eye_outer), _to_xy(landmarks, IDX.right_eye_inner))
    mean_eye_width = (left_eye_width + right_eye_width) * 0.5

    left_eye_height = _distance(left_eye_top, left_eye_bottom)
    right_eye_height = _distance(right_eye_top, right_eye_bottom)
    mean_eye_height = (left_eye_height + right_eye_height) * 0.5

    lip_opening = _distance(upper_lip, lower_lip)
    brow_eye = _distance(brow_center, eye_center)
    nose_width = _distance(nose_left, nose_right)

    features = np.asarray(
        [
            _safe_ratio(inter_eye, face_width),
            _safe_ratio(nose_mouth, face_height),
            _safe_ratio(left_eye_nose, right_eye_nose),
            _safe_ratio(mouth_width, inter_eye),
            _safe_ratio(nose_mouth, inter_eye),
            _safe_ratio(eye_mouth, face_height),
            _safe_ratio(nose_chin, face_height),
            _safe_ratio(nose_forehead, face_height),
            _safe_ratio(eye_y_diff, face_height),
            _safe_ratio(face_width, face_height),
            _safe_ratio(mean_eye_width, inter_eye),
            _safe_ratio(mean_eye_height, face_height),
            _safe_ratio(lip_opening, face_height),
            _safe_ratio(brow_eye, face_height),
            _safe_ratio(nose_width, face_width),
            _safe_ratio(nose_chin, nose_forehead),
        ],
        dtype=np.float32,
    )

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)



def flip_geometry_features(features: np.ndarray) -> np.ndarray:
    """Update geometry features after a horizontal image flip.

    Most ratio features are already symmetric. The only left/right-sensitive
    value in the current feature vector is the eye-to-nose ratio at index 2,
    which becomes its reciprocal after flipping.
    """

    if features is None:
        return features

    flipped = np.asarray(features, dtype=np.float32).copy()
    for feature_index in HORIZONTAL_FLIP_INVERSE_FEATURE_INDICES:
        if feature_index < len(flipped):
            flipped[feature_index] = _safe_ratio(1.0, float(flipped[feature_index]))
    return flipped



def normalize_geometry_features(
    features: np.ndarray,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> np.ndarray:
    """Apply z-score normalization to geometry features."""

    features = np.asarray(features, dtype=np.float32)
    if mean is None or std is None:
        return features
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (features - mean) / (std + EPS)
