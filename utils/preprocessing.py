"""Face detection, alignment, and geometric feature extraction.

The project uses MediaPipe for lightweight face detection and dense landmark
estimation. During training, these steps should be executed offline and cached.
During inference, the same module is reused online for two input images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from .geometry import FEATURE_NAMES, extract_geometric_ratios


@dataclass
class ProcessedFace:
    """Container for the preprocessed face used by the hybrid model."""

    aligned_rgb: np.ndarray
    geometry_features: np.ndarray
    detection_confidence: float


class FacePreprocessor:
    """Preprocess a face image using MediaPipe face detection and face mesh.

    Two different views are used on purpose:
    1. A lightly cropped face region is fed to Face Mesh to compute geometry.
       This preserves naturally occurring proportions.
    2. An eye-based similarity alignment is used for the CNN image branch.

    This avoids the previous failure mode where affine alignment anchored by eye
    and mouth landmarks collapsed the geometry branch into almost-constant
    features.
    """

    def __init__(
        self,
        output_size: int = 224,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_selection: int = 1,
        refine_landmarks: bool = True,
        crop_padding: float = 0.25,
        desired_left_eye: tuple[float, float] = (0.35, 0.35),
    ) -> None:
        self.output_size = int(output_size)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.crop_padding = float(crop_padding)
        self.desired_left_eye = desired_left_eye

        self._mp_face_detection = mp.solutions.face_detection
        self._mp_face_mesh = mp.solutions.face_mesh

        self.face_detector = self._mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=self.min_detection_confidence,
        )
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def __enter__(self) -> 'FacePreprocessor':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self.face_detector is not None:
            self.face_detector.close()
        if self.face_mesh is not None:
            self.face_mesh.close()

    @staticmethod
    def _read_rgb(image_path: str | Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f'Could not read image: {image_path}')
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _relative_point_to_pixel(relative_point, width: int, height: int) -> np.ndarray:
        return np.asarray([relative_point.x * width, relative_point.y * height], dtype=np.float32)

    @staticmethod
    def _relative_bbox_to_xyxy(relative_bbox, width: int, height: int) -> np.ndarray:
        x1 = max(0.0, relative_bbox.xmin * width)
        y1 = max(0.0, relative_bbox.ymin * height)
        x2 = min(float(width), x1 + relative_bbox.width * width)
        y2 = min(float(height), y1 + relative_bbox.height * height)
        return np.asarray([x1, y1, x2, y2], dtype=np.float32)

    def detect_keypoints(self, image_rgb: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """Detect a face and return eye and mouth keypoints for alignment.

        MediaPipe Face Detection returns keypoints in this order for Python:
        right eye, left eye, nose tip, mouth center, right ear, left ear.
        """

        results = self.face_detector.process(image_rgb)
        if results.detections is None:
            return None

        detection = max(results.detections, key=lambda d: d.score[0])
        height, width = image_rgb.shape[:2]
        keypoints = detection.location_data.relative_keypoints
        bbox = detection.location_data.relative_bounding_box
        score = float(detection.score[0])

        return {
            'left_eye': self._relative_point_to_pixel(keypoints[1], width, height),
            'right_eye': self._relative_point_to_pixel(keypoints[0], width, height),
            'nose_tip': self._relative_point_to_pixel(keypoints[2], width, height),
            'mouth': self._relative_point_to_pixel(keypoints[3], width, height),
            'bbox_xyxy': self._relative_bbox_to_xyxy(bbox, width, height),
            'score': np.asarray([score], dtype=np.float32),
        }

    def crop_face_region(self, image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
        """Crop a padded face region for geometry extraction."""

        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy.astype(np.float32)
        bw = x2 - x1
        bh = y2 - y1
        pad_x = bw * self.crop_padding
        pad_y = bh * self.crop_padding

        x1 = int(max(0, np.floor(x1 - pad_x)))
        y1 = int(max(0, np.floor(y1 - pad_y)))
        x2 = int(min(w, np.ceil(x2 + pad_x)))
        y2 = int(min(h, np.ceil(y2 + pad_y)))

        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid face bounding box after padding.')
        return image_rgb[y1:y2, x1:x2]

    def align_face(self, image_rgb: np.ndarray, keypoints: Dict[str, np.ndarray]) -> np.ndarray:
        """Align the face using a similarity transform based on the eye centers."""

        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        d_y = float(right_eye[1] - left_eye[1])
        d_x = float(right_eye[0] - left_eye[0])
        dist = float(np.hypot(d_x, d_y))
        if dist < 1e-6:
            return cv2.resize(image_rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)

        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.output_size
        scale = desired_dist / dist

        eyes_center = tuple(((left_eye + right_eye) * 0.5).tolist())
        angle = np.degrees(np.arctan2(d_y, d_x))
        transform = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        t_x = self.output_size * 0.5
        t_y = self.output_size * self.desired_left_eye[1]
        transform[0, 2] += t_x - eyes_center[0]
        transform[1, 2] += t_y - eyes_center[1]

        aligned = cv2.warpAffine(
            image_rgb,
            transform,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned

    def extract_landmarks(self, image_rgb: np.ndarray):
        """Run MediaPipe face mesh on an image or crop."""

        results = self.face_mesh.process(image_rgb)
        if results.multi_face_landmarks is None:
            return None
        return results.multi_face_landmarks[0].landmark

    def process_rgb(self, image_rgb: np.ndarray) -> Optional[ProcessedFace]:
        """Full preprocessing pipeline on an RGB image."""

        keypoints = self.detect_keypoints(image_rgb)
        if keypoints is None:
            return None

        try:
            face_crop_rgb = self.crop_face_region(image_rgb, keypoints['bbox_xyxy'])
        except Exception:
            face_crop_rgb = image_rgb

        landmarks = self.extract_landmarks(face_crop_rgb)
        if landmarks is None:
            landmarks = self.extract_landmarks(image_rgb)
        if landmarks is None:
            return None

        geom = extract_geometric_ratios(landmarks)
        aligned_rgb = self.align_face(image_rgb, keypoints)
        score = float(keypoints['score'][0])
        return ProcessedFace(aligned_rgb=aligned_rgb, geometry_features=geom, detection_confidence=score)

    def process_path(self, image_path: str | Path) -> Optional[ProcessedFace]:
        """Read an image from disk and preprocess it."""

        image_rgb = self._read_rgb(image_path)
        return self.process_rgb(image_rgb)

    @staticmethod
    def save_aligned_face(aligned_rgb: np.ndarray, save_path: str | Path) -> None:
        """Save an aligned RGB face image to disk."""

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(aligned_rgb).save(save_path, quality=95)



def build_processed_row(
    source_row: Dict[str, object],
    processed_face: ProcessedFace,
    aligned_path: str,
) -> Dict[str, object]:
    """Create an output CSV row for a preprocessed image."""

    row = dict(source_row)
    row['aligned_path'] = aligned_path
    row['detection_confidence'] = processed_face.detection_confidence
    for idx, feature_name in enumerate(FEATURE_NAMES):
        row[f'g{idx}'] = float(processed_face.geometry_features[idx])
        row[f'g{idx}_name'] = feature_name
    return row
