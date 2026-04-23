"""Offline dataset preprocessing.

Run this once after generating an identity-disjoint manifest. The script detects
and aligns faces, extracts landmark-based geometric features, and saves a new
processed manifest.

Expected input CSV columns:
    image_path, identity, age, split

The script keeps any additional columns unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

from utils.preprocessing import FacePreprocessor, build_processed_row



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline preprocessing for hybrid face recognition.')
    parser.add_argument('--input_csv', type=str, required=True, help='Raw manifest CSV with image_path, identity, age, split.')
    parser.add_argument('--output_csv', type=str, required=True, help='Processed output CSV path.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to store aligned faces.')
    parser.add_argument('--image_size', type=int, default=224, help='Aligned output face size.')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5)
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5)
    parser.add_argument('--summary_json', type=str, default=None, help='Optional preprocessing summary JSON.')
    parser.add_argument('--failures_csv', type=str, default=None, help='Optional CSV containing failed image paths.')
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    processed_rows: List[dict] = []
    failed_rows = 0
    failure_records: List[dict] = []

    with FacePreprocessor(
        output_size=args.image_size,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as preprocessor:
        for row in tqdm(df.to_dict(orient='records'), total=len(df), desc='Preprocessing'):
            image_path = Path(str(row['image_path']))
            try:
                processed_face = preprocessor.process_path(image_path)
            except Exception as exc:
                processed_face = None
                failure_records.append({'image_path': str(image_path), 'reason': str(exc)})

            if processed_face is None:
                failed_rows += 1
                if not any(record['image_path'] == str(image_path) for record in failure_records):
                    failure_records.append({'image_path': str(image_path), 'reason': 'No face detected'})
                continue

            identity = str(row.get('identity', 'unknown_identity'))
            aligned_filename = f"{image_path.stem}_aligned.jpg"
            aligned_path = output_dir / identity / aligned_filename
            preprocessor.save_aligned_face(processed_face.aligned_rgb, aligned_path)
            processed_rows.append(build_processed_row(row, processed_face, str(aligned_path)))

    processed_df = pd.DataFrame(processed_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_csv, index=False)

    success_rows = len(processed_df)
    summary = {
        'input_csv': str(input_csv),
        'output_csv': str(output_csv),
        'output_dir': str(output_dir),
        'requested_rows': int(len(df)),
        'successful_rows': int(success_rows),
        'failed_rows': int(failed_rows),
        'success_rate': float(success_rows / max(len(df), 1)),
        'image_size': int(args.image_size),
        'alignment_backend': 'mediapipe',
    }

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open('w', encoding='utf-8') as file:
            json.dump(summary, file, indent=2)

    if args.failures_csv is not None and failure_records:
        failures_path = Path(args.failures_csv)
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failure_records).to_csv(failures_path, index=False)

    print(f'Processed manifest saved to: {output_csv}')
    print(json.dumps(summary, indent=2))
    if failure_records[:10]:
        print('Sample failures:')
        for example in failure_records[:10]:
            print(f"  - {example['image_path']}: {example['reason']}")
    if success_rows == 0:
        raise RuntimeError('No images were successfully preprocessed. Please check dataset paths and MediaPipe installation.')


if __name__ == '__main__':
    main()
