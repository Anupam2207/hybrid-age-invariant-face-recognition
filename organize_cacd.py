"""Organize a flat CACD directory into identity subfolders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path



def infer_identity(filename: str) -> str | None:
    parts = filename.split('_')
    if len(parts) < 3:
        return None
    return '_'.join(parts[1:-1])



def main() -> None:
    parser = argparse.ArgumentParser(description='Organize flat CACD image files into identity folders.')
    parser.add_argument('--src_dir', type=str, default='data/raw/cacd')
    parser.add_argument('--dst_dir', type=str, default='data/raw/cacd_organized')
    parser.add_argument('--move', action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if not image_files:
        print(f'No flat image files found in {src_dir}. The dataset may already be organized.')
        return

    op = shutil.move if args.move else shutil.copy2

    count = 0
    for image_path in image_files:
        identity = infer_identity(image_path.name)
        if identity is None:
            continue
        person_folder = dst_dir / identity
        person_folder.mkdir(parents=True, exist_ok=True)
        op(str(image_path), str(person_folder / image_path.name))
        count += 1

    print(f'Organized {count} files into {dst_dir}')


if __name__ == '__main__':
    main()
