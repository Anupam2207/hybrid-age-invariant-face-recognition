"""Normalize CACD file and folder names to safe ASCII paths."""

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path



def clean_name(name: str) -> str:
    normalized = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    normalized = normalized.replace(' ', '')
    return normalized



def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f'{stem}_{counter}{suffix}'
        if not candidate.exists():
            return candidate
        counter += 1



def main() -> None:
    parser = argparse.ArgumentParser(description='Clean folder and file names under a dataset root.')
    parser.add_argument('--data_dir', type=str, default='data/raw/cacd')
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f'Directory not found: {base_dir}')

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue

        cleaned_folder = clean_name(folder.name)
        folder_path = folder
        if folder.name != cleaned_folder:
            new_folder_path = unique_path(base_dir / cleaned_folder)
            folder.rename(new_folder_path)
            print(f'Renamed folder: {folder.name} -> {new_folder_path.name}')
            folder_path = new_folder_path

        for file_path in sorted(folder_path.iterdir()):
            if not file_path.is_file():
                continue
            cleaned_file = clean_name(file_path.name)
            if file_path.name == cleaned_file:
                continue
            new_file_path = unique_path(folder_path / cleaned_file)
            file_path.rename(new_file_path)
            print(f'Renamed file: {file_path.name} -> {new_file_path.name}')

    print('All filenames cleaned.')


if __name__ == '__main__':
    main()
