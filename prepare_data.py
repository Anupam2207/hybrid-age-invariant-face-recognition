"""Helper script to prepare test data for the pipeline.

This script provides options to:
1. Download a small public face dataset (LFW)
2. Generate synthetic test faces
3. Create a minimal manifest for testing

Usage:
    python prepare_data.py --mode download  # Download LFW dataset
    python prepare_data.py --mode synthetic # Generate synthetic test faces
"""

import argparse
import os
from pathlib import Path
import urllib.request
import zipfile
import csv
import numpy as np
from PIL import Image, ImageDraw
import random


def generate_synthetic_faces(num_people: int = 3, num_images_per_person: int = 4, output_dir: str = 'data/raw/synthetic'):
    """Generate synthetic face-like images for testing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = 'data/manifests/raw_manifest.csv'
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    ages = [10, 15, 25, 35, 50, 65]
    
    print(f'Generating {num_people} synthetic people with ~{num_images_per_person} images each...')
    
    for person_id in range(1, num_people + 1):
        person_dir = Path(output_dir) / f'person_{person_id:03d}'
        person_dir.mkdir(exist_ok=True)
        
        for img_idx, age in enumerate(random.sample(ages, min(num_images_per_person, len(ages)))):
            # Create a synthetic face-like image
            img = Image.new('RGB', (224, 224), color=(200, 180, 160))
            draw = ImageDraw.Draw(img)
            
            # Draw basic face features (circle for face)
            draw.ellipse([56, 56, 168, 168], outline=(100, 80, 60), width=3)
            
            # Eyes
            eye_y = 90 + (age // 10) % 5  # slight variation with age
            draw.ellipse([84, eye_y, 94, eye_y+10], fill=(30, 30, 30))
            draw.ellipse([130, eye_y, 140, eye_y+10], fill=(30, 30, 30))
            
            # Nose
            draw.polygon([(112, 100), (108, 130), (116, 130)], fill=(150, 120, 100))
            
            # Mouth
            mouth_y = 140 + (age // 20)
            draw.arc([100, mouth_y, 124, mouth_y+15], 0, 180, fill=(50, 20, 20), width=2)
            
            # Add age-related variation (wrinkles)
            if age > 40:
                for _ in range(age // 20):
                    x = random.randint(80, 160)
                    y = random.randint(80, 160)
                    draw.line([(x, y), (x+5, y)], fill=(180, 150, 140), width=1)
            
            filename = f'person_{person_id:03d}_age{age:02d}.jpg'
            filepath = person_dir / filename
            img.save(filepath)
            
            split = 'train' if random.random() < 0.8 else 'val'
            rows.append({
                'image_path': str(filepath),
                'identity': f'person_{person_id:03d}',
                'age': age,
                'split': split
            })
    
    # Write manifest CSV
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'identity', 'age', 'split'])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f'✓ Generated {len(rows)} synthetic images')
    print(f'✓ Manifest saved to: {manifest_path}')
    return manifest_path


def download_lfw_dataset(output_dir: str = 'data/raw/lfw'):
    """Download a small subset of the LFW (Labeled Faces in the Wild) dataset."""
    print('LFW dataset download would require significant disk space.')
    print('Recommend instead:')
    print('1. Download from http://vis-www.cs.umass.edu/lfw/')
    print('2. Or use a smaller dataset like CACD or MORPH')
    print('3. Or use the synthetic data generation mode (default)')
    return None


def main():
    parser = argparse.ArgumentParser(description='Prepare test data for the pipeline.')
    parser.add_argument('--mode', choices=['synthetic', 'download'], default='synthetic',
                        help='Data generation mode.')
    parser.add_argument('--num_people', type=int, default=3, help='Number of people to generate (synthetic mode).')
    parser.add_argument('--num_images_per_person', type=int, default=4, help='Images per person (synthetic mode).')
    parser.add_argument('--output_dir', type=str, default='data/raw/synthetic', help='Output directory.')
    
    args = parser.parse_args()
    
    if args.mode == 'synthetic':
        manifest = generate_synthetic_faces(args.num_people, args.num_images_per_person, args.output_dir)
        print(f'\nNext steps:')
        print(f'1. Run preprocessing:')
        print(f'   python preprocess_dataset.py \\')
        print(f'     --input_csv {manifest} \\')
        print(f'     --output_csv data/manifests/processed_manifest.csv \\')
        print(f'     --output_dir data/processed/aligned_faces')
        print(f'\n2. Then run training:')
        print(f'   python train.py --processed_csv data/manifests/processed_manifest.csv --epochs 5')
    elif args.mode == 'download':
        download_lfw_dataset(args.output_dir)


if __name__ == '__main__':
    main()
