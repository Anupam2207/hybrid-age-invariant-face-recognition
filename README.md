# Hybrid Age-Invariant Face Verification

This project verifies whether two face images belong to the same person even when the age gap is large.

The pipeline combines:
- a CNN image embedding branch
- a geometric facial-ratio branch from MediaPipe landmarks
- a fusion head trained with triplet loss

## Important note

The hybrid branch depends on **offline preprocessing**. If you used an older version of this repository, regenerate the processed manifest and retrain. The old geometry extraction path computed landmarks after aggressive affine alignment, which collapses the geometry features.

See `PROJECT_AUDIT_AND_FIXES.md` for a full engineering audit.

## Folder expectations

The recommended CACD layout is identity-organized:

```text
data/raw/cacd/
  PersonA/
    25_Person_A_0001.jpg
    37_Person_A_0002.jpg
  PersonB/
    ...
```

## 1) Generate a manifest

```bash
python generate_manifest.py \
  --data_dir data/raw/cacd \
  --output_csv data/manifests/cacd_manifest.csv \
  --split_by identity \
  --val_ratio 0.10 \
  --test_ratio 0.10
```

## 2) Preprocess the dataset

```bash
python preprocess_dataset.py \
  --input_csv data/manifests/cacd_manifest.csv \
  --output_csv data/manifests/cacd_processed.csv \
  --output_dir data/processed/aligned_faces_cacd \
  --image_size 224
```

This step:
- detects faces
- extracts geometry from a cropped face region
- aligns the image branch with an eye-based similarity transform
- stores aligned face crops and geometry features in the processed manifest

## 3) Train

```bash
python train.py \
  --processed_csv data/manifests/cacd_processed.csv \
  --train_split train \
  --val_split val \
  --backbone mobilenet_v2 \
  --mode hybrid \
  --epochs 20 \
  --batch_size 32 \
  --image_size 224 \
  --embedding_dim 128 \
  --lr 1e-3 \
  --output_dir outputs/experiment_cacd_hybrid
```

Outputs:
- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `plots/train_loss.png`
- `plots/val_metrics.png`
- `geometry_stats.json`
- `history.json`

## 4) Evaluate

```bash
python evaluate.py \
  --checkpoint outputs/experiment_cacd_hybrid/checkpoints/best_model.pt \
  --processed_csv data/manifests/cacd_processed.csv \
  --split test \
  --output_dir outputs/experiment_cacd_hybrid/test_eval
```

Reported metrics:
- accuracy
- ROC-AUC
- FAR
- FRR
- EER
- best threshold

## 5) Inference on two images

```bash
python inference.py \
  --checkpoint outputs/experiment_cacd_hybrid/checkpoints/best_model.pt \
  --image1 path/to/person_young.jpg \
  --image2 path/to/person_old.jpg
```

## Modes

- `hybrid`: CNN + geometry fusion
- `cnn_only`: CNN branch only
- `geom_only`: geometry branch only

## Recommendation

Start with:
- backbone: `mobilenet_v2`
- mode: `hybrid`
- image size: `224`
- embedding dim: `128`

Then compare against `cnn_only` as a baseline.
