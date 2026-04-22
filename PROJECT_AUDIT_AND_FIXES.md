# Project audit and fixes

## What was broken

### 1) The geometry branch was effectively broken
The original pipeline aligned faces using **left eye + right eye + mouth** and then extracted MediaPipe mesh landmarks **after** that affine alignment. That destroys much of the very geometry the branch is supposed to learn, because the alignment pins those landmarks to nearly fixed canonical positions.

Symptoms:
- the saved `outputs/experiment_01/geometry_stats.json` shows per-feature standard deviations ~`1e-6`
- this means the geometry branch was being fed almost constant inputs
- a hybrid model cannot benefit from geometry if the geometry vector has already been normalized away

### 2) Left/right eye keypoints were swapped
MediaPipe Face Detection returns keypoints in Python order:
`right eye, left eye, nose tip, mouth center, ...`

The original code read them as `left_eye = keypoints[0]` and `right_eye = keypoints[1]`, which is reversed. That corrupts alignment.

### 3) Image augmentation did not match geometry augmentation
The training transform used `RandomHorizontalFlip`, but the geometric features were precomputed offline and not flipped. So the image branch could see a mirrored face while the geometry branch still represented the original orientation.

### 4) Training and evaluation were incompatible
- `train.py` saved only `model.state_dict()` to `outputs/best_model.pth`
- `evaluate.py` expected a rich checkpoint containing model config, geometry stats, image size, threshold, etc.

So evaluation would fail on checkpoints produced by training.

### 5) Validation was never actually used during training
`train.py` created a validation loader but never ran validation, never selected the best epoch, and always saved the final weights.

### 6) Inference was not using the real pipeline
The old `inference.py`:
- loaded a model at import time
- hardcoded Windows file paths
- used a fixed `geometry_input_dim=9`
- used **zero geometry** instead of extracting landmarks
- used a fixed threshold of `0.75`

That makes inference inconsistent with training.

### 7) The manifest split leaked identities across train/val
The old manifest generator split images randomly per image, not per identity. For verification research, that leaks the same identity into both training and validation/test, which inflates results.

### 8) Data loading was hardcoded to random subsets
The old training script always sampled `20000` train rows and `5000` validation rows. That silently discarded data and also crashed on smaller manifests.

### 9) Preprocessing was brittle
One unreadable image could abort the preprocessing run instead of just being counted as a failed sample.

## What I changed

### Preprocessing
- fixed MediaPipe eye ordering
- changed geometry extraction to use a **cropped face region before alignment**, not the aligned image
- changed CNN alignment to an **eye-based similarity transform**
- made preprocessing robust to bad files and log sample failures

### Data pipeline
- disabled horizontal flip by default because it breaks left/right-sensitive geometry
- added safer path checks in the dataset loader
- made manifest generation deterministic and added **identity-disjoint** splitting

### Training
- added missing CLI arguments that the README implied existed
- removed the hardcoded 20k/5k subsampling behavior
- added real validation every epoch
- saved a **rich checkpoint** with metadata and geometry stats
- saved training history and plots

### Evaluation
- made evaluation compatible with both:
  - new rich checkpoints
  - old plain `state_dict` checkpoints

### Inference
- replaced the hardcoded script with a real CLI
- runs the same preprocessing pipeline online
- extracts geometry instead of using zeros
- uses the threshold stored in the checkpoint (or a user override)

### Utility scripts
- fixed filename cleaning logic
- improved CACD organization script behavior

## Files changed

- `utils/preprocessing.py`
- `dataset.py`
- `model.py`
- `train.py`
- `evaluate.py`
- `inference.py`
- `generate_manifest.py`
- `preprocess_dataset.py`
- `fix_filenames.py`
- `organize_cacd.py`
- `utils/checkpointing.py` (new)

## Important: what you must redo

Because the original geometry features were extracted from the wrong stage, you should **not trust old processed manifests** for hybrid training.

You should redo these steps:
1. regenerate the manifest (prefer identity split)
2. rerun preprocessing
3. retrain the model
4. reevaluate using the new checkpoint

## Recommended commands

### 1) Generate manifest
```bash
python generate_manifest.py \
  --data_dir data/raw/cacd \
  --output_csv data/manifests/cacd_manifest.csv \
  --split_by identity \
  --val_ratio 0.10 \
  --test_ratio 0.10
```

### 2) Preprocess
```bash
python preprocess_dataset.py \
  --input_csv data/manifests/cacd_manifest.csv \
  --output_csv data/manifests/cacd_processed.csv \
  --output_dir data/processed/aligned_faces_cacd \
  --image_size 224
```

### 3) Train
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

### 4) Evaluate
```bash
python evaluate.py \
  --checkpoint outputs/experiment_cacd_hybrid/checkpoints/best_model.pt \
  --processed_csv data/manifests/cacd_processed.csv \
  --split test \
  --output_dir outputs/experiment_cacd_hybrid/test_eval
```

### 5) Inference on two raw images
```bash
python inference.py \
  --checkpoint outputs/experiment_cacd_hybrid/checkpoints/best_model.pt \
  --image1 path/to/person_young.jpg \
  --image2 path/to/person_old.jpg
```

## Expected result after the fix

The project will now be structurally correct for hybrid age-invariant verification, but performance will still depend on:
- CACD label quality/noise
- how clean your raw images are
- whether the train/val/test protocol matches your intended use case
- whether you compare against a strong face-recognition baseline

If you want the strongest practical next step after this fix, add a stronger identity backbone or a classification head (e.g. ArcFace/CosFace-style supervision) on top of the metric loss.
