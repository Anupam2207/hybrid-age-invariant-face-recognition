"""Streamlit demo for hybrid age-invariant face verification."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from utils.checkpointing import load_checkpoint_bundle
from utils.inference_helpers import compare_pil_images


st.set_page_config(page_title='Hybrid Age-Invariant Face Recognition', layout='wide')
st.title('Hybrid Age-Invariant Face Recognition Demo')
st.caption('Upload two face images, then the app will align faces, compute cosine similarity, and predict whether they belong to the same identity.')


@st.cache_resource(show_spinner=False)
def cached_load_checkpoint(checkpoint_path: str, processed_csv: str | None, train_split: str, device_name: str):
    device = torch.device(device_name)
    return load_checkpoint_bundle(
        checkpoint_path=checkpoint_path,
        device=device,
        processed_csv=processed_csv,
        train_split=train_split,
        legacy_image_size=224,
    )


with st.sidebar:
    st.header('Model')
    checkpoint_path = st.text_input('Checkpoint path', value='outputs/experiment_resnet50_triplet/checkpoints/best_model.pt')
    processed_csv = st.text_input('Processed manifest (optional for legacy checkpoints)', value='')
    train_split = st.text_input('Train split name', value='train')
    threshold_override = st.text_input('Threshold override (optional)', value='')
    min_detection_confidence = st.slider('Min detection confidence', min_value=0.10, max_value=0.99, value=0.50, step=0.01)
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.write(f'Runtime device: {device_name}')


left_col, right_col = st.columns(2)
with left_col:
    uploaded_image1 = st.file_uploader('Upload image 1', type=['jpg', 'jpeg', 'png'], key='image1')
with right_col:
    uploaded_image2 = st.file_uploader('Upload image 2', type=['jpg', 'jpeg', 'png'], key='image2')


if uploaded_image1 is not None and uploaded_image2 is not None:
    image1 = Image.open(uploaded_image1).convert('RGB')
    image2 = Image.open(uploaded_image2).convert('RGB')

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(image1, caption='Uploaded image 1', use_container_width=True)
    with preview_right:
        st.image(image2, caption='Uploaded image 2', use_container_width=True)

    if st.button('Compare faces', type='primary'):
        if not checkpoint_path or not Path(checkpoint_path).exists():
            st.error('Please provide a valid checkpoint path that exists on disk.')
        else:
            try:
                bundle = cached_load_checkpoint(
                    checkpoint_path=checkpoint_path,
                    processed_csv=processed_csv or None,
                    train_split=train_split,
                    device_name=device_name,
                )
                threshold = float(threshold_override) if threshold_override.strip() else None
                result, first, second = compare_pil_images(
                    bundle=bundle,
                    image1=image1,
                    image2=image2,
                    device=torch.device(device_name),
                    threshold=threshold,
                    min_detection_confidence=min_detection_confidence,
                )

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric('Cosine similarity', f"{result['similarity_score']:.4f}")
                metric_col2.metric('0-1 score', f"{result['score_0_to_1']:.4f}")
                metric_col3.metric('Threshold', f"{result['threshold']:.4f}")

                if result['prediction'] == 'same person':
                    st.success('Prediction: same person')
                else:
                    st.error('Prediction: different person')

                aligned_left, aligned_right = st.columns(2)
                with aligned_left:
                    st.image(first.aligned_rgb, caption='Aligned image 1', use_container_width=True)
                with aligned_right:
                    st.image(second.aligned_rgb, caption='Aligned image 2', use_container_width=True)

                with st.expander('Detailed output'):
                    st.json(result)
            except Exception as exc:
                st.exception(exc)
else:
    st.info('Upload two images to start the demo.')
