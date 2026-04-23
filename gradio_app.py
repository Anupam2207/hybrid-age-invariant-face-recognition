"""Gradio demo for age-invariant face verification.

Usage:
    python gradio_app.py --checkpoint outputs/experiment_gtx1050ti_age_invariant/checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from utils.checkpointing import load_checkpoint_bundle
from utils.inference_helpers import compare_pil_images
from utils.runtime import resolve_device



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch the Gradio UI for age-invariant face verification.')
    parser.add_argument('--checkpoint', type=str, default='outputs/experiment_gtx1050ti_age_invariant/checkpoints/best_model.pt')
    parser.add_argument('--processed_csv', type=str, default=None, help='Optional; only needed for legacy checkpoints.')
    parser.add_argument('--train_split', type=str, default='train')
    parser.add_argument('--legacy_image_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--min_detection_confidence', type=float, default=0.5)
    parser.add_argument('--server_name', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=7860)
    parser.add_argument('--share', action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()



def _format_summary(result: dict) -> str:
    return (
        f"### Prediction: **{result['prediction']}**\n\n"
        f"- Cosine similarity: **{result['similarity_score']:.4f}**\n"
        f"- Normalized 0-1 score: **{result['score_0_to_1']:.4f}**\n"
        f"- Threshold used: **{result['threshold']:.4f}**\n"
        f"- Decision margin: **{result['decision_margin']:.4f}**\n"
        f"- Interpretation: **{result['confidence_interpretation']}**\n"
        f"- Detection confidence (image 1): **{result['detection_confidence_image1']:.3f}**\n"
        f"- Detection confidence (image 2): **{result['detection_confidence_image2']:.3f}**\n"
        f"- Backbone: **{result['backbone']}**\n"
        f"- Fusion: **{result['fusion_type']}**\n"
    )



def create_demo(args: argparse.Namespace) -> gr.Blocks:
    device = resolve_device(args.device)

    @lru_cache(maxsize=2)
    def cached_bundle(checkpoint_path: str, processed_csv: str | None, train_split: str, legacy_image_size: int, device_name: str):
        device_obj = resolve_device(device_name)
        return load_checkpoint_bundle(
            checkpoint_path=checkpoint_path,
            device=device_obj,
            processed_csv=processed_csv,
            train_split=train_split,
            legacy_image_size=legacy_image_size,
        )

    def compare(
        image1: Image.Image | None,
        image2: Image.Image | None,
        checkpoint_path: str,
        processed_csv: str,
        train_split: str,
        threshold: float | None,
        min_detection_confidence: float,
    ):
        if image1 is None or image2 is None:
            raise gr.Error('Please upload both images.')

        checkpoint_path = checkpoint_path.strip()
        if not checkpoint_path:
            raise gr.Error('Please provide a checkpoint path.')
        if not Path(checkpoint_path).exists():
            raise gr.Error(f'Checkpoint not found: {checkpoint_path}')

        bundle = cached_bundle(
            checkpoint_path=checkpoint_path,
            processed_csv=processed_csv.strip() or None,
            train_split=train_split.strip() or 'train',
            legacy_image_size=args.legacy_image_size,
            device_name=str(device),
        )

        result, first, second = compare_pil_images(
            bundle=bundle,
            image1=image1,
            image2=image2,
            device=device,
            threshold=threshold,
            min_detection_confidence=min_detection_confidence,
        )

        aligned1 = Image.fromarray(np.asarray(first.aligned_rgb, dtype=np.uint8))
        aligned2 = Image.fromarray(np.asarray(second.aligned_rgb, dtype=np.uint8))
        return (
            _format_summary(result),
            result['similarity_score'],
            result['threshold'],
            result['prediction'],
            result['confidence_interpretation'],
            aligned1,
            aligned2,
            result,
        )

    with gr.Blocks(title='Age-Invariant Face Recognition Demo') as demo:
        gr.Markdown(
            '# Age-Invariant Face Recognition\n'
            'Upload two face images. The app aligns both faces, extracts embeddings, computes cosine similarity, and predicts whether they belong to the same person.'
        )

        with gr.Row():
            with gr.Column(scale=2):
                image1_input = gr.Image(label='Image 1', type='pil')
            with gr.Column(scale=2):
                image2_input = gr.Image(label='Image 2', type='pil')

        with gr.Accordion('Model settings', open=True):
            checkpoint_input = gr.Textbox(label='Checkpoint path', value=args.checkpoint)
            processed_csv_input = gr.Textbox(label='Processed manifest (optional for legacy checkpoints)', value=args.processed_csv or '')
            train_split_input = gr.Textbox(label='Train split name', value=args.train_split)
            threshold_input = gr.Number(label='Threshold override (optional)', value=args.threshold)
            min_detection_input = gr.Slider(label='Minimum detection confidence', minimum=0.1, maximum=0.99, value=args.min_detection_confidence, step=0.01)

        compare_button = gr.Button('Compare faces', variant='primary')

        with gr.Row():
            summary_output = gr.Markdown()
        with gr.Row():
            similarity_output = gr.Number(label='Cosine similarity')
            threshold_output = gr.Number(label='Threshold used')
        with gr.Row():
            prediction_output = gr.Textbox(label='Prediction')
            interpretation_output = gr.Textbox(label='Interpretation')
        with gr.Row():
            aligned1_output = gr.Image(label='Aligned face 1')
            aligned2_output = gr.Image(label='Aligned face 2')
        raw_json_output = gr.JSON(label='Detailed output')

        compare_button.click(
            fn=compare,
            inputs=[
                image1_input,
                image2_input,
                checkpoint_input,
                processed_csv_input,
                train_split_input,
                threshold_input,
                min_detection_input,
            ],
            outputs=[
                summary_output,
                similarity_output,
                threshold_output,
                prediction_output,
                interpretation_output,
                aligned1_output,
                aligned2_output,
                raw_json_output,
            ],
        )

    return demo


if __name__ == '__main__':
    cli_args = parse_args()
    app = create_demo(cli_args)
    app.launch(server_name=cli_args.server_name, server_port=cli_args.server_port, share=cli_args.share)
