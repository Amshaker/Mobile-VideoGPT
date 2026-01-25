"""
Inference with Feedback Naturalizer

Runs MobileVideoGPT inference with natural feedback handling.
Detects repetitive responses and provides varied phrases.

Usage:
    python inference_with_naturalizer.py
"""

import sys
import os
import warnings
import logging

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from mobilevideogpt.utils import preprocess_input
from utils.feedback_naturalizer import FeedbackNaturalizer


def load_model(pretrained_path: str, device: str = "cuda"):
    """Loads the pre-trained model and tokenizer."""
    config = AutoConfig.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        torch_dtype=torch.float16
    )
    model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, video_path: str, prompt: str):
    """Runs inference on the given video file."""
    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, video_path, prompt
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().cuda(),
            context_images=torch.stack(context_frames, dim=0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    return outputs


def main():
    pretrained_path = "Amshaker/Mobile-VideoGPT-0.5B"
    video_path = "sample_videos/00000340.mp4"
    prompt = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

    # Load model
    print("Loading MobileVideoGPT model...")
    model, tokenizer = load_model(pretrained_path)
    print("Model loaded!\n")

    # Initialize feedback naturalizer
    naturalizer = FeedbackNaturalizer(threshold=0.70)

    # Test: Run inference 3 times on same video
    print("=" * 60)
    print("Test: Same video inferenced 3 times")
    print("=" * 60)

    for i in range(1, 4):
        print(f"\n--- Inference {i} ---")

        # Get raw output from model
        raw_output = run_inference(model, tokenizer, video_path, prompt)

        # Process through naturalizer
        result = naturalizer.process(raw_output)

        print(f"Raw Output: {raw_output[:80]}...")
        print(f"Display:    {result['display']}")

        if result['is_repeat']:
            print(f"Status:     REPEAT #{result['repeat_count']} (similarity: {result['similarity']:.2f})")
        else:
            print("Status:     NEW")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
