#!/usr/bin/env python3
"""
Main entry point for polling-based streaming inference.
Run this script to perform real-time exercise form evaluation on video streams.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine


def setup_global_logging(log_level: str = "INFO"):
    """Setup global logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress noisy loggers
    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polling-based streaming inference for Mobile-VideoGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "video_source",
        type=str,
        help="Path to video file or stream URL (e.g., 0 for webcam, rtsp://...)",
    )

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default="Amshaker/Mobile-VideoGPT-0.5B",
        help="HuggingFace path to base model",
    )
    parser.add_argument(
        "--lora-weights",
        type=str,
        default="EdgeVLM-Labs/mobile-videogpt-finetune-2000",
        help="HuggingFace path to LoRA weights",
    )

    # Polling arguments
    parser.add_argument(
        "--polling-interval",
        type=float,
        default=3.0,
        help="Seconds between inference calls",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=300.0,
        help="Maximum total polling duration in seconds",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=None,
        help="Maximum number of polls (None = unlimited)",
    )

    # Inference arguments
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
        help="Inference prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )

    # Video processing arguments
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to sample per inference",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Frame sampling rate",
    )

    # Quantization arguments
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )

    # Warmup arguments
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Number of warmup runs before actual inference (0 = no warmup)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/polling",
        help="Directory to save results",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/polling",
        help="Directory to save logs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         Mobile-VideoGPT Polling Inference Engine                 ║
║                                                                  ║
║  Real-time Exercise Form Evaluation with LoRA Adapters           ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def on_response_callback(poll_index: int, response: str, metrics):
    """Callback for each inference response."""
    print(f"\n{'─'*60}")
    print(f"📊 Poll #{poll_index + 1} Complete")
    print(f"   Latency: {metrics.total_inference_time*1000:.1f}ms")
    print(f"   TTFT: {metrics.time_to_first_token*1000:.1f}ms")
    print(f"   Tokens/s: {metrics.tokens_per_second:.1f}")
    print(f"{'─'*60}\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    setup_global_logging(args.log_level)

    print_banner()

    # Create config
    config = PollingConfig(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        polling_interval=args.polling_interval,
        max_polling_duration=args.max_duration,
        num_frames=args.num_frames,
        fps=args.fps,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )

    # Print configuration
    print("\n📋 Configuration:")
    print(f"   Base Model: {config.base_model_path}")
    print(f"   LoRA Weights: {config.lora_weights_path}")
    print(f"   Video Source: {args.video_source}")
    print(f"   Polling Interval: {config.polling_interval}s")
    print(f"   Max Duration: {config.max_polling_duration}s")
    print(f"   Num Frames: {config.num_frames}")
    print(f"   Prompt: {config.prompt[:80]}...")
    print()

    # Create engine
    engine = PollingInferenceEngine(config)

    try:
        # Load model
        print("🔄 Loading model...")
        if not engine.load_model():
            print("❌ Failed to load model")
            return 1

        print("✅ Model loaded successfully\n")

        # Warmup if requested
        if args.warmup_runs > 0:
            print(f"🔥 Running {args.warmup_runs} warmup run(s)...")
            engine.warmup(num_runs=args.warmup_runs)
            print("✅ Warmup complete\n")

        # Run polling loop
        print(f"🎬 Starting polling on: {args.video_source}")
        print(f"   Press Ctrl+C to stop\n")

        summary = engine.run_polling_loop(
            video_source=args.video_source,
            on_response=on_response_callback,
            max_polls=args.max_polls,
        )

        # Print final summary
        if "error" not in summary:
            print("\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            print(f"   Session ID: {summary['session_id']}")
            print(f"   Duration: {summary['duration_seconds']:.2f}s")
            print(f"   Total Polls: {summary['total_polls']}")
            print(f"   Success Rate: {summary['success_rate']:.1f}%")
            print()
            print("   Latency (ms):")
            print(f"      Mean: {summary['latency_ms']['mean']:.2f}")
            print(f"      Median: {summary['latency_ms']['median']:.2f}")
            print(f"      Min: {summary['latency_ms']['min']:.2f}")
            print(f"      Max: {summary['latency_ms']['max']:.2f}")
            print()
            print("   Time to First Token (ms):")
            print(f"      Mean: {summary['time_to_first_token_ms']['mean']:.2f}")
            print(f"      Median: {summary['time_to_first_token_ms']['median']:.2f}")
            print()
            print(f"   Throughput: {summary['tokens_per_second']['mean']:.2f} tokens/s")
            print("="*60)

            # Save summary to file
            summary_file = os.path.join(
                config.output_dir,
                f"summary_{summary['session_id']}.json"
            )
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Summary saved to: {summary_file}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 0

    finally:
        engine.cleanup()


if __name__ == "__main__":
    sys.exit(main())
