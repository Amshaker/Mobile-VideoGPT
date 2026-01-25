#!/usr/bin/env python3
"""
Polling Inference with Feedback Naturalizer

Combines real-time video polling with natural feedback generation.
Detects repetitive corrections and provides varied responses.

Usage:
    python run_polling_with_naturalizer.py sample_videos/00000340.mp4
    python run_polling_with_naturalizer.py sample_videos/00000340.mp4 --max-polls 5
"""

import os
import sys
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
from utils.feedback_naturalizer import FeedbackNaturalizer


class NaturalizedPollingEngine:
    """
    Wrapper that adds feedback naturalization to polling inference.

    Combines:
    - PollingInferenceEngine: Real-time video processing
    - FeedbackNaturalizer: Repetition detection and natural variations
    """

    def __init__(self, config: PollingConfig, naturalizer_threshold: float = 0.70):
        self.engine = PollingInferenceEngine(config)
        self.naturalizer = None
        self.naturalizer_threshold = naturalizer_threshold
        self.results = []

    def load(self) -> bool:
        """Load model and naturalizer."""
        print("\n" + "=" * 60)
        print("Loading Model...")
        print("=" * 60)

        if not self.engine.load_model():
            return False

        print("\n" + "=" * 60)
        print("Loading Feedback Naturalizer...")
        print("=" * 60)

        self.naturalizer = FeedbackNaturalizer(threshold=self.naturalizer_threshold)
        return True

    def run(self, video_source: str, max_polls: int = None) -> dict:
        """
        Run polling with naturalized feedback.

        Args:
            video_source: Path to video file
            max_polls: Maximum number of polls (None = until video ends)

        Returns:
            Summary dict with all results
        """
        self.results = []

        def on_response(poll_index: int, response: str, metrics):
            # Process through naturalizer
            result = self.naturalizer.process(response)

            # Store result
            self.results.append({
                'poll': poll_index + 1,
                'raw_response': response,
                'display': result['display'],
                'is_repeat': result['is_repeat'],
                'similarity': result['similarity'],
                'repeat_count': result['repeat_count'],
                'latency_ms': metrics.total_inference_time * 1000,
            })

            # Print formatted output
            print("\n" + "─" * 60)
            print(f"📍 Poll #{poll_index + 1}")
            print("─" * 60)
            print(f"Raw:     {response[:80]}...")
            print(f"Display: {result['display']}")

            if result['is_repeat']:
                print(f"Status:  🔄 REPEAT #{result['repeat_count']} (similarity: {result['similarity']:.2f})")
            else:
                print(f"Status:  ✨ NEW")

            print(f"Latency: {metrics.total_inference_time * 1000:.1f}ms")
            print("─" * 60)

        # Run polling loop
        summary = self.engine.run_polling_loop(
            video_source=video_source,
            on_response=on_response,
            max_polls=max_polls,
        )

        # Add naturalization results to summary
        summary['naturalized_results'] = self.results
        summary['repeat_count'] = sum(1 for r in self.results if r['is_repeat'])
        summary['unique_count'] = sum(1 for r in self.results if not r['is_repeat'])

        return summary

    def cleanup(self):
        """Clean up resources."""
        self.engine.cleanup()
        if self.naturalizer:
            self.naturalizer.reset()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Polling inference with feedback naturalization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("video_source", type=str, help="Path to video file")
    parser.add_argument("--base-model", type=str, default="Amshaker/Mobile-VideoGPT-0.5B")
    parser.add_argument("--lora-weights", type=str, default="EdgeVLM-Labs/mobile-videogpt-finetune-2000")
    parser.add_argument("--polling-interval", type=float, default=3.0)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity threshold for repeat detection")
    parser.add_argument("--prompt", type=str,
                        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?")
    parser.add_argument("--output-dir", type=str, default="results/polling")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warmup runs")

    return parser.parse_args()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║       Mobile-VideoGPT Polling with Feedback Naturalizer          ║
║                                                                  ║
║  Real-time Exercise Evaluation with Natural Response Variations  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    args = parse_args()
    print_banner()

    # Create config
    config = PollingConfig(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        polling_interval=args.polling_interval,
        prompt=args.prompt,
        output_dir=args.output_dir,
    )

    # Print settings
    print("📋 Settings:")
    print(f"   Video: {args.video_source}")
    print(f"   Polling Interval: {args.polling_interval}s")
    print(f"   Similarity Threshold: {args.threshold}")
    print(f"   Max Polls: {args.max_polls or 'unlimited'}")
    print()

    # Create engine
    engine = NaturalizedPollingEngine(config, naturalizer_threshold=args.threshold)

    try:
        # Load
        if not engine.load():
            print("❌ Failed to load model")
            return 1

        # Warmup
        if args.warmup > 0:
            print(f"\n🔥 Running {args.warmup} warmup run(s)...")
            engine.engine.warmup(num_runs=args.warmup)

        # Run
        print("\n" + "=" * 60)
        print("🎬 Starting Polling with Naturalization")
        print("=" * 60)

        summary = engine.run(
            video_source=args.video_source,
            max_polls=args.max_polls,
        )

        # Print summary
        if "error" not in summary:
            print("\n" + "=" * 60)
            print("📊 SESSION SUMMARY")
            print("=" * 60)
            print(f"   Total Polls: {summary['total_polls']}")
            print(f"   Unique Responses: {summary['unique_count']}")
            print(f"   Repeated Responses: {summary['repeat_count']}")
            print(f"   Avg Latency: {summary['latency_ms']['mean']:.1f}ms")
            print("=" * 60)

            # Print all results
            print("\n📝 All Responses:")
            for r in summary['naturalized_results']:
                status = f"🔄 Repeat #{r['repeat_count']}" if r['is_repeat'] else "✨ New"
                print(f"   [{r['poll']}] {status}: {r['display'][:60]}...")

            # Save results
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, f"naturalized_{summary['session_id']}.json")
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        return 0

    finally:
        engine.cleanup()


if __name__ == "__main__":
    sys.exit(main())
