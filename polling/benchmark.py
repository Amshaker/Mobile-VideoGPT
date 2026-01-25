#!/usr/bin/env python3
"""
Benchmark script for polling inference.
Runs comprehensive benchmarks and generates a detailed report.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
from polling.utils import (
    get_gpu_memory_info,
    estimate_memory_requirements,
    check_video_file,
    PerformanceProfiler,
)


def run_benchmark(
    video_path: str,
    polling_intervals: list = [1, 2, 3, 5],
    num_polls_per_interval: int = 5,
    base_model: str = "Amshaker/Mobile-VideoGPT-0.5B",
    lora_weights: str = "EdgeVLM-Labs/mobile-videogpt-finetune-2000",
    output_dir: str = "results/benchmarks",
):
    """
    Run benchmarks with different polling intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*70)
    print("MOBILE-VIDEOGPT POLLING BENCHMARK")
    print("="*70)

    # Check video
    video_info = check_video_file(video_path)
    if not video_info.get("exists"):
        print(f"Error: Video file not found: {video_path}")
        return None

    print(f"\nVideo: {video_path}")
    print(f"  Duration: {video_info['duration_seconds']:.2f}s")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Frames: {video_info['total_frames']}")

    # Memory estimates
    mem_est = estimate_memory_requirements()
    print(f"\nEstimated Memory: {mem_est['total_estimated_gb']:.2f} GB")

    # GPU info
    gpu_info = get_gpu_memory_info()
    if "error" not in gpu_info:
        print(f"GPU: {gpu_info['device']}")
        print(f"  Free: {gpu_info['free_memory_gb']:.2f} GB")

    # Create config and engine
    config = PollingConfig(
        base_model_path=base_model,
        lora_weights_path=lora_weights,
        max_polling_duration=3600,  # 1 hour max for benchmarks
        log_dir="logs/benchmarks",
        output_dir=output_dir,
        log_level="WARNING",  # Reduce logging during benchmark
    )

    engine = PollingInferenceEngine(config)

    # Load model once
    print("\nLoading model...")
    load_start = time.time()
    if not engine.load_model():
        print("Failed to load model")
        return None
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Get GPU memory after loading
    gpu_after_load = get_gpu_memory_info()

    # Run benchmarks for each interval
    all_results = []

    for interval in polling_intervals:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: Polling Interval = {interval}s")
        print(f"{'='*70}")

        config.polling_interval = interval
        engine.stream_handler.reset()
        engine.stream_handler.open_video_file(video_path)
        engine.stream_handler.reset()

        # Run polling
        summary = engine.run_polling_loop(
            video_source=video_path,
            max_polls=num_polls_per_interval,
        )

        if "error" not in summary:
            result = {
                "polling_interval": interval,
                "num_polls": summary["total_polls"],
                "latency_mean_ms": summary["latency_ms"]["mean"],
                "latency_median_ms": summary["latency_ms"]["median"],
                "latency_min_ms": summary["latency_ms"]["min"],
                "latency_max_ms": summary["latency_ms"]["max"],
                "ttft_mean_ms": summary["time_to_first_token_ms"]["mean"],
                "ttft_median_ms": summary["time_to_first_token_ms"]["median"],
                "tokens_per_second": summary["tokens_per_second"]["mean"],
                "total_output_tokens": summary["total_output_tokens"],
            }
            all_results.append(result)

            print(f"\n  Latency (mean): {result['latency_mean_ms']:.2f}ms")
            print(f"  TTFT (mean):    {result['ttft_mean_ms']:.2f}ms")
            print(f"  Tokens/s:       {result['tokens_per_second']:.2f}")

    # Clean up
    engine.cleanup()

    # Generate report
    benchmark_report = {
        "timestamp": timestamp,
        "video_info": video_info,
        "model": {
            "base_model": base_model,
            "lora_weights": lora_weights,
            "load_time_s": load_time,
        },
        "gpu_info": gpu_info,
        "gpu_after_load": gpu_after_load,
        "memory_estimates": mem_est,
        "benchmark_config": {
            "polling_intervals": polling_intervals,
            "num_polls_per_interval": num_polls_per_interval,
        },
        "results": all_results,
    }

    # Save report
    report_file = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(benchmark_report, f, indent=2)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\nReport saved to: {report_file}")

    # Print summary table
    print("\n" + "-"*70)
    print(f"{'Interval':>10} {'Latency':>12} {'TTFT':>12} {'Tokens/s':>12}")
    print(f"{'(s)':>10} {'(ms)':>12} {'(ms)':>12} {'':>12}")
    print("-"*70)

    for r in all_results:
        print(f"{r['polling_interval']:>10} {r['latency_mean_ms']:>12.2f} {r['ttft_mean_ms']:>12.2f} {r['tokens_per_second']:>12.2f}")

    print("-"*70)

    return benchmark_report


def main():
    parser = argparse.ArgumentParser(description="Benchmark polling inference")
    parser.add_argument("video_path", type=str, help="Path to video file")
    parser.add_argument("--intervals", type=str, default="1,2,3,5",
                        help="Comma-separated polling intervals to test")
    parser.add_argument("--polls-per-interval", type=int, default=5,
                        help="Number of polls per interval")
    parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                        help="Output directory for benchmark results")
    parser.add_argument("--base-model", type=str,
                        default="Amshaker/Mobile-VideoGPT-0.5B")
    parser.add_argument("--lora-weights", type=str,
                        default="EdgeVLM-Labs/mobile-videogpt-finetune-2000")

    args = parser.parse_args()

    intervals = [int(x) for x in args.intervals.split(",")]

    run_benchmark(
        video_path=args.video_path,
        polling_intervals=intervals,
        num_polls_per_interval=args.polls_per_interval,
        base_model=args.base_model,
        lora_weights=args.lora_weights,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
