"""
Utility functions for polling-based streaming inference.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.cuda.current_device()
    return {
        "device": torch.cuda.get_device_name(device),
        "total_memory_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
        "allocated_memory_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_memory_gb": torch.cuda.memory_reserved(device) / 1e9,
        "free_memory_gb": (torch.cuda.get_device_properties(device).total_memory -
                          torch.cuda.memory_allocated(device)) / 1e9,
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def save_responses_to_file(
    responses: List[Dict[str, Any]],
    output_dir: str,
    session_id: str
) -> str:
    """Save all responses from a session to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"responses_{session_id}.json")

    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)

    return output_file


def create_response_markdown(
    responses: List[Dict[str, Any]],
    output_dir: str,
    session_id: str,
    video_source: str,
) -> str:
    """Create a markdown report of all responses."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"report_{session_id}.md")

    with open(output_file, 'w') as f:
        f.write(f"# Polling Inference Report\n\n")
        f.write(f"**Session ID:** {session_id}\n\n")
        f.write(f"**Video Source:** {video_source}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        for i, resp in enumerate(responses):
            f.write(f"## Poll #{i + 1}\n\n")
            f.write(f"**Timestamp:** {resp.get('timestamp', 'N/A')}\n\n")
            f.write(f"**Latency:** {resp.get('latency_ms', 0):.1f}ms\n\n")
            f.write(f"**TTFT:** {resp.get('ttft_ms', 0):.1f}ms\n\n")
            f.write(f"### Response\n\n")
            f.write(f"{resp.get('response', 'No response')}\n\n")
            f.write("---\n\n")

    return output_file


class PerformanceProfiler:
    """Simple performance profiler for measuring code sections."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        """Start timing a section."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a section and record the duration."""
        if name not in self._start_times:
            return 0.0

        duration = time.perf_counter() - self._start_times[name]

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

        del self._start_times[name]
        return duration

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all recorded timings."""
        import statistics

        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    "count": len(times),
                    "total_s": sum(times),
                    "mean_ms": statistics.mean(times) * 1000,
                    "min_ms": min(times) * 1000,
                    "max_ms": max(times) * 1000,
                }
                if len(times) > 1:
                    summary[name]["std_ms"] = statistics.stdev(times) * 1000

        return summary

    def print_summary(self):
        """Print a formatted summary of all timings."""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("PERFORMANCE PROFILE")
        print("="*60)

        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Total: {stats['total_s']:.3f}s")
            print(f"  Mean:  {stats['mean_ms']:.2f}ms")
            print(f"  Min:   {stats['min_ms']:.2f}ms")
            print(f"  Max:   {stats['max_ms']:.2f}ms")
            if 'std_ms' in stats:
                print(f"  Std:   {stats['std_ms']:.2f}ms")

        print("="*60 + "\n")


def estimate_memory_requirements(
    num_frames: int = 16,
    num_context_images: int = 16,
    batch_size: int = 1,
    precision: str = "float16",
) -> Dict[str, float]:
    """
    Estimate memory requirements for inference.

    Returns memory estimates in GB.
    """
    bytes_per_element = 2 if precision == "float16" else 4

    # Video frames: B x T x C x H x W
    video_memory = batch_size * num_frames * 3 * 224 * 224 * bytes_per_element

    # Context images: B x N x C x H x W
    context_memory = batch_size * num_context_images * 3 * 224 * 224 * bytes_per_element

    # Model weights (approximate for 0.5B model)
    model_memory = 0.5e9 * bytes_per_element  # Parameters

    # KV cache (approximate)
    kv_cache = 0.5e9  # Rough estimate

    # Activation memory (rough estimate)
    activation_memory = 1e9

    total = video_memory + context_memory + model_memory + kv_cache + activation_memory

    return {
        "video_frames_gb": video_memory / 1e9,
        "context_images_gb": context_memory / 1e9,
        "model_weights_gb": model_memory / 1e9,
        "kv_cache_gb": kv_cache / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_estimated_gb": total / 1e9,
    }


def check_video_file(video_path: str) -> Dict[str, Any]:
    """Check video file properties without loading the full video."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        info = {
            "exists": True,
            "path": video_path,
            "fps": vr.get_avg_fps(),
            "total_frames": len(vr),
            "duration_seconds": len(vr) / vr.get_avg_fps(),
        }

        del vr
        return info

    except Exception as e:
        return {
            "exists": False,
            "path": video_path,
            "error": str(e),
        }


def warmup_model(model, tokenizer, device: str = "cuda"):
    """Perform a warmup inference to initialize CUDA kernels."""
    logging.info("Performing model warmup...")

    # Create dummy inputs
    dummy_input = tokenizer("Hello", return_tensors="pt").input_ids.to(device)

    with torch.inference_mode():
        _ = model.generate(
            dummy_input,
            max_new_tokens=5,
            do_sample=False,
        )

    torch.cuda.synchronize()
    logging.info("Model warmup complete")
