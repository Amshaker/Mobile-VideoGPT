"""
Metrics tracking for polling-based streaming inference.
Tracks latency, time to first token, throughput, and other performance metrics.
"""

import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import statistics
import os


@dataclass
class InferenceMetrics:
    """Metrics for a single inference call."""
    poll_index: int
    timestamp: float

    # Timing metrics (all in seconds)
    frame_extraction_time: float = 0.0
    preprocessing_time: float = 0.0
    encoding_time: float = 0.0
    generation_time: float = 0.0
    time_to_first_token: float = 0.0
    total_inference_time: float = 0.0

    # Token metrics
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0

    # Frame metrics
    frames_processed: int = 0
    buffer_size: int = 0

    # Response
    response_length: int = 0
    response_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionMetrics:
    """Aggregated metrics for a complete polling session."""
    session_id: str
    start_time: float
    end_time: float = 0.0

    # Configuration
    polling_interval: float = 3.0
    video_source: str = ""
    prompt: str = ""

    # Aggregated stats
    total_polls: int = 0
    successful_polls: int = 0
    failed_polls: int = 0

    # Individual inference metrics
    inference_metrics: List[InferenceMetrics] = field(default_factory=list)

    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def add_inference(self, metrics: InferenceMetrics):
        """Add metrics from a single inference."""
        self.inference_metrics.append(metrics)
        self.total_polls += 1
        self.successful_polls += 1

    def add_error(self, poll_index: int, error: str, traceback: str = ""):
        """Record an error."""
        self.errors.append({
            "poll_index": poll_index,
            "timestamp": time.time(),
            "error": error,
            "traceback": traceback
        })
        self.total_polls += 1
        self.failed_polls += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated summary statistics."""
        if not self.inference_metrics:
            return {"error": "No inference metrics recorded"}

        latencies = [m.total_inference_time for m in self.inference_metrics]
        ttft = [m.time_to_first_token for m in self.inference_metrics]
        tokens_per_sec = [m.tokens_per_second for m in self.inference_metrics if m.tokens_per_second > 0]

        return {
            "session_id": self.session_id,
            "duration_seconds": self.end_time - self.start_time if self.end_time else time.time() - self.start_time,
            "video_source": self.video_source,
            "polling_interval": self.polling_interval,

            # Poll statistics
            "total_polls": self.total_polls,
            "successful_polls": self.successful_polls,
            "failed_polls": self.failed_polls,
            "success_rate": self.successful_polls / max(self.total_polls, 1) * 100,

            # Latency statistics (ms)
            "latency_ms": {
                "mean": statistics.mean(latencies) * 1000,
                "median": statistics.median(latencies) * 1000,
                "min": min(latencies) * 1000,
                "max": max(latencies) * 1000,
                "std": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
            },

            # Time to first token (ms)
            "time_to_first_token_ms": {
                "mean": statistics.mean(ttft) * 1000,
                "median": statistics.median(ttft) * 1000,
                "min": min(ttft) * 1000,
                "max": max(ttft) * 1000,
                "std": statistics.stdev(ttft) * 1000 if len(ttft) > 1 else 0,
            },

            # Throughput
            "tokens_per_second": {
                "mean": statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
                "max": max(tokens_per_sec) if tokens_per_sec else 0,
            },

            # Total tokens
            "total_input_tokens": sum(m.input_tokens for m in self.inference_metrics),
            "total_output_tokens": sum(m.output_tokens for m in self.inference_metrics),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "video_source": self.video_source,
            "prompt": self.prompt,
            "polling_interval": self.polling_interval,
            "total_polls": self.total_polls,
            "successful_polls": self.successful_polls,
            "failed_polls": self.failed_polls,
            "inference_metrics": [m.to_dict() for m in self.inference_metrics],
            "errors": self.errors,
            "summary": self.get_summary(),
        }


class MetricsTracker:
    """
    Tracks and logs metrics for polling-based streaming inference.
    """

    def __init__(self, log_dir: str = "logs/polling", save_metrics: bool = True):
        self.log_dir = log_dir
        self.save_metrics = save_metrics
        self.logger = logging.getLogger("MetricsTracker")

        os.makedirs(log_dir, exist_ok=True)

        self.current_session: Optional[SessionMetrics] = None
        self._current_inference_start: float = 0.0
        self._current_inference_metrics: Dict[str, float] = {}

    def start_session(
        self,
        video_source: str,
        prompt: str,
        polling_interval: float
    ) -> str:
        """Start a new metrics tracking session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_session = SessionMetrics(
            session_id=session_id,
            start_time=time.time(),
            video_source=video_source,
            prompt=prompt,
            polling_interval=polling_interval,
        )

        self.logger.info(f"Started metrics session: {session_id}")
        return session_id

    def start_inference(self, poll_index: int):
        """Mark the start of an inference."""
        self._current_inference_start = time.time()
        self._current_inference_metrics = {
            "poll_index": poll_index,
            "timestamp": self._current_inference_start,
        }

    def record_timing(self, metric_name: str, duration: float):
        """Record a timing metric for the current inference."""
        self._current_inference_metrics[metric_name] = duration

    def end_inference(
        self,
        input_tokens: int,
        output_tokens: int,
        frames_processed: int,
        buffer_size: int,
        response: str,
        time_to_first_token: float,
    ) -> InferenceMetrics:
        """Complete the current inference and record metrics."""
        total_time = time.time() - self._current_inference_start

        metrics = InferenceMetrics(
            poll_index=self._current_inference_metrics.get("poll_index", 0),
            timestamp=self._current_inference_metrics.get("timestamp", 0),
            frame_extraction_time=self._current_inference_metrics.get("frame_extraction_time", 0),
            preprocessing_time=self._current_inference_metrics.get("preprocessing_time", 0),
            encoding_time=self._current_inference_metrics.get("encoding_time", 0),
            generation_time=self._current_inference_metrics.get("generation_time", 0),
            time_to_first_token=time_to_first_token,
            total_inference_time=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=output_tokens / total_time if total_time > 0 else 0,
            frames_processed=frames_processed,
            buffer_size=buffer_size,
            response_length=len(response),
            response_preview=response[:200] + "..." if len(response) > 200 else response,
        )

        if self.current_session:
            self.current_session.add_inference(metrics)

        # Log the metrics
        self.logger.info(
            f"Poll #{metrics.poll_index}: "
            f"Latency={metrics.total_inference_time*1000:.1f}ms, "
            f"TTFT={metrics.time_to_first_token*1000:.1f}ms, "
            f"Tokens/s={metrics.tokens_per_second:.1f}, "
            f"Output={metrics.output_tokens} tokens"
        )

        return metrics

    def record_error(self, poll_index: int, error: str, traceback: str = ""):
        """Record an error during inference."""
        if self.current_session:
            self.current_session.add_error(poll_index, error, traceback)
        self.logger.error(f"Poll #{poll_index} failed: {error}")

    def end_session(self) -> Optional[Dict[str, Any]]:
        """End the current session and save metrics."""
        if not self.current_session:
            return None

        self.current_session.end_time = time.time()
        summary = self.current_session.get_summary()

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Session ID: {summary['session_id']}")
        self.logger.info(f"Duration: {summary['duration_seconds']:.2f}s")
        self.logger.info(f"Total Polls: {summary['total_polls']} ({summary['successful_polls']} successful)")
        self.logger.info(f"Mean Latency: {summary['latency_ms']['mean']:.2f}ms")
        self.logger.info(f"Mean TTFT: {summary['time_to_first_token_ms']['mean']:.2f}ms")
        self.logger.info(f"Mean Tokens/s: {summary['tokens_per_second']['mean']:.2f}")
        self.logger.info("=" * 60)

        # Save to file
        if self.save_metrics:
            metrics_file = os.path.join(
                self.log_dir,
                f"metrics_{self.current_session.session_id}.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(self.current_session.to_dict(), f, indent=2)
            self.logger.info(f"Metrics saved to: {metrics_file}")

        return summary

    def get_live_stats(self) -> Dict[str, Any]:
        """Get current live statistics."""
        if not self.current_session or not self.current_session.inference_metrics:
            return {}

        recent = self.current_session.inference_metrics[-10:]  # Last 10 polls
        latencies = [m.total_inference_time for m in recent]

        return {
            "polls_completed": self.current_session.successful_polls,
            "elapsed_time": time.time() - self.current_session.start_time,
            "recent_avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0,
        }
