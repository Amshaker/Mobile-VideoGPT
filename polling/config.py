"""
Configuration for polling-based streaming inference.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PollingConfig:
    """Configuration for polling-based video stream inference."""

    # Model configuration
    base_model_path: str = "Amshaker/Mobile-VideoGPT-0.5B"
    lora_weights_path: str = "EdgeVLM-Labs/mobile-videogpt-finetune-2000"

    # Polling configuration
    polling_interval: float = 3.0  # Seconds between inference calls
    max_polling_duration: float = 300.0  # Maximum total polling duration (5 minutes default)

    # Video processing
    num_frames: int = 16  # Number of frames to sample per inference
    num_context_images: int = 16  # Number of context images
    chunk_size: int = 8  # VideoMamba chunk size
    fps: int = 1  # Frame sampling rate
    image_resolution: int = 224  # Frame resolution

    # Frame buffer configuration
    frame_buffer_size: int = 64  # Maximum frames to keep in buffer
    frame_overlap: float = 0.5  # Overlap ratio between polling windows (0.0 - 1.0)

    # Inference configuration
    prompt: str = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
    max_new_tokens: int = 512
    do_sample: bool = False
    num_beams: int = 1
    use_cache: bool = True

    # Model loading options
    load_4bit: bool = False  # 4-bit incompatible with custom model architecture
    load_8bit: bool = False
    num_select_k_frames_in_chunk: int = 4
    topk: bool = True

    # Device configuration
    device: str = "cuda"
    torch_dtype: str = "float16"

    # Logging configuration
    log_level: str = "INFO"
    log_dir: str = "logs/streaming"
    save_metrics: bool = True

    # Output configuration
    output_dir: str = "results/streaming"
    save_responses: bool = True

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "PollingConfig":
        """Create config from environment variables."""
        return cls(
            base_model_path=os.getenv("BASE_MODEL_PATH", cls.base_model_path),
            lora_weights_path=os.getenv("LORA_WEIGHTS_PATH", cls.lora_weights_path),
            polling_interval=float(os.getenv("POLLING_INTERVAL", cls.polling_interval)),
            max_polling_duration=float(os.getenv("MAX_POLLING_DURATION", cls.max_polling_duration)),
            num_frames=int(os.getenv("NUM_FRAMES", cls.num_frames)),
            device=os.getenv("DEVICE", cls.device),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )
