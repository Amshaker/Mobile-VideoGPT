"""
Polling-based inference engine for Mobile-VideoGPT.
Loads LoRA adapters and performs inference at configurable intervals.
"""

import os
import sys
import time
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

# Suppress warnings before imports
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoConfig

# PyTorch optimizations for faster inference
torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels for input shapes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mobilevideogpt.model import MobileVideoGPTQwenForCausalLM
from mobilevideogpt.mm_utils import tokenizer_image_token
from mobilevideogpt.conversation import conv_templates
from mobilevideogpt.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    NUM_FRAMES,
    NUM_CONTEXT_IMAGES,
)

from polling.config import PollingConfig
from polling.metrics import MetricsTracker, InferenceMetrics
from polling.stream_handler import VideoStreamHandler


class FirstTokenStreamer:
    """Helper to capture time to first token during generation."""

    def __init__(self):
        self.first_token_time: Optional[float] = None
        self.start_time: float = 0.0

    def reset(self):
        self.first_token_time = None
        self.start_time = time.time()

    def on_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()

    @property
    def time_to_first_token(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return self.first_token_time - self.start_time


class PollingInferenceEngine:
    """
    Main engine for polling-based streaming inference.

    Loads the model with LoRA adapters and performs inference
    at configurable polling intervals on video streams.
    """

    def __init__(self, config: PollingConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.video_processor = None

        # Metrics tracking
        self.metrics = MetricsTracker(
            log_dir=config.log_dir,
            save_metrics=config.save_metrics,
        )

        # Stream handler
        self.stream_handler = VideoStreamHandler(
            buffer_size=config.frame_buffer_size,
            num_frames=config.num_frames,
            fps=config.fps,
            image_resolution=config.image_resolution,
        )

        # First token streamer
        self._first_token_streamer = FirstTokenStreamer()

        # State
        self._is_loaded = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("PollingInference")
        logger.setLevel(getattr(logging, self.config.log_level))

        # Clear existing handlers
        logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        os.makedirs(self.config.log_dir, exist_ok=True)
        log_file = os.path.join(
            self.config.log_dir,
            f"polling_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        return logger

    def load_model(self) -> bool:
        """
        Load the base model with LoRA adapters from HuggingFace.

        Returns:
            True if model loaded successfully
        """
        self.logger.info("=" * 60)
        self.logger.info("LOADING MODEL WITH LORA ADAPTERS")
        self.logger.info("=" * 60)
        self.logger.info(f"Base model: {self.config.base_model_path}")
        self.logger.info(f"LoRA weights: {self.config.lora_weights_path}")

        load_start = time.time()

        try:
            # Import peft for LoRA
            from peft import PeftModel

            # Setup kwargs for model loading
            kwargs = {}
            if self.config.load_8bit:
                kwargs['load_in_8bit'] = True
            elif self.config.load_4bit:
                from transformers import BitsAndBytesConfig
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for better precision
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
                # Note: Flash Attention incompatible with 4-bit quantization
            else:
                kwargs['torch_dtype'] = torch.bfloat16  # Use bfloat16 for better stability
                try:
                    kwargs['attn_implementation'] = 'flash_attention_2'  # Enable Flash Attention 2
                except:
                    self.logger.warning("Flash Attention 2 not available")

            # Load config from base model (LoRA adapters don't have config.json)
            self.logger.info("Loading model configuration from base model...")
            model_cfg = AutoConfig.from_pretrained(self.config.base_model_path)

            # Load tokenizer from base model
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                use_fast=False
            )
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)

            # Load base model
            self.logger.info("Loading base model...")
            try:
                self.model = MobileVideoGPTQwenForCausalLM.from_pretrained(
                    self.config.base_model_path,
                    low_cpu_mem_usage=False,
                    config=model_cfg,
                    num_select_k_frames_in_chunk=self.config.num_select_k_frames_in_chunk,
                    topk=self.config.topk,
                    **kwargs
                )
            except Exception as quant_error:
                if self.config.load_4bit or self.config.load_8bit:
                    self.logger.warning(f"Quantization failed: {quant_error}")
                    self.logger.info("Retrying without quantization...")
                    # Fallback to FP16 without quantization
                    kwargs = {'torch_dtype': torch.bfloat16}
                    try:
                        kwargs['attn_implementation'] = 'flash_attention_2'
                    except:
                        pass
                    self.model = MobileVideoGPTQwenForCausalLM.from_pretrained(
                        self.config.base_model_path,
                        low_cpu_mem_usage=False,
                        config=model_cfg,
                        num_select_k_frames_in_chunk=self.config.num_select_k_frames_in_chunk,
                        topk=self.config.topk,
                        **kwargs
                    )
                else:
                    raise

            # Resize token embeddings
            token_num, token_dim = self.model.lm_head.out_features, self.model.lm_head.in_features
            if self.model.lm_head.weight.shape[0] != token_num:
                self.model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype)
                )
                self.model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype)
                )

            # Load non-LoRA trainables
            self.logger.info("Loading non-LoRA trainables...")
            non_lora_path = os.path.join(self.config.lora_weights_path, 'non_lora_trainables.bin')

            # Try to load from HuggingFace hub
            from huggingface_hub import hf_hub_download
            try:
                non_lora_local = hf_hub_download(
                    repo_id=self.config.lora_weights_path,
                    filename="non_lora_trainables.bin"
                )
                non_lora_trainables = torch.load(non_lora_local, map_location='cpu')
            except Exception as e:
                self.logger.warning(f"Could not load non_lora_trainables from hub: {e}")
                # Try local path
                if os.path.exists(non_lora_path):
                    non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
                else:
                    self.logger.warning("No non_lora_trainables found, proceeding without")
                    non_lora_trainables = {}

            # Clean up keys
            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v
                    for k, v in non_lora_trainables.items()
                }

            self.model.load_state_dict(non_lora_trainables, strict=False)

            # Load and merge LoRA weights
            self.logger.info("Loading LoRA adapter weights...")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.config.lora_weights_path)

                self.logger.info("Merging LoRA weights...")
                self.model = self.model.merge_and_unload()
                self.logger.info("LoRA adapters loaded and merged successfully")
            except Exception as e:
                self.logger.warning(f"Could not load LoRA adapters: {e}")
                self.logger.info("Proceeding with base model only")

            # Setup special tokens
            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)

            from mobilevideogpt.constants import (
                DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN
            )

            if mm_use_im_patch_token:
                self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            self.model.resize_token_embeddings(len(self.tokenizer))

            # Move to device
            self.model.to(self.config.device)
            self.model.eval()

            # Compile model for faster inference (PyTorch 2.0+)
            try:
                self.logger.info("Compiling model with torch.compile()...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self.logger.info("Model compiled successfully")
            except Exception as e:
                self.logger.warning(f"torch.compile() not available or failed: {e}")
                self.logger.info("Proceeding without compilation")

            # Setup vision processors
            self.logger.info("Setting up vision processors...")
            vision_tower = self.model.get_vision_tower()
            vision_tower.load_model(self.model.config.mm_vision_tower)
            self.video_processor = vision_tower.image_processor

            image_vision_tower = self.model.get_image_vision_tower()
            image_vision_tower.load_model()
            self.image_processor = image_vision_tower.image_processor

            load_time = time.time() - load_start
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            self.logger.info("=" * 60)

            self._is_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    def prepare_prompt(self, prompt: str, slice_len: int) -> torch.Tensor:
        """Prepare the prompt with image tokens."""
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)

        if mm_use_im_start_end:
            from mobilevideogpt.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt

        # Use Qwen2 instruct template
        conv = conv_templates["qwen2_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            formatted_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.config.device)

        return input_ids, conv.sep

    def warmup(self, num_runs: int = 1):
        """
        Perform warmup runs to load model into memory and optimize caching.

        Args:
            num_runs: Number of warmup inference runs
        """
        self.logger.info("="*60)
        self.logger.info(f"Starting warmup with {num_runs} run(s)...")
        self.logger.info("="*60)

        # Create dummy data
        dummy_frames = [
            torch.zeros(
                (3, self.config.image_resolution, self.config.image_resolution),
                dtype=torch.bfloat16,
                device=self.config.device
            )
            for _ in range(self.config.num_frames)
        ]

        dummy_context = [
            torch.zeros(
                (3, self.config.image_resolution, self.config.image_resolution),
                dtype=torch.bfloat16,
                device=self.config.device
            )
            for _ in range(self.config.num_context_images)
        ]

        warmup_prompt = "Analyze this exercise."

        for i in range(num_runs):
            start_time = time.time()
            self.logger.info(f"Warmup run {i+1}/{num_runs}...")

            try:
                _, ttft, _, _ = self.run_single_inference(
                    dummy_frames,
                    dummy_context,
                    warmup_prompt,
                    self.config.num_frames
                )

                warmup_time = time.time() - start_time
                self.logger.info(f"  Completed in {warmup_time:.2f}s (TTFT: {ttft*1000:.1f}ms)")

            except Exception as e:
                self.logger.warning(f"  Warmup run {i+1} failed: {e}")

        self.logger.info("Warmup complete!")
        self.logger.info("="*60)

    def run_single_inference(
        self,
        video_frames: torch.Tensor,
        context_frames: torch.Tensor,
        prompt: str,
        slice_len: int,
    ) -> Tuple[str, float, int, int]:
        """
        Run a single inference on the provided frames.

        Returns:
            Tuple of (response, time_to_first_token, input_tokens, output_tokens)
        """
        # Prepare input
        input_ids, stop_str = self.prepare_prompt(prompt, slice_len)

        # Prepare frames with bfloat16 to match model dtype
        video_tensor = torch.stack(video_frames, dim=0).to(dtype=torch.bfloat16, device=self.config.device)
        context_tensor = torch.stack(context_frames, dim=0).to(dtype=torch.bfloat16, device=self.config.device)

        input_token_count = input_ids.shape[1]

        # Reset first token timer
        self._first_token_streamer.reset()

        # Generate with inference_mode (more efficient than no_grad)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor,
                context_images=context_tensor,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,  # Always use KV cache for faster generation
            )

        # Record first token time (approximate since we can't hook into generate)
        generation_end = time.time()

        # Decode output
        output_tokens = output_ids.shape[1] - input_token_count
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        # Estimate TTFT (first token is roughly 1/output_tokens of total time)
        # This is approximate since transformers doesn't expose per-token timing
        ttft = self._first_token_streamer.time_to_first_token
        if ttft == 0:
            # Estimate: TTFT is typically the encoding time + first decoding step
            ttft = (generation_end - self._first_token_streamer.start_time) / max(output_tokens, 1) * 2

        return response, ttft, input_token_count, output_tokens

    def run_polling_loop(
        self,
        video_source: str,
        prompt: Optional[str] = None,
        on_response: Optional[Callable[[int, str, InferenceMetrics], None]] = None,
        max_polls: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the main polling loop on a video source.

        Args:
            video_source: Path to video file or stream URL
            prompt: Inference prompt (uses config default if None)
            on_response: Callback called after each inference (poll_index, response, metrics)
            max_polls: Maximum number of polls (None = run until video ends or max_duration)

        Returns:
            Session summary dictionary
        """
        if not self._is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return {"error": "Model not loaded"}

        prompt = prompt or self.config.prompt

        self.logger.info("=" * 60)
        self.logger.info("STARTING POLLING LOOP")
        self.logger.info("=" * 60)
        self.logger.info(f"Video source: {video_source}")
        self.logger.info(f"Polling interval: {self.config.polling_interval}s")
        self.logger.info(f"Prompt: {prompt[:100]}...")
        self.logger.info("=" * 60)

        # Open video source
        if os.path.isfile(video_source):
            if not self.stream_handler.open_video_file(video_source):
                return {"error": f"Failed to open video file: {video_source}"}
        else:
            self.stream_handler.start_stream_capture(video_source)

        # Start metrics session
        self.metrics.start_session(
            video_source=video_source,
            prompt=prompt,
            polling_interval=self.config.polling_interval,
        )

        poll_index = 0
        start_time = time.time()

        try:
            while True:
                # Check termination conditions
                elapsed = time.time() - start_time
                if elapsed >= self.config.max_polling_duration:
                    self.logger.info(f"Max polling duration ({self.config.max_polling_duration}s) reached")
                    break

                if max_polls is not None and poll_index >= max_polls:
                    self.logger.info(f"Max polls ({max_polls}) reached")
                    break

                if self.stream_handler.is_exhausted:
                    self.logger.info("Video exhausted")
                    break

                self.logger.info(f"\n{'='*40}")
                self.logger.info(f"POLL #{poll_index + 1}")
                self.logger.info(f"Video position: {self.stream_handler.current_position:.2f}s / {self.stream_handler.total_duration:.2f}s")
                self.logger.info(f"{'='*40}")

                # Start metrics for this inference
                self.metrics.start_inference(poll_index)

                try:
                    # Extract frames from current position
                    frame_start = time.time()
                    video_frames, context_frames, slice_len = self.stream_handler.get_frames_for_inference(
                        self.image_processor,
                        self.video_processor,
                        num_video_frames=self.config.num_frames,
                        num_context_images=self.config.num_context_images,
                        polling_interval=self.config.polling_interval,
                    )
                    frame_time = time.time() - frame_start
                    self.metrics.record_timing("frame_extraction_time", frame_time)

                    if slice_len == 0:
                        self.logger.warning("No frames extracted, end of video reached")
                        break

                    self.logger.info(f"Extracted {slice_len} frames in {frame_time*1000:.1f}ms")

                    # Run inference
                    inference_start = time.time()
                    response, ttft, input_tokens, output_tokens = self.run_single_inference(
                        video_frames, context_frames, prompt, slice_len
                    )
                    inference_time = time.time() - inference_start
                    self.metrics.record_timing("generation_time", inference_time)

                    # Record metrics
                    metrics = self.metrics.end_inference(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        frames_processed=slice_len,
                        buffer_size=len(self.stream_handler.frame_buffer),
                        response=response,
                        time_to_first_token=ttft,
                    )

                    # Log response
                    self.logger.info(f"\n📝 Response:\n{response}\n")

                    # Callback
                    if on_response:
                        on_response(poll_index, response, metrics)

                except Exception as e:
                    import traceback
                    self.metrics.record_error(poll_index, str(e), traceback.format_exc())
                    self.logger.error(f"Inference failed: {e}", exc_info=True)

                poll_index += 1

                # Wait for next poll
                if not self.stream_handler.is_exhausted:
                    self.logger.info(f"Waiting {self.config.polling_interval}s until next poll...")
                    time.sleep(self.config.polling_interval)

        except KeyboardInterrupt:
            self.logger.info("\nPolling interrupted by user")

        finally:
            self.stream_handler.close()

        # End session and get summary
        summary = self.metrics.end_session()

        return summary

    def cleanup(self):
        """Clean up resources."""
        self.stream_handler.close()
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        self.logger.info("Resources cleaned up")
