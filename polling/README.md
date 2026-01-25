# Polling-based Streaming Inference for Mobile-VideoGPT

This module provides real-time video stream analysis using a polling approach, where inference is performed at configurable intervals on video content.

Instead of interrupt-based inference, this module polls the video stream every N seconds (configurable) and runs inference on the most recent frames. This approach is simpler to implement and works well for exercise form evaluation where feedback doesn't need to be instantaneous.

## Features

- **Configurable polling interval** (default: 3 seconds)
- **LoRA adapter support** - Loads fine-tuned adapters from HuggingFace
- **Comprehensive metrics tracking** - Latency, TTFT, throughput
- **Support for video files and live streams** (webcam, RTSP)
- **Detailed logging** with timestamps
- **Benchmark mode** for performance testing

## Files

| File                       | Description                               |
| -------------------------- | ----------------------------------------- |
| `config.py`                | Configuration dataclass with all settings |
| `metrics.py`               | Metrics tracking and session statistics   |
| `stream_handler.py`        | Video/stream frame extraction             |
| `inference_engine.py`      | Main inference engine with LoRA loading   |
| `run_polling.py`           | Python entry point script                 |
| `run_polling_inference.sh` | Shell script wrapper with logging         |
| `benchmark.py`             | Performance benchmarking script           |
| `utils.py`                 | Utility functions                         |

## Quick Start

### Using the Shell Script

```bash
# Basic usage with default 3s polling
./polling/run_polling_inference.sh sample_videos/00000340.mp4

# Custom polling interval (5 seconds)
./polling/run_polling_inference.sh sample_videos/00000340.mp4 --polling-interval 5

# Limit number of polls
./polling/run_polling_inference.sh sample_videos/00000340.mp4 --max-polls 10

# Use webcam (camera index 0)
./polling/run_polling_inference.sh 0 --polling-interval 2

# With 4-bit quantization for lower memory usage
./polling/run_polling_inference.sh sample_videos/00000340.mp4 --load-4bit
```

### Using Python Directly

```bash
python polling/run_polling.py sample_videos/00000340.mp4 \
    --polling-interval 3 \
    --max-duration 60 \
    --lora-weights EdgeVLM-Labs/mobile-videogpt-finetune-2000
```

### Running Benchmarks

```bash
bash polling/run_polling_inference.sh sample_videos/test_stream.mp4 -polling-interval 3 --num-frames 16 --max-new-tokens 64 --warmup-runs 1
```

### Sample

```bash
python polling/benchmark.py sample_videos/00000340.mp4 \
    --intervals 1,2,3,5 \
    --polls-per-interval 5
```

## Configuration Options

| Option               | Default                                    | Description                              |
| -------------------- | ------------------------------------------ | ---------------------------------------- |
| `--polling-interval` | 3                                          | Seconds between inference calls          |
| `--max-duration`     | 300                                        | Maximum total polling duration (seconds) |
| `--max-polls`        | unlimited                                  | Maximum number of polls                  |
| `--num-frames`       | 16                                         | Frames to sample per inference           |
| `--fps`              | 1                                          | Frame sampling rate                      |
| `--max-new-tokens`   | 512                                        | Max tokens to generate                   |
| `--load-4bit`        | -                                          | Load model in 4-bit quantization         |
| `--load-8bit`        | -                                          | Load model in 8-bit quantization         |
| `--base-model`       | Amshaker/Mobile-VideoGPT-0.5B              | Base model path                          |
| `--lora-weights`     | EdgeVLM-Labs/mobile-videogpt-finetune-2000 | LoRA weights path                        |
| `--log-level`        | INFO                                       | Logging verbosity                        |


## Metrics Tracked

### Per-Inference Metrics

- **Total inference time** - End-to-end latency
- **Time to first token (TTFT)** - Time until first output token
- **Frame extraction time** - Time to extract and preprocess frames
- **Tokens per second** - Generation throughput
- **Input/output token counts**

### Session Metrics

- Mean, median, min, max, std for all timing metrics
- Total polls, success rate
- GPU memory usage

## Output Files

Logs and results are saved to:

- `logs/streaming/polling_YYYYMMDD_HHMMSS.log` - Detailed logs
- `logs/streaming/metrics_YYYYMMDD_HHMMSS.json` - Per-poll metrics
- `results/streaming/summary_YYYYMMDD_HHMMSS.json` - Session summary

## Inference Prompt

Default prompt for exercise evaluation:

> "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

Custom prompts can be specified via `--prompt` option.
