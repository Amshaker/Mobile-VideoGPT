#!/bin/bash
#
# Polling-based Streaming Inference Script for Mobile-VideoGPT
# This script loads LoRA adapters from HuggingFace and runs polling inference on video streams
#
# Usage:
#   ./run_polling_inference.sh <video_source> [options]
#
# Examples:
#   ./run_polling_inference.sh sample_videos/00000340.mp4
#   ./run_polling_inference.sh sample_videos/00000340.mp4 --polling-interval 5
#   ./run_polling_inference.sh 0  # Webcam
#

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Model paths
BASE_MODEL_PATH="Amshaker/Mobile-VideoGPT-0.5B"
LORA_WEIGHTS_PATH="EdgeVLM-Labs/mobile-videogpt-finetune-2000"

# Default polling configuration (can be overridden via command line)
POLLING_INTERVAL="${POLLING_INTERVAL:-3}"      # seconds between polls
MAX_DURATION="${MAX_DURATION:-300}"            # maximum total duration in seconds
MAX_POLLS="${MAX_POLLS:-}"                     # maximum number of polls (empty = unlimited)

# Video processing
NUM_FRAMES="${NUM_FRAMES:-16}"                 # frames per inference
FPS="${FPS:-1}"                                # frame sampling rate
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"        # max generation tokens
WARMUP_RUNS="${WARMUP_RUNS:-0}"                # number of warmup runs

# Inference prompt
PROMPT="${PROMPT:-Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?}"

# Output directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_ROOT}/logs/polling"
OUTPUT_DIR="${PROJECT_ROOT}/results/polling"

# Logging configuration
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/polling_${TIMESTAMP}.log"

# ═══════════════════════════════════════════════════════════════════════════════
# Functions
# ═══════════════════════════════════════════════════════════════════════════════

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                              ║"
    echo "║        Mobile-VideoGPT Polling Inference with LoRA Adapters                  ║"
    echo "║                                                                              ║"
    echo "║        Real-time Exercise Form Evaluation                                    ║"
    echo "║                                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

print_config() {
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "Configuration"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Model Configuration:"
    echo "    Base Model:     ${BASE_MODEL_PATH}"
    echo "    LoRA Weights:   ${LORA_WEIGHTS_PATH}"
    echo ""
    echo "  Polling Configuration:"
    echo "    Interval:       ${POLLING_INTERVAL}s"
    echo "    Max Duration:   ${MAX_DURATION}s"
    echo "    Max Polls:      ${MAX_POLLS:-unlimited}"
    echo ""
    echo "  Video Processing:"
    echo "    Num Frames:     ${NUM_FRAMES}"
    echo "    FPS:            ${FPS}"
    echo "    Max Tokens:     ${MAX_NEW_TOKENS}"
    echo ""
    echo "  Output:"
    echo "    Log File:       ${LOG_FILE}"
    echo "    Results Dir:    ${OUTPUT_DIR}"
    echo ""
    echo "  Prompt:"
    echo "    ${PROMPT:0:70}..."
    echo ""
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo ""
}

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] [${level}] ${message}" | tee -a "${LOG_FILE}"
}

check_dependencies() {
    log "INFO" "Checking dependencies..."

    # Check Python
    if ! command -v python &> /dev/null; then
        log "ERROR" "Python is not installed"
        exit 1
    fi

    # Check required Python packages
    python -c "import torch" 2>/dev/null || {
        log "ERROR" "PyTorch is not installed"
        exit 1
    }

    python -c "import transformers" 2>/dev/null || {
        log "ERROR" "Transformers is not installed"
        exit 1
    }

    python -c "import peft" 2>/dev/null || {
        log "ERROR" "PEFT is not installed. Install with: pip install peft"
        exit 1
    }

    python -c "import decord" 2>/dev/null || {
        log "ERROR" "Decord is not installed. Install with: pip install decord"
        exit 1
    }

    log "INFO" "All dependencies satisfied"
}

setup_environment() {
    log "INFO" "Setting up environment..."

    # Activate conda environment
    CONDA_ENV="mobile_videogpt"
    log "INFO" "Activating conda environment: ${CONDA_ENV}"

    # Initialize conda for bash
    eval "$(conda shell.bash hook)" || {
        log "ERROR" "Failed to initialize conda"
        exit 1
    }

    # Activate environment
    conda activate "${CONDA_ENV}" || {
        log "ERROR" "Failed to activate conda environment: ${CONDA_ENV}"
        log "ERROR" "Please create it first with: conda create -n mobile_videogpt python=3.10"
        exit 1
    }

    log "INFO" "Conda environment activated: ${CONDA_ENV}"

    # Create directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    # Set Python path
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

    # Suppress warnings
    export PYTHONWARNINGS='ignore'
    export TOKENIZERS_PARALLELISM=false

    # GPU settings
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

    log "INFO" "Environment configured"
}

usage() {
    echo "Usage: $0 <video_source> [options]"
    echo ""
    echo "Arguments:"
    echo "  video_source       Path to video file or stream URL (0 for webcam)"
    echo ""
    echo "Options:"
    echo "  --polling-interval SECONDS   Time between polls (default: 3)"
    echo "  --max-duration SECONDS       Maximum total duration (default: 300)"
    echo "  --max-polls NUM              Maximum number of polls"
    echo "  --num-frames NUM             Frames per inference (default: 16)"
    echo "  --fps NUM                    Frame sampling rate (default: 1)"
    echo "  --max-new-tokens NUM         Max generation tokens (default: 512)"
    echo "  --warmup-runs NUM            Number of warmup runs (default: 0)"
    echo "  --prompt TEXT                Custom inference prompt"
    echo "  --load-4bit                  Load model in 4-bit quantization"
    echo "  --load-8bit                  Load model in 8-bit quantization"
    echo "  --log-level LEVEL            Logging level (DEBUG, INFO, WARNING, ERROR)"
    echo "  --base-model PATH            HuggingFace path to base model"
    echo "  --lora-weights PATH          HuggingFace path to LoRA weights"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 sample_videos/00000340.mp4"
    echo "  $0 sample_videos/00000340.mp4 --polling-interval 5 --max-polls 10"
    echo "  $0 0 --polling-interval 2  # Webcam with 2s interval"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# Parse Arguments
# ═══════════════════════════════════════════════════════════════════════════════

# Check for help
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Check for video source
if [[ -z "$1" ]]; then
    echo "Error: Video source is required"
    echo ""
    usage
    exit 1
fi

VIDEO_SOURCE="$1"
shift

# Additional arguments for Python script
EXTRA_ARGS=""
LOAD_QUANTIZED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --polling-interval)
            POLLING_INTERVAL="$2"
            shift 2
            ;;
        --max-duration)
            MAX_DURATION="$2"
            shift 2
            ;;
        --max-polls)
            MAX_POLLS="$2"
            shift 2
            ;;
        --num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --load-4bit)
            LOAD_QUANTIZED="--load-4bit"
            shift
            ;;
        --load-8bit)
            LOAD_QUANTIZED="--load-8bit"
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --base-model)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --lora-weights)
            LORA_WEIGHTS_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════════════

# Print banner
print_banner

# Setup
setup_environment
check_dependencies

# Print configuration
print_config

# Build Python command
PYTHON_CMD="python ${PROJECT_ROOT}/polling/run_polling.py"
PYTHON_CMD+=" \"${VIDEO_SOURCE}\""
PYTHON_CMD+=" --base-model \"${BASE_MODEL_PATH}\""
PYTHON_CMD+=" --lora-weights \"${LORA_WEIGHTS_PATH}\""
PYTHON_CMD+=" --polling-interval ${POLLING_INTERVAL}"
PYTHON_CMD+=" --max-duration ${MAX_DURATION}"
PYTHON_CMD+=" --num-frames ${NUM_FRAMES}"
PYTHON_CMD+=" --fps ${FPS}"
PYTHON_CMD+=" --max-new-tokens ${MAX_NEW_TOKENS}"
PYTHON_CMD+=" --warmup-runs ${WARMUP_RUNS}"
PYTHON_CMD+=" --prompt \"${PROMPT}\""
PYTHON_CMD+=" --log-dir \"${LOG_DIR}\""
PYTHON_CMD+=" --output-dir \"${OUTPUT_DIR}\""
PYTHON_CMD+=" --log-level ${LOG_LEVEL}"

if [[ -n "${MAX_POLLS}" ]]; then
    PYTHON_CMD+=" --max-polls ${MAX_POLLS}"
fi

if [[ -n "${LOAD_QUANTIZED}" ]]; then
    PYTHON_CMD+=" ${LOAD_QUANTIZED}"
fi

# Log the command
log "INFO" "Starting polling inference..."
log "INFO" "Video source: ${VIDEO_SOURCE}"
log "DEBUG" "Command: ${PYTHON_CMD}"

# Execute
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Starting Inference"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

# Run the Python script and capture output
eval "${PYTHON_CMD}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "Execution Complete"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "  Total Runtime:   ${DURATION}s"
echo "  Exit Code:       ${EXIT_CODE}"
echo "  Log File:        ${LOG_FILE}"
echo "  Results Dir:     ${OUTPUT_DIR}"
echo ""

if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "INFO" "Polling inference completed successfully"
else
    log "ERROR" "Polling inference failed with exit code ${EXIT_CODE}"
fi

exit ${EXIT_CODE}
