#!/bin/bash
# Script to re-encode video files to fix codec corruption
# This creates a clean, properly encoded copy of the video

INPUT_VIDEO="$1"

if [ -z "$INPUT_VIDEO" ]; then
    echo "Usage: $0 <input_video>"
    echo "Example: $0 concatenated_exercises_20260110_123251.mp4"
    exit 1
fi

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: File '$INPUT_VIDEO' not found"
    exit 1
fi

# Generate output filename
BASENAME=$(basename "$INPUT_VIDEO" .mp4)
OUTPUT_VIDEO="${BASENAME}_fixed.mp4"

echo "Re-encoding video to fix codec issues..."
echo "Input:  $INPUT_VIDEO"
echo "Output: $OUTPUT_VIDEO"
echo ""

# Re-encode with H.264 codec, AAC audio, and proper keyframe intervals
# -c:v libx264: Use H.264 video codec
# -preset medium: Balance between encoding speed and quality
# -crf 23: Constant Rate Factor (18-28, lower = better quality)
# -g 30: Keyframe every 30 frames (1 second at 30fps)
# -c:a aac: Use AAC audio codec
# -b:a 128k: Audio bitrate
# -movflags +faststart: Enable streaming
# -pix_fmt yuv420p: Pixel format for compatibility

ffmpeg -i "$INPUT_VIDEO" \
    -c:v libx264 \
    -preset medium \
    -crf 23 \
    -g 30 \
    -c:a aac \
    -b:a 128k \
    -movflags +faststart \
    -pix_fmt yuv420p \
    -y \
    "$OUTPUT_VIDEO"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Video successfully re-encoded!"
    echo "New file: $OUTPUT_VIDEO"
    echo ""
    echo "To use the fixed video:"
    echo "1. Move/rename the fixed video to use it"
    echo "2. Or update your Gradio app to select: $OUTPUT_VIDEO"
else
    echo ""
    echo "❌ Error: Re-encoding failed"
    exit 1
fi
