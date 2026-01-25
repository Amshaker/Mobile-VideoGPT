import os
import random
import glob
from pathlib import Path
import subprocess
from datetime import datetime

# Target exercises
target_exercises = {
    "alternating_single_leg_glutes_bridge": "alternating single leg glutes bridge",
    "cat-cow_pose": "cat-cow pose",
    "elbow_plank": "elbow plank",
    "glute_hamstring_walkout": "glute hamstring walkout",
    "glutes_bridge": "glutes bridge",
    "heel_lift": "heel lift",
    "high_plank": "high plank",
    "lunges_leg_out_in_front": "lunges leg out in front",
    "opposite_arm_and_leg_lifts_on_knees": "opposite arm and leg lifts (on knees)",
    "pushups": "pushups",
    "side_plank": "side plank",
    "squats": "squats",
    "toe_touch": "toe touch",
    "tricep_stretch": "tricep stretch"
}

def normalize_exercise_name(name):
    """Normalize exercise name for folder matching"""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

def find_exercise_videos(base_path, exercise_folder_name):
    """Find all videos for a given exercise"""
    # Search directly in dataset/exercise_folder
    exercise_path = os.path.join(base_path, "dataset", exercise_folder_name)

    videos = []

    # Search for video files in the exercise folder
    if os.path.exists(exercise_path):
        videos.extend(glob.glob(os.path.join(exercise_path, "*.mp4")))
        videos.extend(glob.glob(os.path.join(exercise_path, "*.avi")))
        videos.extend(glob.glob(os.path.join(exercise_path, "*.mov")))

    return videos

def select_random_videos(base_path):
    """Select one random video from each exercise"""
    selected_videos = []

    print("="*80)
    print("SELECTING RANDOM VIDEOS FROM EACH EXERCISE")
    print("="*80)

    for folder_name, display_name in target_exercises.items():
        videos = find_exercise_videos(base_path, folder_name)

        if videos:
            selected_video = random.choice(videos)
            selected_videos.append({
                'path': selected_video,
                'exercise': display_name,
                'folder': folder_name
            })
            print(f"\n✓ {display_name}")
            print(f"  Found {len(videos)} videos")
            print(f"  Selected: {os.path.basename(selected_video)}")
        else:
            print(f"\n✗ {display_name}")
            print(f"  No videos found in folder: {folder_name}")

    return selected_videos

def create_file_list(videos, output_dir):
    """Create a text file listing all videos for ffmpeg concat"""
    list_file = os.path.join(output_dir, "video_list.txt")

    with open(list_file, 'w') as f:
        for video in videos:
            # FFmpeg requires absolute paths with forward slashes and proper escaping
            video_path = video['path'].replace('\\', '/')
            f.write(f"file '{video_path}'\n")

    return list_file

def concatenate_videos_ffmpeg(videos, output_path):
    """Concatenate videos using ffmpeg"""
    print("\n" + "="*80)
    print("CONCATENATING VIDEOS")
    print("="*80)

    # Create temp directory for file list
    temp_dir = os.path.dirname(output_path)
    list_file = create_file_list(videos, temp_dir)

    print(f"\nCreated file list: {list_file}")
    print(f"Output path: {output_path}")
    print(f"\nConcatenating {len(videos)} videos...")

    # Use ffmpeg to concatenate
    # -f concat: use concat demuxer
    # -safe 0: allow absolute paths
    # -i: input file list
    # -c copy: copy streams without re-encoding (faster)
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        '-y',  # Overwrite output file if it exists
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\n✓ Concatenation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ FFmpeg error: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

        # Try with re-encoding if copy fails
        print("\nRetrying with re-encoding...")
        cmd_reencode = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264',  # Re-encode video
            '-c:a', 'aac',      # Re-encode audio
            '-y',
            output_path
        ]

        try:
            result = subprocess.run(cmd_reencode, capture_output=True, text=True, check=True)
            print("\n✓ Concatenation successful (with re-encoding)!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"\n✗ Re-encoding also failed: {e2}")
            print(f"stderr: {e2.stderr}")
            return False
    except FileNotFoundError:
        print("\n✗ Error: ffmpeg not found. Please install ffmpeg and add it to PATH.")
        print("   Download from: https://ffmpeg.org/download.html")
        return False

def main():
    # Base path - get the project root directory (parent of sample_videos)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)  # Go up one level to project root

    # Output directory - sample_videos folder
    output_dir = os.path.join(base_path, "sample_videos")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Project root: {base_path}")
    print(f"Output directory: {output_dir}")

    # Select random videos
    selected_videos = select_random_videos(base_path)

    if not selected_videos:
        print("\n✗ No videos found. Cannot create concatenated video.")
        return

    if len(selected_videos) < len(target_exercises):
        print(f"\n⚠ Warning: Only found {len(selected_videos)} out of {len(target_exercises)} exercises")
        response = input("Continue with available videos? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"concatenated_exercises_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Concatenate videos
    success = concatenate_videos_ffmpeg(selected_videos, output_path)

    if success:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nConcatenated video saved to:")
        print(f"  {output_path}")
        print(f"\nExercises included ({len(selected_videos)}):")
        for i, video in enumerate(selected_videos, 1):
            print(f"  {i}. {video['exercise']}")

        # Get file size
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\nOutput file size: {file_size_mb:.2f} MB")
    else:
        print("\n✗ Failed to create concatenated video.")

    # Save manifest
    manifest_path = os.path.join(output_dir, f"manifest_{timestamp}.txt")
    with open(manifest_path, 'w') as f:
        f.write(f"Concatenated Video Manifest\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output: {output_filename}\n")
        f.write(f"\n{'='*80}\n\n")
        f.write(f"Videos included ({len(selected_videos)}):\n\n")
        for i, video in enumerate(selected_videos, 1):
            f.write(f"{i}. {video['exercise']}\n")
            f.write(f"   File: {video['path']}\n\n")

    print(f"\nManifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
