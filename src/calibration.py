import cv2
import os
# create images from the intrinsics video for each camera, to be used for calibration
# 'False' as the images are already created, and manually filtered through.
CREATE_VIDEO_STILLS = True

def clear_folder(folder_path):
    """Removes all files in the directory."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def extract_frames(video_path, output_folder, interval_seconds):
    # Extract frames from a video at specified intervals and save them to an output folder.
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        clear_folder(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate how many frames to skip
    frame_interval = int(fps * interval_seconds)
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only save if we hit the interval
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved: {filename}")

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} images.")

# Example usage for Camera 1
video_count = 4
for i in range(1, video_count + 1):
    if CREATE_VIDEO_STILLS == True:
        video_path = f"data/cam{i}/intrinsics.avi"
        output_folder = f"data/cam{i}/intrinsics_frames_{i}"
        print(video_path, output_folder)
        extract_frames(video_path, output_folder, interval_seconds=3)
