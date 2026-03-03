import os
import json
import numpy as np
import cv2 as cv

# Grid dimensions
width, height, depth = 128, 64, 128
SQUARE_SIZE = 250 # mm
TUNE_BACKGROUND_SUBTRACTION = True # Set to True to enable interactive tuning of background subtraction thresholds

# Create ranges centered around (0,0,0) on the floor
x_range = np.arange(-width//2, width//2) * SQUARE_SIZE
y_range = np.arange(-height, 0) * SQUARE_SIZE # Negative because engine uses Y-up
z_range = np.arange(-depth//2, depth//2) * SQUARE_SIZE

# Create the 3D grid of points
z, y, x = np.meshgrid(z_range, y_range, x_range, indexing='ij')
voxel_coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1).astype(np.float32)
# This function creates a background model by taking the median of all frames' pixel values.
def create_background_model(video_path):
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    # Calculate median across all frames for a clean background
    return np.median(frames, axis=0).astype(np.uint8)

def get_foreground_mask(frame, background_model, thresholds):
    # Convert both frame and background to HSV
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_bg = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)

    # Calculate absolute difference for each channel
    diff = cv.absdiff(hsv_frame, hsv_bg)
    h_diff, s_diff, v_diff = cv.split(diff)

    # Apply thresholds for H, S, and V
    # 'thresholds' should be a list/tuple: (h_thresh, s_thresh, v_thresh)
    _, mask_h = cv.threshold(h_diff, thresholds[0], 255, cv.THRESH_BINARY)
    _, mask_s = cv.threshold(s_diff, thresholds[1], 255, cv.THRESH_BINARY)
    _, mask_v = cv.threshold(v_diff, thresholds[2], 255, cv.THRESH_BINARY)

    # Combine channels: pixel is foreground if any channel shows a significant difference
    foreground_mask = cv.bitwise_or(mask_h, cv.bitwise_or(mask_s, mask_v))

    # Post-processing: Morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    
    # Erode to remove small noise dots
    foreground_mask = cv.erode(foreground_mask, kernel, iterations=1)
    
    # Dilate to fill small holes in the person's silhouette
    foreground_mask = cv.dilate(foreground_mask, kernel, iterations=2)

    return foreground_mask

def tune_background_subtraction(video_path, background_model, cam_id, save_dir):
    cap = cv.VideoCapture(video_path)
    window_name = f"Tuning_Cam_{cam_id}"
    cv.namedWindow(window_name)

    cv.createTrackbar("H_Thresh", window_name, 10, 255, lambda x: None)
    cv.createTrackbar("S_Thresh", window_name, 30, 255, lambda x: None)
    cv.createTrackbar("V_Thresh", window_name, 50, 255, lambda x: None)

    cv.waitKey(100)

    final_thresholds = None
    final_mask = None

    while True:
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        h_t = cv.getTrackbarPos("H_Thresh", window_name)
        s_t = cv.getTrackbarPos("S_Thresh", window_name)
        v_t = cv.getTrackbarPos("V_Thresh", window_name)

        mask = get_foreground_mask(frame, background_model, (h_t, s_t, v_t))

        cv.imshow("Foreground Mask", mask)
        cv.imshow("Original", frame)

        if cv.waitKey(30) & 0xFF == ord('q'):
            final_thresholds = (h_t, s_t, v_t)
            final_mask = mask.copy()
            break

    cap.release()
    cv.destroyAllWindows()

    if final_thresholds is not None:
        os.makedirs(save_dir, exist_ok=True)

        mask_path = os.path.join(save_dir, f"cam{cam_id}_mask.png")
        settings_path = os.path.join(save_dir, f"cam{cam_id}_thresholds.json")

        cv.imwrite(mask_path, final_mask)

        with open(settings_path, "w") as f:
            json.dump({
                "H_Thresh": final_thresholds[0],
                "S_Thresh": final_thresholds[1],
                "V_Thresh": final_thresholds[2]
            }, f, indent=4)

        return final_thresholds

    return None

def load_camera_params(path):
    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"Failed to open {path}")
        return None

    # Retrieve the matrices
    # .mat() is used to get the data as a numpy array
    mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    rvec = fs.getNode("rotation_vector").mat()
    tvec = fs.getNode("translation_vector").mat()

    fs.release()
    return mtx, dist, rvec, tvec



#calculate background model for each camera and save it as an image for later use in foreground extraction
video_count = 4
for i in range(1, video_count + 1):
    bg_path = f"data/cam{i}/background.avi"
    model = create_background_model(bg_path)
    cv.imwrite(f"data/cam{i}/bg_model.png", model)

         # Run this once for each camera to find your H, S, V values
    if TUNE_BACKGROUND_SUBTRACTION:
        #BGR is too sensitive to lighting changes, whereas HSV separates color from brightness which separates easier
        bg_model = cv.imread(f"data/cam{i}/bg_model.png")
        thresholds = tune_background_subtraction(
            f"data/cam{i}/video.avi",
            bg_model,
            cam_id=i,
            save_dir="data/tuned_settings"
        )