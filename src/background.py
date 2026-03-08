import os
import json
import numpy as np
import cv2 as cv

# Grid dimensions
width, height, depth = 128, 64, 128
SQUARE_SIZE = 115  # mm
TUNE_BACKGROUND_SUBTRACTION = False # Set to True to enable interactive tuning of background subtraction thresholds
# Create ranges centered around (0,0,0) on the floor
x_range = np.arange(-width // 2, width // 2) * SQUARE_SIZE
y_range = np.arange(-height, 0) * SQUARE_SIZE  # Negative because engine uses Y-up
z_range = np.arange(-depth // 2, depth // 2) * SQUARE_SIZE

# todo add gaussian for extra points
# Create the 3D grid of points
z, y, x = np.meshgrid(z_range, y_range, x_range, indexing="ij")
voxel_coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1).astype(np.float32)


# This function creates a background model by taking the median of all frames' pixel values.
def create_background_model(video_path):
    """creates a background model by taking the median of all frames' pixel values.
    Args: video path (str)
    returns:
    np.ndarray: background image"""
    cap = cv.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # Calculate median across all frames for a clean background
    return np.median(frames, axis=0).astype(np.uint8)


def get_foreground_mask(frame, background_model, thresholds):
    """
    Computes the initial foreground mask based on HSV differences.

    Args:
        frame (np.ndarray): Current BGR frame.
        background_model (np.ndarray): Background image in BGR.
        thresholds (tuple[int, int, int]): HSV thresholds.

    Returns:
        np.ndarray: Binary mask before post-processing.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_bg = cv.cvtColor(background_model, cv.COLOR_BGR2HSV)

    diff = cv.absdiff(hsv_frame, hsv_bg)
    h_diff, s_diff, v_diff = cv.split(diff)

    _, mask_h = cv.threshold(h_diff, thresholds[0], 255, cv.THRESH_BINARY)
    _, mask_s = cv.threshold(s_diff, thresholds[1], 255, cv.THRESH_BINARY)
    _, mask_v = cv.threshold(v_diff, thresholds[2], 255, cv.THRESH_BINARY)

    mask_before = cv.bitwise_or(mask_h, cv.bitwise_or(mask_s, mask_v))
    return mask_before


def postprocess_foreground_mask(mask_before):
    """
    Cleans up and refines the initial foreground mask.

    Args:
        mask_before (np.ndarray): Initial binary mask.

    Returns:
        np.ndarray: Post-processed mask keeping only the largest blob and smoothing edges.
    """
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask_processed = cv.morphologyEx(mask_before, cv.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
        mask_processed, connectivity=8
    )
    mask_after = np.zeros_like(mask_processed)

    if num_labels > 1:
        blob_areas = stats[1:, cv.CC_STAT_AREA]
        largest_label = 1 + np.argmax(blob_areas)
        mask_after[labels == largest_label] = 255

    mask_after = cv.dilate(mask_after, kernel, iterations=1)
    return mask_after


def tune_background_subtraction(video_path, background_model, cam_id, save_dir):
    """provides an interactive interface to tune the HSV values
       Args:
        video_path (str): Path to the video used for tuning.
        background_model (np.ndarray): Background image for comparison.
        cam_id (int): Camera identifier for naming saved files.
        save_dir (str): Directory to save tuned thresholds and mask images.

    Returns:
        tuple[int, int, int] | None: The final H, S, V thresholds if tuning is completed, else None.
    """
    cap = cv.VideoCapture(video_path)
    window_name = f"Tuning_Cam_{cam_id}"
    cv.namedWindow(window_name)

    cv.createTrackbar("H_Thresh", window_name, 10, 255, lambda x: None)
    cv.createTrackbar("S_Thresh", window_name, 30, 255, lambda x: None)
    cv.createTrackbar("V_Thresh", window_name, 50, 255, lambda x: None)

    final_thresholds = None
    final_mask_before = None
    final_mask_after = None

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            continue

        h_t = cv.getTrackbarPos("H_Thresh", window_name)
        s_t = cv.getTrackbarPos("S_Thresh", window_name)
        v_t = cv.getTrackbarPos("V_Thresh", window_name)

        # Get initial mask
        m_before = get_foreground_mask(frame, background_model, (h_t, s_t, v_t))
        
        # Post-process the mask
        m_after = postprocess_foreground_mask(m_before)

        cv.imshow("Foreground Mask (After)", m_after)
        cv.imshow("Original", frame)

        if cv.waitKey(30) & 0xFF == ord("q"):
            final_thresholds = (h_t, s_t, v_t)
            final_mask_before = m_before.copy()
            final_mask_after = m_after.copy()
            break

    cap.release()
    cv.destroyAllWindows()

    if final_thresholds is not None:
        os.makedirs(save_dir, exist_ok=True)

        # Saving both versions
        cv.imwrite(
            os.path.join(save_dir, f"cam{cam_id}_mask_before_postprocessing.png"),
            final_mask_before,
        )
        cv.imwrite(
            os.path.join(save_dir, f"cam{cam_id}_mask_after_postprocessing.png"),
            final_mask_after,
        )

        settings_path = os.path.join(save_dir, f"cam{cam_id}_thresholds.json")
        with open(settings_path, "w") as f:
            json.dump(
                {
                    "H_Thresh": final_thresholds[0],
                    "S_Thresh": final_thresholds[1],
                    "V_Thresh": final_thresholds[2],
                },
                f,
                indent=4,
            )

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


# calculate background model for each camera and save it as an image for later use in foreground extraction
video_count = 4
for i in range(1, video_count + 1):
    bg_path = f"data/cam{i}/background.avi"
    model = create_background_model(bg_path)
    cv.imwrite(f"data/cam{i}/bg_model.png", model)

    # Run this once for each camera to find the H, S, V cut-off point
    if TUNE_BACKGROUND_SUBTRACTION:
        # BGR is too sensitive to lighting changes, whereas HSV separates color from brightness which separates easier
        bg_model = cv.imread(f"data/cam{i}/bg_model.png")
        thresholds = tune_background_subtraction(
            f"data/cam{i}/video.avi",
            bg_model,
            cam_id=i,
            save_dir = f"data/cam{i}"
        )
