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

def tune_background_subtraction(video_path, background_model):
    cap = cv.VideoCapture(video_path)
    window_name = "Tuning"
    cv.namedWindow(window_name)
    
    cv.createTrackbar("H_Thresh", window_name, 10, 255, lambda x: None)
    cv.createTrackbar("S_Thresh", window_name, 30, 255, lambda x: None)
    cv.createTrackbar("V_Thresh", window_name, 50, 255, lambda x: None)

    # Crucial: Give the OS a moment to actually create the window
    cv.waitKey(100)

    while True:
        # Check if the window was closed by the user clicking 'X'
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
            print(f"Final Settings: H={h_t}, S={s_t}, V={v_t}")
            break
            
    cap.release()
    cv.destroyAllWindows()

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
        print(f"Tuning background subtraction for Camera {i}...")
        bg_model = cv.imread(f"data/cam{i}/bg_model.png")
        #BGR is too sensitive to lighting changes, whereas HSV separates color from brightness which separates easier
        tune_background_subtraction(f"data/cam{i}/video.avi", bg_model)
        # results from tuning: TODO: safe this to some file perhaps?     

        tuned_thresholds = {
            1: (54, 68, 75),
            2: (19, 47, 87),
            3: (14, 31, 73),
            4: (24, 73, 78)
        }