import cv2 as cv
import os
from pathlib import Path
import glob
import numpy as np
import json
import shutil

CREATE_VIDEO_STILLS = False
MANUAL_CALIBRATION = False
SHOW_IMAGES = True
if SHOW_IMAGES == False:
    print(
        "\033[31mimages are not being displayed. Set SHOW_IMAGES to True to see the pictures\033[0m"
    )

NX, NY = 9, 6  # inner corners of chessboard in assignment 2
SQUARE_SIZE = 0.20  # TODO this is a guessed value for assignment 2. The true value might not be needed, in that case this variable will be removed later.
SIZE = (NX, NY)

manual_detections_count = 0
automatic_detections_count = 0
data_to_save = []
camera_positions = []


def clear_folder(folder_path):
    """Removes all files in the directory folder_path."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def extract_frames(video_path, output_folder, interval_seconds):
    """Extract frames from a video at specified intervals and save them to an output folder."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        #if output folder already exists, clear it to avoid confusion with old frames
        clear_folder(output_folder)

    # capture video
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv.CAP_PROP_FPS)
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
            cv.imwrite(filename, frame)
            saved_count += 1
            print(f"Saved: {filename}")

        frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} images.")


def detect_corners_manual(img, fname):
    display = img.copy()
    clicked = []
    instruction = "Click 4 corners: TopLeft, TopRight, BottomRight, BottomLeft"

    def mouse_cb(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append((x, y))

    cv.namedWindow("Manual", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Manual", mouse_cb)
    while len(clicked) < 4:
        tmp = display.copy()
        for pt in clicked:
            cv.circle(tmp, pt, 2, (0, 0, 255), -1)
        cv.putText(
            tmp, instruction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )
        cv.imshow("Manual", tmp)
        cv.waitKey(1)
    cv.destroyAllWindows()

    # 4 object corners
    obj4 = np.array(
        [
            [0, 0],
            [(NX - 1) * SQUARE_SIZE, 0],
            [(NX - 1) * SQUARE_SIZE, (NY - 1) * SQUARE_SIZE],
            [0, (NY - 1) * SQUARE_SIZE],
        ],
        np.float32,
    )
    H, _ = cv.findHomography(obj4, np.array(clicked, np.float32))
    # generate full grid of inner corners using homography
    grid = np.zeros((NX * NY, 2), np.float32)
    grid[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * SQUARE_SIZE
    ones = np.ones((grid.shape[0], 1), np.float32)
    grid_h = np.hstack([grid, ones])
    proj = (H @ grid_h.T).T
    proj = proj[:, :2] / proj[:, 2:]
    corners = proj.reshape(-1, 1, 2).astype(np.float32)
    return corners, grid

# not needed, but I prefer it for readability and modularity, and it also allows for future expansion if needed
def calibrate_camera(objpoints, imgpoints, image_shape):
    return cv.calibrateCamera(objpoints, imgpoints, image_shape[::-1], None, None)[1:3]


def save_extrinsics_xml(cam_idx, rvec, tvec):
    """Saves only the extrinsics (rotation and translation vectors) to an xml."""
    # Updated file path naming as per your TODO
    file_path = f"data/cam{cam_idx}/calculated_extrinsics.xml"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    fs = cv.FileStorage(file_path, cv.FILE_STORAGE_WRITE)
    fs.write("rotation_vector", rvec)
    fs.write("translation_vector", tvec)
    fs.release()
    print(f"Successfully saved Extrinsics XML: {file_path}")

def calculate_extrinsics(cam_idx, mtx, dist):
    """Calculates extrinsics or loads them from XML if the file already exists."""
    xml_path = f"data/cam{cam_idx}/calculated_extrinsics.xml"
    
    # Set global numpy printing options for cleaner console output
    np.set_printoptions(precision=4, suppress=True)
    
    rvec, tvec = None, None

    # 1. Load or Calculate rvec and tvec
    if Path(xml_path).exists() and not MANUAL_CALIBRATION:
        print(f"\n--- Camera {cam_idx} (Loaded from XML) ---")
        fs = cv.FileStorage(xml_path, cv.FILE_STORAGE_READ)
        rvec = fs.getNode("rotation_vector").mat()
        tvec = fs.getNode("translation_vector").mat()
        fs.release()
    else:
        print(f"\n--- Camera {cam_idx} (New Calculation) ---")
        cap = cv.VideoCapture(f"data/cam{cam_idx}/checkerboard.avi")
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, SIZE, None)
        
        if not found:
            corners, _ = detect_corners_manual(frame, f"Cam {cam_idx} Extrinsics")
        else:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        objp = np.zeros((NX * NY, 3), np.float32)
        objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * SQUARE_SIZE
        
        success, rvec, tvec = cv.solvePnP(objp, corners, mtx, dist)
        if success:
            save_extrinsics_xml(cam_idx, rvec, tvec)

    # 2. Extract and Print Rotation Matrix and Translation
    if rvec is not None and tvec is not None:
        # Convert rvec to 3x3 Rotation Matrix
        R, _ = cv.Rodrigues(rvec)
        
        print(f"Rotation Matrix (R):\n{R}")
        print(f"Translation Vector (t):\n{tvec}")
        
        # Calculate full Projection Matrix for Task 3
        P = get_extrinsics_as_matrix(mtx, rvec, tvec)
        
        # Visual Verification
        if SHOW_IMAGES:
            # We need a frame for verification; reload if we loaded from XML
            cap = cv.VideoCapture(f"data/cam{cam_idx}/checkerboard.avi")
            _, frame = cap.read()
            cap.release()
            
            axis = np.float32([[3*SQUARE_SIZE,0,0], [0,3*SQUARE_SIZE,0], [0,0,-3*SQUARE_SIZE]]).reshape(-1,3)
            # We need the corners for the origin point in visualization
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, corners = cv.findChessboardCorners(gray, SIZE, None)
            
            if corners is not None:
                imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
                origin = tuple(corners[0].ravel().astype(int))
                for i, col in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
                    cv.line(frame, origin, tuple(imgpts[i].ravel().astype(int)), col, 5)
                cv.imshow(f"Cam {cam_idx} Verify", frame)
                cv.waitKey(1)
            
        return P
    
    return None

def get_extrinsics_as_matrix(mtx, rvec, tvec):
    """
    Calculates the 3x4 Projection Matrix P = K * [R | t].
    Used to project 3D voxels onto the 2D image plane.
    """
    # 1. Convert the rotation vector to a 3x3 rotation matrix
    R, _ = cv.Rodrigues(rvec)
    
    # 2. Create the 3x4 extrinsic matrix [R | t]
    # We stack R and tvec horizontally
    extrinsic_matrix = np.hstack((R, tvec.reshape(3, 1)))
    
    # 3. Calculate P = K * [R | t]
    projection_matrix = np.dot(mtx, extrinsic_matrix)
    
    
    return projection_matrix

# Main loop
video_count = 4
for i in range(1, video_count + 1):
    video_path = f"data/cam{i}/intrinsics.avi" # provided file to base the intrinsics calibration on
    images_path = f"data/cam{i}/intrinsics_frames_{i}"  # created folder to save the extracted frames from the videos

    # step 1: create video stills from intrinsics.avi
    if CREATE_VIDEO_STILLS == True:
        extract_frames(video_path, images_path, interval_seconds=3)

    image_files = glob.glob(os.path.join(images_path, "*.jpg"))

    # step 2: manually detect corners and calibrate camera, using code from assignment 1
    #TODO rename to intrinsiics_data, also remove the 'images' in the json
    #TODO remove the old image masks
    #TODO add more post-processing
    #TODO make a post-processing image compared to the pre-processed image

    SAVE_JSON = f"data/cam{i}/calculated_intrinsics_{i}.json"
    data_to_save = []
    objpoints_list, imgpoints_list = [], []

    if Path(SAVE_JSON).exists() and not MANUAL_CALIBRATION:
        print("Loading existing calibration from JSON...")

        with open(SAVE_JSON, "r") as f:
            saved_data = json.load(f)

        mtx = np.array(saved_data["camera_matrix"], np.float32)
        dist = np.array(saved_data["dist_coeffs"], np.float32)

    else:
        print("Running new calibration...")

        data_to_save = []
        objpoints_list, imgpoints_list = [], []

        for fname in image_files:
            img = cv.imread(fname)
            if img is None or img.size == 0:
                print(f"Failed to load {fname}")
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

           
            found, corners = cv.findChessboardCorners(gray, SIZE, None)

            if found:
                # Refine and use automatic points
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                obj_2d = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2) * SQUARE_SIZE
                automatic_detections_count += 1
            else:
                # Only manual click if automatic fails
                print(f"Auto-detect failed for {fname}. Clicking manually...")
                corners, obj_2d = detect_corners_manual(img, fname)
                manual_detections_count += 1
            
            objp_i = np.zeros((NX * NY, 3), np.float32)
            objp_i[:, :2] = obj_2d
            objpoints_list.append(objp_i)
            imgpoints_list.append(corners)

        # Calibrate
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        mtx, dist = calibrate_camera(objpoints_list, imgpoints_list, gray.shape)

        # Save the instrinsics to JSON
        full_data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeffs": dist.tolist()
        }
        with open(SAVE_JSON, "w") as f:
            json.dump(full_data, f, indent=4)
        print(f"Saved corner & calibration data to {SAVE_JSON}")

    # step 3: get camera extrinsics (position and orientation) using solvePnP for each image, and save the extrinsics to a JSON file.
    calculate_extrinsics(i, mtx, dist)
