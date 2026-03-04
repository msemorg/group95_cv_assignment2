import numpy as np
import cv2 as cv
import os
import json
from background import get_foreground_mask

# --- 1. GLOBAL GRID DEFINITION ---
# --- RE-CENTERED GRID ---
VOXEL_SIZE = 0.2  # Keep 2cm resolution
grid_dim_x = 140    # Wider to catch arms
grid_dim_y = 240   # Taller for full body
grid_dim_z = 140    # Deeper

center_x = 0.5     # Shift left from 0.8
center_y = 0.5     # Shift up (was 0.0)
center_z = 0.8     # Push back (was 0.0)

x_range = (np.arange(-grid_dim_x // 2, grid_dim_x // 2) * VOXEL_SIZE) + center_x
y_range = (np.arange(-grid_dim_y // 2, grid_dim_y // 2) * VOXEL_SIZE) + center_y
z_range = (np.arange(-grid_dim_z // 2, grid_dim_z // 2) * VOXEL_SIZE) + center_z


z, y, x = np.meshgrid(z_range, y_range, x_range, indexing="ij")
voxel_coords = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1).astype(np.float32)


def load_camera_params_combined(cam_idx):
    """
    Loads intrinsics from JSON and extrinsics from XML.
    Ensures data types are correct for OpenCV projection.
    """
    intrinsics_path = f"data/cam{cam_idx}/calculated_intrinsics_{cam_idx}.json"
    extrinsics_path = f"data/cam{cam_idx}/calculated_extrinsics.xml"

    if not os.path.exists(intrinsics_path) or not os.path.exists(extrinsics_path):
        print(f"Missing calibration files for cam {cam_idx}")
        return None

    # Load Intrinsics
    with open(intrinsics_path, "r") as f:
        data = json.load(f)
    mtx = np.array(data["camera_matrix"], np.float32)
    dist = np.array(data["dist_coeffs"], np.float32)

    # Load Extrinsics
    fs = cv.FileStorage(extrinsics_path, cv.FILE_STORAGE_READ)
    rvec = fs.getNode("rotation_vector").mat()
    tvec = fs.getNode("translation_vector").mat()
    fs.release()

    if rvec is None or tvec is None:
        print(f"Error: Extrinsics for cam {cam_idx} are empty!")
        return None

    return mtx, dist, rvec.astype(np.float32), tvec.astype(np.float32)

def reconstruct_voxels(voxel_coords, masks, camera_params):
    # Initialize a counter for how many cameras see each voxel
    votes = np.zeros(len(voxel_coords), dtype=np.int32)

    for i, (mask, params) in enumerate(zip(masks, camera_params)):
        mtx, dist, rvec, tvec = params
        
        # Project 3D points to 2D
        img_points, _ = cv.projectPoints(voxel_coords, rvec, tvec, mtx, dist)
        img_points = img_points.reshape(-1, 2)

        h, w = mask.shape
        x_p, y_p = img_points[:, 0], img_points[:, 1]

        # 1. Identify points that actually fall within the image boundaries
        valid_idx = (x_p >= 0) & (x_p < w) & (y_p >= 0) & (y_p < h)
        
        # 2. Initialize occupancy for THIS camera as all False
        current_mask_occupancy = np.zeros(len(voxel_coords), dtype=bool)
        
        # 3. Fill in the occupancy for valid points
        coords_to_query = img_points[valid_idx].astype(np.int32)
        current_mask_occupancy[valid_idx] = mask[coords_to_query[:, 1], coords_to_query[:, 0]] > 0
        
        # 4. Add this camera's "vote" to the global counter
        votes += current_mask_occupancy.astype(np.int32)


        # DEBUG: Visual Check
        debug_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        
        # Filter for points that are strictly inside image boundaries 
        # to avoid the "Can't parse center" error
        h, w = mask.shape
        x_p, y_p = img_points[:, 0], img_points[:, 1]
        on_screen = (x_p >= 0) & (x_p < w) & (y_p >= 0) & (y_p < h)
        visible_points = img_points[on_screen]
        
        # Draw every 50th point for better density visualization
        for p in visible_points[::50]: 
            cv.circle(debug_mask, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        
        cv.imshow(f"Debug Cam {i}", debug_mask)
        cv.waitKey(1) # Refresh window

    cv.waitKey(1000) # Give you a second to look at the windows
    
    # CRITICAL: Start with a lower threshold (e.g., 2 or 3) to see if ANY cameras overlap
    # If this works but 'votes == 4' doesn't, your calibration is slightly off.
    threshold = len(camera_params) # Change this to 2 or 3 if you see nothing
    return votes >= threshold


# --- EXECUTION FLOW ---

camera_params = []
masks = []
frames = []

print("Loading camera data and generating masks...")
for i in range(1, 5):
    params = load_camera_params_combined(i)
    if params is None:
        continue
    camera_params.append(params)
    
    # Get current frame
    cap = cv.VideoCapture(f"data/cam{i}/video.avi")
    ret, frame = cap.read() 
    if not ret:
        print(f"Failed to read video for cam {i}")
        continue
    frames.append(frame)
    cap.release()
    
    # Load foreground mask settings
    bg_model = cv.imread(f"data/cam{i}/bg_model.png")
    if bg_model is None:
        print(f"Missing bg_model for cam {i}")
        continue

    with open(f"data/tuned_settings/cam{i}_thresholds.json", 'r') as f:
        t = json.load(f)
    
    # Process mask
    _, mask = get_foreground_mask(frame, bg_model, (t['H_Thresh'], t['S_Thresh'], t['V_Thresh']))
    masks.append(mask)

# Run Reconstruction
if len(camera_params) > 0:
    print(f"Reconstructing from {len(camera_params)} cameras...")
    occupied_indices = reconstruct_voxels(voxel_coords, masks, camera_params)
    active_voxels = voxel_coords[occupied_indices]
    print(f"Done! Created {len(active_voxels)} active voxels.")
else:
    print("No camera parameters loaded. Check your file paths.")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_voxels_matplotlib(active_voxels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot: x, y, z
    # s is the size of the points
    ax.scatter(active_voxels[:, 0], active_voxels[:, 2], active_voxels[:, 1], 
               c=active_voxels[:, 1], cmap='viridis', s=2)

    ax.set_xlabel('X (Meters)')
    ax.set_ylabel('Z (Meters)')
    ax.set_zlabel('Y (Meters)')
    ax.set_title(f"Reconstructed Voxel Cloud ({len(active_voxels)} points)")
    
    # Equalize the axes so it doesn't look squashed
    max_range = np.array([active_voxels[:,0].max()-active_voxels[:,0].min(), 
                          active_voxels[:,1].max()-active_voxels[:,1].min(), 
                          active_voxels[:,2].max()-active_voxels[:,2].min()]).max() / 2.0

    mid_x = (active_voxels[:,0].max()+active_voxels[:,0].min()) * 0.5
    mid_y = (active_voxels[:,1].max()+active_voxels[:,1].min()) * 0.5
    mid_z = (active_voxels[:,2].max()+active_voxels[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)

    plt.show()

# Call this after your reconstruction
plot_voxels_matplotlib(active_voxels)