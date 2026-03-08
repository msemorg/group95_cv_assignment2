import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from background import get_foreground_mask, postprocess_foreground_mask
from skimage import measure

# --- 1. VOXEL GRID ---
voxel_size = 0.02  # coarse for testing
xmin, xmax = -1, 1
ymin, ymax = -1, 1
zmin, zmax = -2, 2

xs = np.arange(xmin, xmax, voxel_size)
ys = np.arange(ymin, ymax, voxel_size)
zs = np.arange(zmin, zmax, voxel_size)



X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


occupied = np.ones(len(voxels), dtype=bool)

# --- 2. LOAD CAMERA CONFIGS AND HSV THRESHOLDS ---
cams = []
for i in range(1,5):
    with open(f"data/cam{i}/master_config_{i}.json") as config_file:
        cfg = json.load(config_file)
    K = np.array(cfg["mtx"]["data"]).reshape(3,3).astype(np.float32)
    rvec = np.array(cfg["rvec"]["data"]).reshape(3,1)
    tvec = np.array(cfg["tvec"]["data"]).reshape(3,1)

    with open(f"data/cam{i}/cam{i}_thresholds.json") as threshold_file:
        thresh = json.load(threshold_file)
    H_thresh = thresh["H_Thresh"]
    S_thresh = thresh["S_Thresh"]
    V_thresh = thresh["V_Thresh"]

    cap = cv2.VideoCapture(f"data/cam{i}/video.avi")

    cams.append({
        "K": K,
        "rvec": rvec,
        "tvec": tvec,
        "H": H_thresh,
        "S": S_thresh,
        "V": V_thresh,
        "cap": cap,
        "id": i
    })

    

# --- 3. CARVE VOXELS (single frame per cam for debugging) ---
for cam in cams:
    ret, frame = cam["cap"].read()
    if not ret:
        print(f"Camera {cam['id']}: cannot read frame")
        continue

    # load background model for this camera
    bg_model = cv2.imread(f"data/cam{cam['id']}/bg_model.png")

    # compute mask
    mask_before = get_foreground_mask(
        frame, bg_model, (cam["H"], cam["S"], cam["V"])
    )

    mask_after = postprocess_foreground_mask(mask_before)
    mask = mask_after.astype(bool)

    h, w = mask.shape
    print(f"Camera {cam['id']}: mask coverage = {mask.sum()} pixels ({mask.sum()/mask.size*100:.2f}%)")

    # debug plot of mask
    plt.imshow(mask.astype(np.uint8)*255, cmap='gray')
    plt.title(f"Camera {cam['id']} mask")
    plt.show()

    # project only occupied voxels
    voxels_f = voxels[occupied].astype(np.float32)
    dist_coeffs = np.array(cfg["dist"]["data"]).astype(np.float32)
    pts2d, _ = cv2.projectPoints(voxels_f, cam["rvec"], cam["tvec"], cam["K"], dist_coeffs)
    pts2d = np.round(pts2d.reshape(-1,2)).astype(int)

    inside = (pts2d[:,0]>=0) & (pts2d[:,0]<w) & (pts2d[:,1]>=0) & (pts2d[:,1]<h)
    if inside.sum() == 0:
        print(f"Camera {cam['id']}: no projected points inside the mask!")

    tmp = np.zeros_like(occupied)
    tmp[np.where(occupied)[0][inside]] = mask[pts2d[inside,1], pts2d[inside,0]]
    occupied &= tmp

    ############
    # --- DEBUG: DRAW WORLD AXES ON FRAME ---
    cube_pts = np.float32([
    [xmin, ymin, zmin],
    [xmax, ymin, zmin],
    [xmax, ymax, zmin],
    [xmin, ymax, zmin],
    [xmin, ymin, zmax],
    [xmax, ymin, zmax],
    [xmax, ymax, zmax],
    [xmin, ymax, zmax]
])

    imgpts, _ = cv2.projectPoints(cube_pts, cam["rvec"], cam["tvec"], cam["K"], dist_coeffs)
    imgpts = imgpts.reshape(-1,2).astype(int)

    debug_frame = frame.copy()

    # bottom square
    cv2.line(debug_frame, tuple(imgpts[0]), tuple(imgpts[1]), (255,0,0), 3)
    cv2.line(debug_frame, tuple(imgpts[1]), tuple(imgpts[2]), (255,0,0), 3)
    cv2.line(debug_frame, tuple(imgpts[2]), tuple(imgpts[3]), (255,0,0), 3)
    cv2.line(debug_frame, tuple(imgpts[3]), tuple(imgpts[0]), (255,0,0), 3)

    # top square
    cv2.line(debug_frame, tuple(imgpts[4]), tuple(imgpts[5]), (0,255,0), 3)
    cv2.line(debug_frame, tuple(imgpts[5]), tuple(imgpts[6]), (0,255,0), 3)
    cv2.line(debug_frame, tuple(imgpts[6]), tuple(imgpts[7]), (0,255,0), 3)
    cv2.line(debug_frame, tuple(imgpts[7]), tuple(imgpts[4]), (0,255,0), 3)

    # vertical edges
    for i in range(4):
        cv2.line(debug_frame, tuple(imgpts[i]), tuple(imgpts[i+4]), (0,0,255), 3)



    # --- DRAW WORLD AXES ---

    axis_len = 1.0
    axis_pts = np.float32([
        [0,0,0],            # origin
        [axis_len,0,0],     # X
        [0,axis_len,0],     # Y
        [0,0,axis_len]      # Z
    ])

    axis_img, _ = cv2.projectPoints(axis_pts, cam["rvec"], cam["tvec"], cam["K"], dist_coeffs)
    axis_img = axis_img.reshape(-1,2).astype(int)

    o = tuple(axis_img[0])
    x = tuple(axis_img[1])
    y = tuple(axis_img[2])
    z = tuple(axis_img[3])

    # X = red HORIZONTAL PARALLEL TO CAMERA
    cv2.line(debug_frame, o, x, (0,0,255), 4)

    # Y = green AWAY FROM CAMERA
    cv2.line(debug_frame, o, y, (0,255,0), 4)

    # Z = blue DDOWN
    cv2.line(debug_frame, o, z, (255,0,0), 4)

    plt.imshow(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Camera {cam['id']} 3D Cube Alignment")
    plt.show()
    ############

# release videos
for cam in cams:
    cam["cap"].release()

# --- 4. KEEP ONLY OCCUPIED VOXELS ---
voxels = voxels[occupied]
print(f"Total occupied voxels = {len(voxels)}")

# --- 5. VISUALIZE ---
# --- 4. CONVERT VOXEL MASK BACK TO 3D GRID ---
# Marching cubes needs a structured 3D array (volume), not a list of points
volume = occupied.reshape(len(xs), len(ys), len(zs))

# --- 5. GENERATE MESH (The "Statue") ---
# level=0.5 finds the boundary between 0 (empty) and 1 (occupied)
verts, faces, normals, values = measure.marching_cubes(volume, level=0.5, spacing=(voxel_size, voxel_size, voxel_size))

# Offset vertices to match real-world coordinates
verts += [xmin, ymin, zmin]

# --- 6. VISUALIZE AS A SOLID ---
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy rendering of the "statue" surface
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                cmap='Spectral', lw=0.1, edgecolors='none')

ax.set_title("3D Statue Reconstruction")
plt.show()


