import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from background import get_foreground_mask

# --- 1. VOXEL GRID ---
voxel_size = 0.02  # coarse for testing
xmin, xmax = -1, 1
ymin, ymax = -1, 1
zmin, zmax = 0, 2

xs = np.arange(xmin, xmax, voxel_size)
ys = np.arange(ymin, ymax, voxel_size)
zs = np.arange(zmin, zmax, voxel_size)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
voxels = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
occupied = np.ones(len(voxels), dtype=bool)

# --- 2. LOAD CAMERA CONFIGS AND HSV THRESHOLDS ---
cams = []
for i in range(1,5):
    with open(f"data/cam{i}/master_config_{i}.json") as f:
        cfg = json.load(f)
    K = np.array(cfg["mtx"]["data"]).reshape(3,3).astype(np.float32)
    rvec = np.array(cfg["rvec"]["data"]).reshape(3,1)
    tvec = np.array(cfg["tvec"]["data"]).reshape(3,1)

    with open(f"data/tuned_settings/cam{i}_thresholds.json") as f:
        thresh = json.load(f)
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

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    # load background model for this camera
    bg_model = cv2.imread(f"data/cam{i}/bg_model.png")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_bg = cv2.cvtColor(bg_model, cv2.COLOR_BGR2HSV)

    diff = cv2.absdiff(hsv_frame, hsv_bg)
    h_diff, s_diff, v_diff = cv2.split(diff)

    # thresholds from tuned settings
    mask_h = (h_diff > H_thresh)
    mask_s = (s_diff > S_thresh)
    mask_v = (v_diff > V_thresh)

    mask = mask_h | mask_s | mask_v
    _, mask_after = get_foreground_mask(frame, bg_model, (H_thresh, S_thresh, V_thresh))
    mask = mask.astype(bool)

    h, w = mask.shape

    print(f"Camera {cam['id']}: mask coverage = {mask.sum()} pixels ({mask.sum()/mask.size*100:.2f}%)")
    
    # debug plot of mask
    plt.imshow(mask.astype(np.uint8)*255, cmap='gray')
    plt.title(f"Camera {cam['id']} mask")
    plt.show()

    # project only occupied voxels
    voxels_f = voxels[occupied].astype(np.float32)
    pts2d, _ = cv2.projectPoints(voxels_f, cam["rvec"], cam["tvec"], cam["K"], distCoeffs=None)
    pts2d = np.round(pts2d.reshape(-1,2)).astype(int)

    # check some projected points
    print(f"Camera {cam['id']}: sample projected points (first 10) = {pts2d[:10]}")

    inside = (pts2d[:,0]>=0) & (pts2d[:,0]<w) & (pts2d[:,1]>=0) & (pts2d[:,1]<h)
    if inside.sum() == 0:
        print(f"Camera {cam['id']}: no projected points inside the mask!")

    tmp = np.zeros_like(occupied)
    tmp[np.where(occupied)[0][inside]] = mask[pts2d[inside,1], pts2d[inside,0]]
    occupied &= tmp

# release videos
for cam in cams:
    cam["cap"].release()

# --- 4. KEEP ONLY OCCUPIED VOXELS ---
voxels = voxels[occupied]
print(f"Total occupied voxels = {len(voxels)}")

# --- 5. VISUALIZE ---
if len(voxels) > 0:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(voxels[:,0], voxels[:,1], voxels[:,2], s=5)
    plt.show()
else:
    print("No voxels survived! Check thresholds, mask, and voxel bounding box.")