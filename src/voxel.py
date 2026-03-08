import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

# --- 1. VOXEL GRID ---
voxel_size = 0.2
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
    # config
    with open(f"data/cam{i}/master_config_{i}.json") as f:
        cfg = json.load(f)
    K = np.array(cfg["mtx"]["data"]).reshape(3,3).astype(np.float32)
    rvec = np.array(cfg["rvec"]["data"]).reshape(3,1)
    tvec = np.array(cfg["tvec"]["data"]).reshape(3,1)

    print('config done')

    # HSV thresholds
    with open(f"data/tuned_settings/cam{i}_thresholds.json") as f:
        thresh = json.load(f)
    H_thresh = thresh["H_Thresh"]
    S_thresh = thresh["S_Thresh"]
    V_thresh = thresh["V_Thresh"]

    print('thresholds loading done')
    # video
    cap = cv2.VideoCapture(f"data/cam{i}/video.avi")

    print("frame getting from video done")
    cams.append({
        "K": K,
        "rvec": rvec,
        "tvec": tvec,
        "H": H_thresh,
        "S": S_thresh,
        "V": V_thresh,
        "cap": cap
    })

# --- 3. CARVE VOXELS FOR ALL FRAMES ---
done = False
while not done:
    done = True
    print("loading carving")
    for cam in range(1) :
        ret, frame = cam["cap"].read()
        if not ret:
            continue
        done = False  # at least one cam has more frames

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (hsv[:,:,0] > cam["H"]) & (hsv[:,:,1] > cam["S"]) & (hsv[:,:,2] > cam["V"])
        mask = mask.astype(bool)
        h, w = mask.shape

        pts2d, _ = cv2.projectPoints(voxels, cam["rvec"], cam["tvec"], cam["K"], distCoeffs=None)
        pts2d = np.round(pts2d.reshape(-1,2)).astype(int)

        inside = (pts2d[:,0]>=0) & (pts2d[:,0]<w) & (pts2d[:,1]>=0) & (pts2d[:,1]<h)
        tmp = np.zeros_like(occupied)
        tmp[inside] = mask[pts2d[inside,1], pts2d[inside,0]]
        occupied &= tmp

# release videos
for cam in cams:
    cam["cap"].release()
    print("released video")

# --- 4. KEEP ONLY OCCUPIED VOXELS ---
voxels = voxels[occupied]

# --- 5. VISUALIZE ---
print("starting visualization")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(voxels[:,0], voxels[:,1], voxels[:,2], s=1)
plt.show()

print('done')