import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

cam_positions = []
cam_directions = []

for i in range(1, 5):
    fs_ext = cv.FileStorage(
        f"data/cam{i}/calculated_extrinsics.xml", cv.FileStorage_READ
    )
    tvec = fs_ext.getNode("translation_vector").mat()
    rvec = fs_ext.getNode("rotation_vector").mat()
    fs_ext.release()

    # Camera position in world coordinates
    R, _ = cv.Rodrigues(rvec)
    cam_pos = -R.T @ tvec  # camera center in world frame
    cam_positions.append(cam_pos.flatten())

    # Camera forward direction (Z axis of camera in world frame)
    forward = R.T @ np.array([0, 0, 1], dtype=np.float64)
    cam_directions.append(forward.flatten())

cam_positions = np.array(cam_positions)
cam_directions = np.array(cam_directions)

# Plot cameras
ax.scatter(
    cam_positions[:, 0],
    cam_positions[:, 1],
    cam_positions[:, 2],
    c="r",
    s=50,
    label="Cameras",
)

# Plot direction arrows
for i, (pos, dir_vec) in enumerate(zip(cam_positions, cam_directions)):
    ax.quiver(
        pos[0],
        pos[1],
        pos[2],
        dir_vec[0],
        dir_vec[1],
        dir_vec[2],
        length=0.5,
        color="b",
    )
    ax.text(pos[0], pos[1], pos[2], f"Cam {i+1}", color="black")

# Chessboard origin
ax.scatter(0, 0, 0, c="k", marker="x", s=100, label="Origin")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Camera Positions and Look Directions")
ax.view_init(elev=30, azim=45)
ax.legend()
plt.show()
