import glm
import random
import numpy as np
import cv2
import os
from src.voxel import build_lookup_table, reconstruct_voxels, convert_to_render_format, load_camera_parameters,generate_voxel_grid
block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):

    print("[DEBUG] Starting full reconstruction pipeline...")

    camera_params = load_camera_parameters()
    voxels = generate_voxel_grid(width, height, depth)

    silhouettes = []
    for cam_id in range(1, 5):
        silhouette_path = f"data/tuned_settings/cam{cam_id}_mask.png"        
        sil = cv2.imread(silhouette_path, 0)
        silhouettes.append(sil)

    lookup_table = build_lookup_table(
        voxels,
        camera_params,
        silhouettes[0].shape
    )

    active_voxels = reconstruct_voxels(
        voxels,
        lookup_table,
        silhouettes
    )

    data, colors = convert_to_render_format(
        active_voxels,
        width,
        height,
        depth
    )

    print("[DEBUG] Reconstruction complete.")
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
