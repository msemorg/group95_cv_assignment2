import glm
import random
import numpy as np
import cv2
import os
block_size = 1.0

#---------------------------------------DATA---------------------------------------

def load_camera_parameters(data_path="data"):
    # loads configuration 
    print("[DEBUG] Loading camera parameters...")

    camera_params = []

    for cam_id in range(1, 5):
        config_path = os.path.join(data_path, f"cam{cam_id}", "config.xml")
        print(f"[DEBUG] Reading {config_path}")

        fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)

        camera_matrix = fs.getNode("camera_matrix").mat()
        dist_coeffs = fs.getNode("distortion_coefficients").mat()
        rvec = fs.getNode("rotation_vector").mat()
        tvec = fs.getNode("translation_vector").mat()

        fs.release()

        camera_params.append({
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "rvec": rvec,
            "tvec": tvec
        })

    print("[DEBUG] Camera parameters loaded successfully.")
    return camera_params

#---------------------------------------GRID---------------------------------------

def generate_voxel_grid(width=128, height=64, depth=128):
    # generates 3d voxel grid
    print("[DEBUG] Generating adjusted voxel grid...")

    voxels = []

    # Smaller cube around checkerboard origin
    scale = 0.3   # shrink world
    z_offset = 6  # move forward near camera scene

    for x in range(width):
        for y in range(height):
            for z in range(depth):

                world_x = (x - width/2) * scale
                world_y = y * scale
                world_z = (z - depth/2) * scale + z_offset

                voxels.append([world_x, world_y, world_z])

    voxels = np.array(voxels, dtype=np.float32)
    print(f"[DEBUG] Total voxels generated: {len(voxels)}")

    return voxels

#---------------------------------------LOOKUP TABLE---------------------------------------

def build_lookup_table(voxels, camera_params, image_shape):
    # projects all voxels into the 2d imdage plane of each camera and builds a lookup table
    print("[DEBUG] Building lookup table...")

    height, width = image_shape

    lookup_table = []

    for cam_idx, cam in enumerate(camera_params):
        print(f"[DEBUG] Processing camera {cam_idx+1}")

        projected_points, _ = cv2.projectPoints(
            voxels,
            cam["rvec"],
            cam["tvec"],
            cam["camera_matrix"],
            cam["dist_coeffs"]
        )

        projected_points = projected_points.reshape(-1, 2)
        # create a mask that marks wheter the projected point lies witin boundaries
        valid_mask = (
            (projected_points[:, 0] >= 0) &
            (projected_points[:, 0] < width) &
            (projected_points[:, 1] >= 0) &
            (projected_points[:, 1] < height)
        )
        # stores projected points and mask in dictionary
        lookup_table.append({
            "points": projected_points.astype(np.int32),
            "valid_mask": valid_mask
        })

        print(f"[DEBUG] Camera {cam_idx+1}: {np.sum(valid_mask)} valid projections")

    print("[DEBUG] Lookup table built successfully.")
    return lookup_table


#---------------------------------------RECONSTRUCTION---------------------------------------

def reconstruct_voxels(voxels, lookup_table, silhouettes):
    # performs voxel carving, removes voxels that are not consistent with silloutte
    print("[DEBUG] Starting voxel reconstruction")

    active_voxels = []
    total_voxels = len(voxels)

    used_cameras = [0, 1,  3] # i think somehting might be wrong with cam 3 so i excluded it 
    for voxel_idx in range(total_voxels):

        keep_voxel = True

        for cam_idx in used_cameras:

            # check if projection is valid
            if not lookup_table[cam_idx]["valid_mask"][voxel_idx]:
                keep_voxel = False
                break

            x, y = lookup_table[cam_idx]["points"][voxel_idx]

            # bounds safety 
            h, w = silhouettes[cam_idx].shape
            if x < 0 or x >= w or y < 0 or y >= h:
                keep_voxel = False
                break

            if silhouettes[cam_idx][y, x] == 0:
                keep_voxel = False
                break

        if keep_voxel:
            active_voxels.append(voxels[voxel_idx])

        if voxel_idx % 100000 == 0:
            print(f"[DEBUG] Processed {voxel_idx}/{total_voxels} voxels")

    print(f"[DEBUG] Active voxels after carving: {len(active_voxels)}")
    return active_voxels


def convert_to_render_format(active_voxels, width=128, height=64, depth=128):
    # prepares the voxel poistions and colors for rendering 
    print("[DEBUG] Converting voxels to render format...")

    data = []
    colors = []

    for voxel in active_voxels:
        data.append(voxel.tolist())

        # simple color mapping by height
        y_norm = voxel[1] / height
        colors.append([y_norm, 0.3, 1 - y_norm])

    print(f"[DEBUG] Prepared {len(data)} voxels for rendering.")

    return data, colors
