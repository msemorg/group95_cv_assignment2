import glm
import random
import numpy as np
import cv2
import os
block_size = 1.0

def load_data(cam_id):
    # loads the data from the xml files into a format that opencv can use
    fs = cv2.FileStorage(f'data/cam{cam_id}/config.xml', cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coeffiecients").mat()
    rvec = fs.getNode("rotation_vector").mat()
    tvec = fs.getNode("translation_vector").mat()
    fs.release()
    return camera_matrix, dist_coeffs, rvec, tvec

def create_and_save_masks():
    # background subtraction, creates masks
    for i in range(1, 5):
        # load background and current video frame
        cap_bg = cv2.VideoCapture(f'data/cam{i}/background.avi')
        cap_vid = cv2.VideoCapture(f'data/cam{i}/video.avi')
        
        # grab the first frame of background
        _, background_frame = cap_bg.read()
        _, video_frame = cap_vid.read()
        
        # convert to HSV 
        bg_hsv = cv2.cvtColor(background_frame, cv2.COLOR_BGR2HSV)
        vid_hsv = cv2.cvtColor(video_frame, cv2.COLOR_BGR2HSV)
        
        # calculate absolute difference and threshold 
        diff = cv2.absdiff(bg_hsv, vid_hsv)
        
        # thresholds 
        lower_thresh = np.array([0, 50, 50])
        upper_thresh = np.array([180, 255, 255])
        
        mask = cv2.inRange(diff, lower_thresh, upper_thresh)

        cv2.imwrite(f'data/cam{i}/mask.png', mask)
        
        cap_bg.release()
        cap_vid.release()
#create_and_save_masks()

def foreground_masks():
    # loads the masks created 
    masks = []
    for i in range(1,5):
        mask = cv2.imread(f'data/cam{i}/mask.png', cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks

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
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    masks = foreground_masks()
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
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
