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

# ----------------POST PROCESSING----------------
def keep_largest_blob(mask):
    # remives the other "blobs" in the image
    # find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    # assume that the largest blob is the person
    largest_label = 1 + np.argmax(areas)

    clean_mask = np.zeros_like(mask)
    clean_mask[labels == largest_label] = 255

    return clean_mask


def remove_shadows(frame, mask, mean):
    # removes shadows that get detected as foreground
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    H = hsv[:,:,0]
    S = hsv[:,:,1]
    V = hsv[:,:,2]

    H_bg = mean[:,:,0]
    S_bg = mean[:,:,1]
    V_bg = mean[:,:,2]

    # shadow conditions
    hue_diff = np.abs(H - H_bg) < 10
    sat_diff = np.abs(S - S_bg) < 40

    # shadow = darker but not black
    value_ratio = V / (V_bg + 1e-5)
    darker = (value_ratio > 0.5) & (value_ratio < 0.95)

    shadow = hue_diff & sat_diff & darker

    # remove shadows from foreground
    mask[shadow] = 0

    return mask

# ----------------BACKGROUND SUBTRACTION----------------
def create_and_save_masks():
    # creates foregorund masks for all cameras 
    for i in range(1,5):
        if not os.path.exists(f"data/cam{i}/gmm_mean.npy"):
            train_gmm_background(i)
        # gaussian parameters
        mean = np.load(f"data/cam{i}/gmm_mean.npy")
        var = np.load(f"data/cam{i}/gmm_var.npy")

        cap_vid = cv2.VideoCapture(f"data/cam{i}/video.avi")
        ret, frame = cap_vid.read()

        mask = gmm_foreground_mask(frame, mean, var)

        # post processing
        mask = remove_shadows(frame, mask, mean)
        kernel_small = np.ones((3,3), np.uint8)
        kernel_big = np.ones((7,7), np.uint8)
        # remove salt noise 
        mask = cv2.medianBlur(mask, 5)
        # remove small blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # fill holes in person can also be kept out, but then small holes are present soo idk
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        # deletes all the extra noise and only keeps the "big blob"
        mask = keep_largest_blob(mask)
        # smoothing edges
        mask = cv2.GaussianBlur(mask, (5,5), 0)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"data/cam{i}/mask.png", mask)

        cap_vid.release()


def train_gmm_background(cam_id, num_frames=120):
    # builds background model using the background videos
    cap = cv2.VideoCapture(f"data/cam{cam_id}/background.avi")

    frames = []
    count = 0

    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frames.append(hsv.astype(np.float32))
        count += 1

    cap.release()

    frames = np.stack(frames, axis=0)

    # Gaussian parameters 
    mean = np.mean(frames, axis=0)
    variance = np.var(frames, axis=0)

    # stabilize variance, we dont want division by tiny numbers 
    variance = np.maximum(variance, 25.0)
    
    np.save(f"data/cam{cam_id}/gmm_mean.npy", mean)
    np.save(f"data/cam{cam_id}/gmm_var.npy", variance)


def gmm_foreground_mask(frame, mean, var, threshold=6.0):
    # detecrts foreground pixels using gaussian distance

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # ignore hue 
    hsv = hsv[:,:,1:]
    mean = mean[:,:,1:]
    var = var[:,:,1:]

    diff = hsv - mean

    # stabilized variance
    var = np.maximum(var, 25.0)

    # mahalanobis like distance
    dist = (diff ** 2) / var
    dist_sum = np.sum(dist, axis=2)
    # if it is far from the background --> should be foreground
    mask = dist_sum > threshold

    return (mask.astype(np.uint8) * 255)


def foreground_masks():
    # saves masks into list and into png
    masks = []
    for i in range(1,5):
        mask = cv2.imread(f'data/cam{i}/mask.png', cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks


# --------VOXEL RECONSTRUCTION--------
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



create_and_save_masks()
