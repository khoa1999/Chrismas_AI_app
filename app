#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:56:51 2024

@author: dangkhoa
"""
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
# Define 3D model points of key facial landmarks
model_points = np.array([
    [0.0, 0.0, 0.0],         # Nose tip
    [0.0, -330.0, -65.0],    # Chin
    [-225.0, 170.0, -135.0], # Left eye left corner
    [225.0, 170.0, -135.0],  # Right eye right corner
    [-150.0, -150.0, -125.0],# Left mouth corner
    [150.0, -150.0, -125.0]  # Right mouth corner
], dtype=np.float64)

landmark_indices = [1, 152, 33, 263, 61, 291]  # corresponding to the model points
# Detect GPU availability
if torch.cuda.is_available():
    if torch.cuda.get_device_name().startswith('NVIDIA'):
        device = torch.device('cuda')
        print("Using NVIDIA GPU with CUDA")
    elif torch.cuda.get_device_name().startswith('gfx'):
        device = torch.device('cuda')
        print("Using AMD GPU with ROCm")
    else:
        device = torch.device('cpu')
        print("CUDA device detected but not supported, using CPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple GPU with Metal Performance Shaders (MPS)")
else:
    device = torch.device('cpu')
    print("No GPU available, using CPU")

# Load the YOLOv8 segmentation seg_seg_model
byte_tracker = sv.ByteTrack()
byte_tracker.reset()
seg_model_name = 'yolo11l-seg'
seg_model = YOLO(f'{seg_model_name}.pt')
#seg_model.export(format="coreml")
#seg_model = YOLO(f'{seg_model_name}.mlpackage')
seg_model.to(device)  # Move the seg_model to GPU if available
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Could not access the webcam")
#cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    exit()
# Initialize snowflake parameters
num_snowflakes = 500
snowflake_size = 11
snowflake_speed = 50
min_speed = 10
drift_speed = 0
limit = 6
# Create initial positions for snowflakes
ret, frame = cap.read()
height, width, _ = frame.shape
dtype = np.int16
# Vectorized snowflake generation
x_coords = np.random.randint(0, width, size=num_snowflakes,dtype = dtype)  # X-coordinates
y_coords = np.random.randint(-50, height, size=num_snowflakes,dtype = dtype)  # Y-coordinates
sizes = snowflake_size + np.random.randint(-3, 1, size=num_snowflakes,dtype = dtype)  # Snowflake sizes

# Combine into a single array
snowflakes = np.column_stack((x_coords, y_coords, sizes))
def detect_smile(face_landmarks, frame_width, frame_height, wide_ratio=1.1, parted_ratio=0.25):
    """
    Returns True if a smile is detected, False otherwise.
    We measure:
      - mouth vs. eye distance (mouth_eye_ratio)
      - parted lips ratio
    :param face_landmarks: MediaPipe face landmarks
    :param frame_width, frame_height: Dimensions of the cropped region
    :param wide_ratio: threshold for mouth-eye ratio
    :param parted_ratio: threshold for parted lips ratio
    """
    # Indices
    idx_left_mouth = 61
    idx_right_mouth = 291
    idx_left_eye = 33
    idx_right_eye = 263
    idx_upper_lip = 13
    idx_lower_lip = 14

    lm = face_landmarks.landmark

    def px(idx):
        x = int(lm[idx].x * frame_width)
        y = int(lm[idx].y * frame_height)
        return x, y

    # Mouth corners
    lx_m, ly_m = px(idx_left_mouth)
    rx_m, ry_m = px(idx_right_mouth)

    # Eye corners
    lx_e, ly_e = px(idx_left_eye)
    rx_e, ry_e = px(idx_right_eye)

    # Lip vertical
    ux, uy = px(idx_upper_lip)
    bx, by = px(idx_lower_lip)

    mouth_width = np.linalg.norm([rx_m - lx_m, ry_m - ly_m])
    eye_width = np.linalg.norm([rx_e - lx_e, ry_e - ly_e]) + 1e-6  # avoid zero
    parted_lips = np.linalg.norm([by - uy])  # vertical gap 13..14

    # Ratios
    mouth_eye_ratio = mouth_width / eye_width
    lips_eye_ratio = parted_lips / eye_width

    is_big_smile = mouth_eye_ratio > wide_ratio  # corners of mouth widen
    is_open_lips = lips_eye_ratio > parted_ratio # lips parted significantly

    # Combine
    is_smiling = is_big_smile or is_open_lips
    return is_smiling


def draw_snowflakes(frame, snowflakes, snowflake_size, snowflake_speed):
    global drift_speed, limit, width, height, min_speed

    # Simulate wind: drift_speed changes like gusts
    gust_change = np.random.randint(-2, 3)
    drift_speed += gust_change
    drift_speed = np.clip(drift_speed, -limit, limit)

    # Normalize snowflake sizes (0 to 1)
    normalized_sizes = (snowflakes[:, 2] - snowflakes[:, 2].min()) / (snowflakes[:, 2].ptp() + 1e-6)

    # Compute fall speed (larger = faster)
    fall_offsets = (normalized_sizes * snowflake_speed).astype(np.int16)
    fall_offsets = np.clip(fall_offsets, min_speed, snowflake_speed + 5)
    fall_offsets += np.random.randint(-2, 3, size=len(snowflakes))  # Add some jitter

    # Compute wind drift (larger = more affected)
    drift_offsets = (normalized_sizes * drift_speed).astype(np.int16)

    # Update snowflake positions
    snowflakes[:, 0] += drift_offsets
    snowflakes[:, 1] += fall_offsets

    # Wrap/reset snowflakes that go out of frame
    out_of_bounds = (snowflakes[:, 1] > height) | (snowflakes[:, 0] < 0) | (snowflakes[:, 0] > width)
    snowflakes[out_of_bounds, 1] = np.random.randint(-50, -10, size=np.count_nonzero(out_of_bounds), dtype=np.int16)
    snowflakes[out_of_bounds, 0] = np.random.randint(0, width, size=np.count_nonzero(out_of_bounds), dtype=np.int16)
    snowflakes[out_of_bounds, 2] = snowflake_size + np.random.randint(-3, 1, size=np.count_nonzero(out_of_bounds), dtype=np.int16)

    # Draw all snowflakes
    for x, y, size in snowflakes:
        cv2.circle(frame, (int(x), int(y)), int(size), (255, 255, 255), -1)

    return snowflakes


def load_and_upsample_sprite(sprite_path, scale_factor):
    """
    Load a sprite sheet and upscale it by a given scale factor.

    Args:
        sprite_path (str): Path to the sprite sheet image.
        scale_factor (float): Scale factor to resize the sprite sheet.

    Returns:
        numpy.ndarray: Upscaled sprite sheet.
    """
    sprite_sheet = cv2.imread(sprite_path, cv2.IMREAD_UNCHANGED)
    if sprite_sheet is None:
        raise FileNotFoundError(f"Sprite sheet not found: {sprite_path}")

    original_height, original_width = sprite_sheet.shape[:2]
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the sprite sheet
    upscaled_sprite = cv2.resize(sprite_sheet, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return upscaled_sprite
# Load the PNG overlay with transparency
overlay_path = "overlay.png"  # Replace with your PNG file path
overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
# Load the PNG overlay with transparency
overlay_path = "overlay.png"  # Replace with your PNG file path
overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
candle_path = "candle.png"
candle = load_and_upsample_sprite(candle_path, 3.0)
num_sprite = 14
num_rol = 2
num_col = int(num_sprite/num_rol)
cur_sprite = 0
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[0], overlay.shape[1]
    if x >= background.shape[1] or y >= background.shape[0]:
        return background
    if x + w > background.shape[1]:
        w = background.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > background.shape[0]:
        h = background.shape[0] - y
        overlay = overlay[:h]
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [overlay, np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255],
            axis=2,
        )
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image
    return background
def sprite_overlay(background, sprite, x, y):
    """
    Overlay a frame from a sprite sheet onto the background.
    
    Args:
        background (numpy.ndarray): The main frame.
        sprite (numpy.ndarray): The sprite sheet.
        x (int): X-coordinate for overlay.
        y (int): Y-coordinate for overlay.

    Returns:
        numpy.ndarray: Updated background with sprite overlay.
    """
    global num_sprite, num_rol, cur_sprite
    h, w, channels = sprite.shape
    h_sprite = h // num_rol
    w_sprite = w // num_col
    
    # Get the current frame from the sprite sheet
    row = cur_sprite // num_col
    col = cur_sprite % num_col
    sprite_frame = sprite[row * h_sprite: (row + 1) * h_sprite, col * w_sprite: (col + 1) * w_sprite]
    # Overlay the sprite frame
    background = overlay_transparent(background, sprite_frame, x, y)
    cur_sprite = (cur_sprite + 1) % num_sprite
    return background
mask = np.zeros(frame.shape[:2], dtype = np.uint8)
frame1 = np.zeros(frame.shape,dtype = frame.dtype)
yellow = np.full_like(frame, (0, 255, 255)) 
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    landmark_indices = [1, 152, 33, 263, 61, 291]
    model_points = np.array([
        [0.0, 0.0, 0.0],         # Nose tip
        [0.0, -330.0, -65.0],    # Chin
        [-225.0, 170.0, -135.0], # Left eye left corner
        [225.0, 170.0, -135.0],  # Right eye right corner
        [-150.0, -150.0, -125.0],# Left mouth corner
        [150.0, -150.0, -125.0]  # Right mouth corner
    ], dtype=np.float64)
    # Optionally resize the frame for performance
    # frame = cv2.resize(frame, (640, 480))
    #frame = cv2.flip(frame, 1)
    # MediaPipe head pose tracking
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #results = face_mesh.process(rgb_frame)

    #if results.multi_face_landmarks:
    #    landmarks = results.multi_face_landmarks[0]
    #    landmark_indices = [1, 152, 33, 263, 61, 291]
    #    model_points = np.array([
    #        [0.0, 0.0, 0.0],         # Nose tip
    #        [0.0, -330.0, -65.0],    # Chin
    #        [-225.0, 170.0, -135.0], # Left eye left corner
    #        [225.0, 170.0, -135.0],  # Right eye right corner
    #        [-150.0, -150.0, -125.0],# Left mouth corner
    #        [150.0, -150.0, -125.0]  # Right mouth corner
    #    ], dtype=np.float64)
  
        #image_points = []
        
        # for idx in landmark_indices:
        #     lm = landmarks.landmark[idx]
        #     x, y = int(lm.x * width), int(lm.y * height)
        #     image_points.append([x, y])
        # image_points = np.array(image_points, dtype=np.float64)

        # focal_length = width
        # center = (width / 2, height / 2)
        # camera_matrix = np.array([
        #     [focal_length, 0, center[0]],
        #     [0, focal_length, center[1]],
        #     [0, 0, 1]
        #     ], dtype=np.float64)
        # dist_coeffs = np.zeros((4, 1))

        # success, rot_vec, trans_vec = cv2.solvePnP(
        # model_points, image_points, camera_matrix, dist_coeffs)

        # if success:
        #     rot_mat, _ = cv2.Rodrigues(rot_vec)
        #     pose_mat = np.hstack((rot_mat, trans_vec))
        #     _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        #     yaw, pitch, roll = euler_angles.flatten()
        #     print(f"[Head Pose] Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

    # Run YOLOv8 segmentation seg_model on the frame
    results = seg_model(frame)
    boxes = results[0].boxes  # Accessing the boxes from results
    masks = results[0].masks  # Accessing the masks from results

    is_smiling = False
    if boxes is not None:
        # Extract detection attributes
        # class_ids = boxes.cls.cpu().numpy().astype(int)
        # scores = boxes.conf.cpu().numpy()
        # xyxys = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls
        scores = boxes.conf
        xyxys = boxes.xyxy
        # Filter detections based on confidence score
        confidence_threshold = 0.7
        valid_indices = scores > confidence_threshold  # Boolean mask for valid detections

        # Apply the confidence filter
        filtered_class_ids = class_ids[valid_indices].to(torch.int16)
        filtered_scores = scores[valid_indices]
        filtered_xyxys = xyxys[valid_indices]

        # Create filtered detections
        detections = sv.Detections(
            xyxy=filtered_xyxys.cpu().numpy(),
            confidence=filtered_scores.cpu().numpy(),
            class_id=filtered_class_ids.cpu().numpy()
        )

        # Update tracker with filtered detections
        tracked_detections = byte_tracker.update_with_detections(detections)
        for box in boxes:
            if int(box.cls) != 0:
                continue  # Skip non-person detections
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]
            rgb_frame = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb_frame)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Use improved detect_smile function
                    e = detect_smile(
                        face_landmarks,
                        frame_width=(x2 - x1),
                        frame_height=(y2 - y1)
                    )
                    if(e):
                        is_smiling = True
                        print("Smiling")
    else:
        tracked_detections = []

    # Create an empty mask the size of the frame
    #mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    loc = []
    if masks is not None:
        masks_data = masks.data.cpu().numpy()
        for track in tracked_detections:
            track_id = track[4]  # tracker_id is the fifth element
            bbox = track[0].astype(int)
            x, y, w, h = bbox
            
            for mask_array, class_id, score in zip(masks_data, class_ids, scores):
                if score > 0.55 and class_id == 0:  # Assuming class ID 0 corresponds to 'person'
                    # Resize mask to match frame size if necessary
                    mask_resized = cv2.resize(mask_array, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_resized = (mask_resized > 0.5).astype(np.uint8)  # Threshold mask
                    if(mask_resized[y:h, x:w].any()):
                        # Combine masks
                        mask = cv2.bitwise_or(mask, mask_resized * 255)
                        # Draw the tracking ID text on top of an invisible box
                        text = 'Merry Christmas'
                        loc.append((x,y,text))
        
    else:
        print("Segmentation masks not available in the seg_model output.")
        continue

    # Create an inverse mask for the background
    mask_inv = cv2.bitwise_not(mask)

    # Apply a Gaussian blur to the entire frame
    np.copyto(frame1, frame)
    snowflakes = draw_snowflakes(frame, snowflakes, snowflake_size, snowflake_speed)
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    # Combine the blurred background with the original frame using the masks
    foreground = cv2.bitwise_and(frame1, frame, mask=mask)
    background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_inv)
    mask.fill(0)
    frame = cv2.addWeighted(foreground, 1, background, 1, 0)
    for x,y,text in loc:
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
    x, y = width - overlay.shape[1], height - overlay.shape[0]   # Bottom-right corner
    frame = overlay_transparent(frame, overlay, x, y)
    frame = sprite_overlay(frame, candle, 0 + 50,  height - 200)
    # Display the output
    cv2.imshow('Webcam Background Blur and Snow with YOLOv8 Segmentation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() #blur just the snow not the background
