#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 21:56:51 2024

@author: dangkhoa
"""
import cv2
import numpy as np
import requests
import base64
import os

CLOUD_GPU_ENDPOINT = os.getenv("CLOUD_GPU_ENDPOINT", "http://localhost:8000/infer")
REQUEST_TIMEOUT_SEC = float(os.getenv("CLOUD_GPU_TIMEOUT", "5"))


def request_human_mask(frame):
    """
    Send the frame to the cloud GPU endpoint and return a mask pair.

    Expected response payload format (JSON):
    {
      "mask": [[0..255, ...], ...],
      "mask_inv": [[0..255, ...], ...]  # same HxW as input frame
    }
    """
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame before HTTP inference request")

    payload = {"image": base64.b64encode(encoded).decode("utf-8")}
    response = requests.post(CLOUD_GPU_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()

    data = response.json()
    container = data.get("data") if isinstance(data.get("data"), dict) else data

    mask_data = container.get("mask")
    mask_inv_data = container.get("mask_inv")
    if mask_data is None or mask_inv_data is None:
        raise ValueError("Inference response must include 'mask' and 'mask_inv' fields")

    mask = np.asarray(mask_data, dtype=np.uint8)
    mask_inv = np.asarray(mask_inv_data, dtype=np.uint8)
    if mask.shape != frame.shape[:2] or mask_inv.shape != frame.shape[:2]:
        raise ValueError(
            f"Mask shape mismatch. Expected {frame.shape[:2]}, got {mask.shape} and {mask_inv.shape}"
        )
    return mask, mask_inv
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
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
frame1 = np.zeros(frame.shape,dtype = frame.dtype)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        mask, mask_inv = request_human_mask(frame)
    except Exception as exc:
        print(f"Segmentation request failed: {exc}")
        mask.fill(0)
        continue

    # Apply a Gaussian blur to the entire frame
    np.copyto(frame1, frame)
    snowflakes = draw_snowflakes(frame, snowflakes, snowflake_size, snowflake_speed)
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
    # Combine the blurred background with the original frame using the masks
    foreground = cv2.bitwise_and(frame1, frame, mask=mask)
    background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask_inv)
    mask.fill(0)
    frame = cv2.addWeighted(foreground, 1, background, 1, 0)
    x, y = width - overlay.shape[1], height - overlay.shape[0]   # Bottom-right corner
    frame = overlay_transparent(frame, overlay, x, y)
    frame = sprite_overlay(frame, candle, 0 + 50,  height - 200)
    # Display the output
    cv2.imshow('Webcam Background Blur and Snow', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows() #blur just the snow not the background
