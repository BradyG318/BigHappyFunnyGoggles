# pfp_scanner.py – supports single image (local or URL) or JSON batch processing with image URLs

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mediapipe as mp
from deepface import DeepFace
import math
import sys
import argparse
import json
import os
import requests
import io
from urllib.parse import urlparse

# For database storage
import DB_Link

# --- CONFIGURATION ---
DEEPFACE_MODEL = 'Facenet512'
REQUEST_TIMEOUT = 10  # seconds for HTTP requests

# Initialize MediaPipe FaceMesh once
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.65
)

# ----------------------------------------------------------------------
def load_image(path):
    """
    Load image from local path or URL.
    Returns OpenCV image (numpy array) or None on failure.
    """
    if not os.path.isfile(path):
        print(f"Local file not found: {path}")
        return None
    return cv2.imread(path)

# ----------------------------------------------------------------------
def get_deepface_embedding(face_crop):
    """Generate embedding from a face crop."""
    if face_crop is None or face_crop.size == 0:
        return None
    try:
        embeddings = DeepFace.represent(
            img_path=face_crop,
            model_name=DEEPFACE_MODEL,
            enforce_detection=False,
            align=True
        )
        if embeddings:
            return np.array(embeddings[0]['embedding'])
        else:
            return None
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

# ----------------------------------------------------------------------
def save_data_to_database(encoding, name, age):
    """Save the vector to PostgreSQL database."""
    print(f"\n--- DATABASE SAVE: {name} ---")
    #success = DB_Link.db_link.replace_encoding(id, encoding.tolist())
    
    success, id = DB_Link.db_link.save_encoding(encoding.tolist())
    
    if not success:
        print(f"!!! ERROR saving vector for {name} !!!")
        return False
    print(f"Vector saved for {name}\n")
    
    success = DB_Link.db_link.save_info(id, fullname=name, age=age)    
    
    return True

# ----------------------------------------------------------------------
def conservative_lighting_normalization(face_crop: np.ndarray) -> np.ndarray:
    """Conservative lighting normalization that preserves facial features."""
    if face_crop is None or face_crop.size == 0: return face_crop
    
    try:
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        mean_brightness = np.mean(l_channel); std_brightness = np.std(l_channel)
        shadow_area = np.percentile(face_crop, 10) # Checking the shadows passed by the glasses 
        
        if mean_brightness > 200 and std_brightness < 40: #this is for too bright 
            gamma = 1.5         #; inv_gamma = 1.0 / gamma  |darken the overexposured image
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8") #inv_gamma changes to gamma
            return cv2.LUT(face_crop, table)
        elif mean_brightness < 60 or shadow_area < 35: # originally (40) checking for shadows casted by the glasses to make sure that they arent't too much 
            alpha = 1.3; beta = 45 # originally 1.2, 30 (hopefully 45 will lift the shadows)
            return cv2.convertScaleAbs(face_crop, alpha=alpha, beta=beta)
        else:
            return face_crop
    except Exception:
        return face_crop

def process_images(name, age, images):
    """
    Process one image (local path or URL): detect face, generate embedding, save to DB.
    Returns True on success, False otherwise.
    """
    encodings = []
    
    for image_source in images:
        print(f"\nProcessing: {image_source}")
        img = load_image("Face Bank/" + image_source)
        if img is None:
            print(f"Error: Could not load image from {image_source}")
            return False

        # Apply CLAHE prior to looking for the face                                    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img[:,:,0] = clahe.apply(img[:,:,0])
        
        # Converting image from LAB Color model to BGR color space
        img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            print(f"Error: No face detected in {image_source}")
            return False

        # Use the first detected face
        face_landmarks = results.multi_face_landmarks[0]
        h, w = img.shape[:2]
        x_coords = [lm.x * w for lm in face_landmarks.landmark]
        y_coords = [lm.y * h for lm in face_landmarks.landmark]

        left, right = int(min(x_coords)), int(max(x_coords))
        top, bottom = int(min(y_coords)), int(max(y_coords))

        # Padding
        pad_x = 0.05 * (right - left)
        pad_y = 0.05 * (bottom - top)
        
        left = int(max(0, left - pad_x))
        right = int(min(w, right + pad_x))
        top = int(max(0, top - pad_y))
        bottom = int(min(h, bottom + pad_y))

        face_crop = img[top:bottom, left:right]
        
        face_crop = conservative_lighting_normalization(face_crop)
        encoding = get_deepface_embedding(face_crop)
        if encoding is None:
            print(f"Error: Could not generate embedding for {image_source}")
            return False
        
        # If it exists, add to the list of encodings to average later
        encodings.append(encoding)

    # Average all vectors together
    if encodings:
        encoding = np.mean(encodings, axis=0)
    
    # Normalize
    encoding = encoding / np.linalg.norm(encoding)
    
    # Save to DB
    return save_data_to_database(encoding, name, age)

# ----------------------------------------------------------------------
def process_batch(json_file):
    """
    Read JSON file containing face data.
    Supported format:
        [{"name": "Andrew McCleary", "image_paths": ["test1.jpg", "test2.jpg", ...]},
         {"name": "Jane Doe", "image_paths": ["test1.jpg", "test2.jpg", ...]},
         ...]
    """
    if not os.path.isfile(json_file):
        print(f"Error: JSON file '{json_file}' not found.")
        return False

    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return False

    if not isinstance(data, list):
        print("Error: JSON root must be an array.")
        return False

    # Parse entries from JSON objects
    entries = {}
    ages = {}
    
    for item in data:
        if isinstance(item, dict) and "name" in item and "image_paths" in item and "age" in item:
            name = item["name"]
            image_paths = item["image_paths"]
            age = item["age"]

            if isinstance(image_paths, list):
                entries[name] = image_paths
                ages[name] = age
            else:
                print(f"Skipping invalid entry for {name}: image_paths must be a list")
        else:
            print(f"Skipping invalid item: {item}")

    if not entries:
        print("No valid entries found in JSON file.")
        return False

    success_count = 0
    fail_count = 0

    for name in entries.keys():
        if process_images(name, ages[name], entries[name]):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n=== Batch processing completed: {success_count} succeeded, {fail_count} failed ===")
    return fail_count == 0

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Add face(s) to database from images (local only).')
    parser.add_argument('input', help='JSON file path')

    args = parser.parse_args()

    # Initialize database connection
    print("Initializing database...")
    DB_Link.db_link.initialize()

    args.input = "Face Bank/" + args.input

    process_batch(args.input)

if __name__ == "__main__":
    main()
    DB_Link.db_link.close()