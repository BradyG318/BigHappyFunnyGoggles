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
def is_url(path):
    """Check if the given string is a URL."""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc]) and result.scheme in ('http', 'https')
    except:
        return False

def load_image_from_url(url):
    """Download image from URL and return as OpenCV image (numpy array)."""
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type:
            print(f"URL does not point to an image (content-type: {content_type})")
            return None
        # Read image data
        img_bytes = response.content
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def load_image(source):
    """
    Load image from local path or URL.
    Returns OpenCV image (numpy array) or None on failure.
    """
    if is_url(source):
        return load_image_from_url(source)
    else:
        # Local file
        if not os.path.isfile(source):
            print(f"Local file not found: {source}")
            return None
        return cv2.imread(source)

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
def save_data_to_database(encoding, image_source):
    """Save the vector to PostgreSQL database."""
    print(f"\n--- DATABASE SAVE: Image Source {image_source} ---")
    success = DB_Link.db_link.save_encoding(encoding.tolist(), image_source)
    if not success:
        print(f"!!! ERROR saving vector for image source {image_source} !!!")
        return False
    print(f"Vector saved for image source {image_source}\n")
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

def process_single_image(image_source):
    """
    Process one image (local path or URL): detect face, generate embedding, save to DB.
    Returns True on success, False otherwise.
    """
    print(f"\nProcessing: {image_source}")
    img = load_image(image_source)
    if img is None:
        print(f"Error: Could not load image from {image_source}")
        return False

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
    pad = 20
    left = max(0, left - pad)
    right = min(w, right + pad)
    top = max(0, top - pad)
    bottom = min(h, bottom + pad)

    face_crop = img[top:bottom, left:right]
    face_crop = conservative_lighting_normalization(face_crop)
    encoding = get_deepface_embedding(face_crop)
    if encoding is None:
        print(f"Error: Could not generate embedding for {image_source}")
        return False

    # Normalize
    encoding = encoding / np.linalg.norm(encoding)

    # Save to DB
    return save_data_to_database(encoding, image_source)

# ----------------------------------------------------------------------
def process_batch(json_file):
    """
    Read JSON file containing an array of objects with keys:
        "image_path" (string, local path or URL) and "face_id" (int).
    Process each entry.
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
        print("Error: JSON root must be an array of objects.")
        return False

    success_count = 0
    fail_count = 0

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Row {idx} is not an object, skipping.")
            fail_count += 1
            continue

        meta_tags = item.get('metatags', {})
        
        image_path = meta_tags[0].get('og:image')

        if image_path is None:
            print(f"Row {idx} missing 'metatags' or 'og:image', skipping.")
            fail_count += 1
            continue

        if process_single_image(image_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n=== Batch processing completed: {success_count} succeeded, {fail_count} failed ===")
    return fail_count == 0

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Add face(s) to database from images (local or URL).')
    parser.add_argument('input', help='Image path/URL OR JSON file path (use --json flag)')
    parser.add_argument('face_id', nargs='?', type=int,
                        help='Face ID (required if input is a single image)')
    parser.add_argument('--json', action='store_true',
                        help='Treat input as a JSON file with an array of {"image_path": "...", "face_id": N}')

    args = parser.parse_args()

    # Initialize database connection
    print("Initializing database...")
    DB_Link.db_link.initialize()

    if args.json:
        # Batch mode
        process_batch(args.input)
    else:
        DB_Link.db_link.close()
        # Single image mode
        if args.face_id is None:
            print("Error: For single image you must provide a face_id.")
            sys.exit(1)
        if process_single_image(args.input, args.face_id):
            print("\n✅ Face added successfully!")
            sys.exit(0)
        else:
            print("\n❌ Failed to add face.")
            sys.exit(1)

if __name__ == "__main__":
    main()