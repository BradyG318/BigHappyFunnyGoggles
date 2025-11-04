import os
import cv2
import numpy as np
import json
import mediapipe as mp

# --- CONFIGURATION (LOCAL FILE SAVING) ---

# This is where your collected data will be stored locally.
DATABASE_FOLDER = "remote_database"
VECTORS_FILE = os.path.join(DATABASE_FOLDER, "vectors.json")
FRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "best_frames")

# Ensure necessary folders exist
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Recognition parameters
RECOGNITION_THRESHOLD = 0.48
TARGET_FACE_ID = None # Set to None to save all unique faces

# Sampling parameters for robust data collection
MIN_SAMPLES_FOR_AVERAGE = 30     # Minimum frames needed before we calculate the final vector
NUM_BEST_FRAMES_TO_SEND = 5      # Number of sharpest frames to save locally
MIN_SHARPNESS_THRESHOLD = 20.0   # Minimum image quality score to consider a frame

# --- DATA STRUCTURES ---

next_face_id = 1
known_face_encodings = [] # List of final, averaged vectors
known_face_ids = []       # List of corresponding Face IDs

# Tracks state and collected samples for each detected face
face_trackers = {} 

# --- LOAD EXISTING VECTORS (Persistence) ---
# Load vectors from the local JSON file if it exists, so IDs are consistent.
try:
    with open(VECTORS_FILE, 'r') as f:
        loaded_vectors = json.load(f)
        for face_id_str, vector_list in loaded_vectors.items():
            face_id = int(face_id_str)
            known_face_ids.append(face_id)
            known_face_encodings.append(np.array(vector_list))
            next_face_id = max(next_face_id, face_id + 1)
            
            face_trackers[face_id] = {
                'samples': [],
                'complete': True # Mark as complete since the data already exists
            }
            
        print(f"Loaded {len(known_face_ids)} existing faces. Next ID will be {next_face_id}.")
except FileNotFoundError:
    print("No existing vectors file found. Starting fresh.")
except Exception as e:
    print(f"Error loading vectors: {e}. Starting fresh.")
    
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- HELPER FUNCTIONS ---

def get_sharpness_score(image):
    """Calculates the image sharpness using the variance of the Laplacian."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- ⬇️ UPDATED 3D EMBEDDING FUNCTION ⬇️ ---

def get_face_embedding(landmarks):
    """
    Creates a rotation- and translation-normalized 3D feature vector 
    from the 468 facial landmarks for robust recognition.
    """
    all_x = [lm.x for lm in landmarks.landmark]
    all_y = [lm.y for lm in landmarks.landmark]
    all_z = [lm.z for lm in landmarks.landmark] # GET Z
    
    centroid_x = np.mean(all_x)
    centroid_y = np.mean(all_y)
    centroid_z = np.mean(all_z) # GET Z CENTROID
    
    normalized_landmarks = []
    for lm in landmarks.landmark:
        # NORMALIZE X, Y, AND Z
        normalized_landmarks.append((
            lm.x - centroid_x, 
            lm.y - centroid_y, 
            lm.z - centroid_z
        ))

    # --- Rotation is still based on 2D eye position (for roll) ---
    p1_2d = (normalized_landmarks[33][0], normalized_landmarks[33][1]) # (x, y)
    p2_2d = (normalized_landmarks[263][0], normalized_landmarks[263][1]) # (x, y)
    
    delta_x = p2_2d[0] - p1_2d[0]
    delta_y = p2_2d[1] - p1_2d[1]
    
    angle = np.arctan2(delta_y, delta_x)
    
    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)
    
    flat_vector = []
    # We rotate the x/y plane, but keep the normalized z
    for x, y, z in normalized_landmarks: 
        rotated_x = x * cos_theta - y * sin_theta
        rotated_y = x * sin_theta + y * cos_theta
        
        # Add all three coordinates to the final vector
        flat_vector.extend([rotated_x, rotated_y, z]) 
        
    return np.array(flat_vector)

# --- ⬆️ END OF UPDATED FUNCTION ⬆️ ---


def save_data_locally(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    and saves the vector and best frames to local files.
    """
    print(f"\n--- LOCAL SAVE START: Face ID #{face_id} ---")
    
    # 1. Calculate the final, averaged face vector
    all_vectors = np.array([s['vector'] for s in samples])
    final_vector = np.mean(all_vectors, axis=0)
    
    # 2. Update and save the vectors JSON file
    
    # Reload existing data to ensure concurrent updates are handled (basic sync)
    try:
        if os.path.exists(VECTORS_FILE):
            with open(VECTORS_FILE, 'r') as f:
                all_vectors_db = json.load(f)
        else:
            all_vectors_db = {}
    except Exception:
        all_vectors_db = {}
        
    all_vectors_db[str(face_id)] = final_vector.tolist()
    
    try:
        with open(VECTORS_FILE, 'w') as f:
            json.dump(all_vectors_db, f, indent=2)
        print(f"Vector saved to {VECTORS_FILE}")
    except Exception as e:
        print(f"!!! ERROR writing vector to JSON file: {e}")
        return False
    
    # 3. Select the sharpest frames
    samples.sort(key=lambda s: s['sharpness'], reverse=True)
    best_samples = samples[:NUM_BEST_FRAMES_TO_SEND]

    # 4. Save the sharpest frames to disk
    person_frame_folder = os.path.join(FRAMES_FOLDER, f"Face_{face_id}")
    os.makedirs(person_frame_folder, exist_ok=True)
    
    for i, sample in enumerate(best_samples):
        filename = os.path.join(person_frame_folder, f"face_{face_id}_sharp_{i+1:02d}.jpg")
        try:
            cv2.imwrite(filename, sample['crop'])
        except Exception as e:
            print(f"!!! ERROR saving frame {filename}: {e}")
            return False

    print(f"{len(best_samples)} frames saved to {person_frame_folder}")
    print(f"--- LOCAL SAVE COMPLETE: Face ID #{face_id} ---\n")
    return True

# --- MAIN EXECUTION ---

cap = cv2.VideoCapture(0)

# Initialize Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.01) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        current_frame_data = [] 

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                face_encoding = get_face_embedding(face_landmarks)
                face_id = None
                
                # 1. Recognize/Identify the face
                if known_face_encodings:
                    # Compare the new 3D vector to the known 3D vectors
                    distances = [np.linalg.norm(known_enc - face_encoding) 
                                 for known_enc in known_face_encodings]
                    
                    best_match_index = np.argmin(distances)
                    
                    if distances[best_match_index] < RECOGNITION_THRESHOLD:
                        face_id = known_face_ids[best_match_index]
                
                # 2. Handle New Face (First time seeing this person)
                if face_id is None:
                    face_id = next_face_id
                    
                    face_trackers[face_id] = {
                        'samples': [],
                        'complete': False
                    }
                    
                    # Temporarily store the first vector for immediate recognition purposes
                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(face_id)
                    next_face_id += 1
                    print(f"NEW FACE DETECTED: Face ID #{face_id}. Starting collection.")

                # 3. Get Bounding Box and Frame Data
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                # Use max/min to define a simple bounding box
                top = int(min(y_coords))
                bottom = int(max(y_coords))
                left = int(min(x_coords))
                right = int(max(x_coords))
                
                # Add a small buffer around the face crop
                buffer_x = int((right - left) * 0.1)
                buffer_y = int((bottom - top) * 0.1)
                
                top = max(0, top - buffer_y)
                bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x)
                right = min(w, right + buffer_x)

                face_crop = frame[top:bottom, left:right].copy()
                sharpness = get_sharpness_score(face_crop)
                
                current_frame_data.append({
                    'id': face_id,
                    'box': (top, right, bottom, left),
                    'tracker': face_trackers[face_id],
                    'crop': face_crop,
                    'sharpness': sharpness,
                    'encoding': face_encoding
                })


        # --- DRAWING & SAMPLING LOGIC ---
        
        for data in current_frame_data:
            face_id = data['id']
            top, right, bottom, left = data['box']
            tracker = data['tracker']
            
            is_target = TARGET_FACE_ID is None or face_id == TARGET_FACE_ID
            
            if is_target and not tracker['complete']:
                
                # 1. Sample collection is in progress
                samples_collected = len(tracker['samples'])
                
                if samples_collected < MIN_SAMPLES_FOR_AVERAGE:
                    
                    # Check sharpness threshold before adding sample
                    if data['sharpness'] >= MIN_SHARPNESS_THRESHOLD:
                        tracker['samples'].append({
                            'vector': data['encoding'],
                            'crop': data['crop'],
                            'sharpness': data['sharpness']
                        })
                        color = (0, 255, 0) # Green: Collecting
                        status = f"COLLECTING ({samples_collected + 1}/{MIN_SAMPLES_FOR_AVERAGE})"
                    else:
                        color = (255, 255, 0) # Yellow: Too blurry
                        status = f"BLURRY ({data['sharpness']:.1f}/{MIN_SHARPNESS_THRESHOLD:.1f})"
                        
                # 2. Collection is complete, save data once
                else:
                    if save_data_locally(face_id, tracker['samples']):
                        tracker['complete'] = True
                        color = (0, 0, 255) # Red: Complete (Success)
                        status = "SAVE SUCCESS"
                        
                        # --- CRITICAL ---
                        # Once saved, we must update the 'known_face_encodings'
                        # with the new, final averaged vector.
                        # Find the index for this face_id
                        idx = known_face_ids.index(face_id)
                        # Reload the final vector from the save function
                        final_avg_vector = np.mean(np.array([s['vector'] for s in tracker['samples']]), axis=0)
                        known_face_encodings[idx] = final_avg_vector
                        print(f"--- Face ID #{face_id} embedding has been finalized. ---")
                        
                    else:
                        color = (0, 165, 255) # Orange: Complete (Failed save)
                        status = "SAVE FAILED" 
            
            # 3. Not a target face or already complete
            elif tracker['complete']:
                color = (0, 0, 255) # Red: Complete (Recognized)
                status = "RECOGNIZED"
            else:
                color = (255, 165, 0) # Orange: Non-target (Skip)
                status = "SKIPPED"

            # --- Drawing on Frame ---
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"Face #{face_id}", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{status}", (left, top - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # --- DISPLAY STATUS ---
        
        num_faces = len(current_frame_data)
        
        info_text = f"Total Unique People: {next_face_id - 1}. Target: {'ALL' if TARGET_FACE_ID is None else f'#{TARGET_FACE_ID}'}"
        
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Use a non-fullscreen window for easier debugging
        # cv2.namedWindow('Landmark-Based Face Recognition Tracker', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('Landmark-Based Face Recognition Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Landmark-Based Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
 
cv2.destroyAllWindows()
cap.release()