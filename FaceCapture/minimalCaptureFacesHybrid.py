import time
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mediapipe as mp
from deepface import DeepFace
import os
import json
# import DB_Link # COMMENTED OUT: Database link is no longer used

# --- CONFIGURATION ---
# DB_Link.db_link.initialize() # COMMENTED OUT: Database initialization
# DB_Link.db_link.clear_db()    # COMMENTED OUT: Database clear

# --- LOCAL STORAGE CONFIGURATION ---
LOCAL_DB_FOLDER = "rm_db"
LOCAL_DB_FILE = os.path.join(LOCAL_DB_FOLDER, "face_vectors.json")
# --- END LOCAL STORAGE CONFIGURATION ---

# --- FACE COLLECTION CONFIG ---
SAMPLES_TO_COLLECT = 30 
BEST_SAMPLES_TO_AVERAGE = 10 
# --- END FACE COLLECTION CONFIG ---


DEEPFACE_MODEL = 'Facenet512'
mp_face_mesh = mp.solutions.face_mesh

# SIMPLIFIED CONFIG
RECOGNITION_THRESHOLD = 0.85
MIN_SAMPLES_FOR_AVERAGE = 20
POSE_QUALITY_THRESHOLD = 0.20

# DATA STRUCTURES
next_face_id = 1
known_face_encodings = []
known_face_ids = []
face_trackers = {} # Stores temporary samples for new faces

# Ensure the local database folder exists
os.makedirs(LOCAL_DB_FOLDER, exist_ok=True)

# Helper function to save known faces locally
def save_known_faces_locally():
    """Saves known_face_ids and known_face_encodings to a JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    data = {
        "next_face_id": next_face_id,
        "faces": [
            {
                "id": known_face_ids[i],
                "vector": known_face_encodings[i].tolist() # Convert numpy array to list
            }
            for i in range(len(known_face_ids))
        ]
    }
    with open(LOCAL_DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved {len(known_face_ids)} faces locally to {LOCAL_DB_FILE}")

# --- NEW FUNCTION: FINALIZES AND SAVES A FACE VECTOR ---
def finalize_face_vector(temp_face_id):
    """
    Selects the BEST_SAMPLES_TO_AVERAGE from the temporary samples, 
    averages their vectors, and adds the result to the known lists.
    """
    global known_face_encodings, known_face_ids
    
    tracker = face_trackers[temp_face_id]
    
    # 1. Sort samples by pose_quality (highest first)
    sorted_samples = sorted(tracker['best_samples'], key=lambda x: x['quality'], reverse=True)
    
    # 2. Select the top N best samples
    best_vectors = [s['vector'] for s in sorted_samples[:BEST_SAMPLES_TO_AVERAGE]]
    
    if not best_vectors:
        print(f"ERROR: Face #{temp_face_id} has no valid vectors to finalize.")
        del face_trackers[temp_face_id]
        return
    
    # 3. Average the vectors to create the final reference vector
    avg_vector = np.mean(best_vectors, axis=0)
    
    # 4. Normalize the final vector (essential for cosine similarity)
    final_vector = avg_vector / np.linalg.norm(avg_vector)

    # 5. Add to the global known lists
    known_face_encodings.append(final_vector)
    known_face_ids.append(temp_face_id)
    
    # 6. Mark as complete and clean up temporary samples
    tracker['complete'] = True
    tracker['best_samples'] = [] # Clear samples to save memory
    
    print(f"*** FACE {temp_face_id} FINALIZED! Averaged {BEST_SAMPLES_TO_AVERAGE} samples. ***")


# Load existing vectors (modified for local file loading)
try:
    if os.path.exists(LOCAL_DB_FILE):
        with open(LOCAL_DB_FILE, 'r') as f:
            data = json.load(f)
        
        # Load global tracking variables
        next_face_id = data.get("next_face_id", 1)
        
        for face_data in data.get("faces", []):
            face_id = face_data['id']
            vector = np.array(face_data['vector'])
            
            known_face_ids.append(face_id)
            known_face_encodings.append(vector)
            # Ensure tracked faces are marked complete at load
            face_trackers[face_id] = {'samples_collected': 0, 'complete': True, 'best_samples': []} 
            
        print(f"Loaded {len(known_face_ids)} existing faces from {LOCAL_DB_FILE}")
    else:
        print("No local face data found. Starting fresh.")
except Exception as e:
    print(f"Error loading: {e}")

# --- FUNCTION DEFINITIONS START HERE ---

# SIMPLIFIED POSE QUALITY
def get_pose_quality_score(landmarks):
    """Calculates a simple pose quality score based on eye alignment."""
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    left_eye = points_3d[33]
    right_eye = points_3d[263]
    eye_vector = right_eye - left_eye
    roll_penalty = abs(eye_vector[1]) / np.linalg.norm(eye_vector)
    return max(0, 1.0 - roll_penalty * 2)

# SIMPLIFIED RECOGNITION
def recognize_face_simple(face_encoding, known_encodings, known_ids):
    """Performs cosine similarity search for a known face."""
    if face_encoding is None or not known_encodings:
        return None
    
    best_similarity = 0
    best_match_id = None
    
    for face_id, known_encoding in zip(known_ids, known_encodings):
        # NOTE: Cosine similarity is effectively dot product if vectors are normalized
        similarity = np.dot(face_encoding, known_encoding)
        
        if similarity > best_similarity and similarity >= RECOGNITION_THRESHOLD:
            best_similarity = similarity
            best_match_id = face_id
    
    return best_match_id

def get_deepface_embedding(face_crop):
    try:
        # Convert BGR to RGB for DeepFace
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # DeepFace aligns and normalizes the vector
        embeddings = DeepFace.represent(img_path=face_crop_rgb, model_name=DEEPFACE_MODEL, align=True, enforce_detection=False)
        return np.array(embeddings[0]['embedding']) if embeddings else None
    except Exception as e:
        warnings.filterwarnings("default") # Temporarily re-enable warnings
        print(f"DeepFace error: {e}")
        warnings.filterwarnings("ignore")
        return None

# MAIN LOOP
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    print("Testing basic face recognition. Press 'Esc' to exit.")
    
    frame_count = 0
    
    while cap.isOpened():
        frame_count += 1
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            time.sleep(1)
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        face_detected = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                left, right = int(min(x_coords)), int(max(x_coords))
                top, bottom = int(min(y_coords)), int(max(y_coords))
                
                # Add padding to bounding box
                padding = 20
                left = max(0, left - padding)
                right = min(w, right + padding)
                top = max(0, top - padding)
                bottom = min(h, bottom + padding)
                
                if (right - left) < 50 or (bottom - top) < 50:
                    continue
                
                # Check pose and skip if poor
                pose_quality = get_pose_quality_score(face_landmarks)
                
                if pose_quality < POSE_QUALITY_THRESHOLD:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, "Poor Pose", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    continue
                
                face_crop = frame[top:bottom, left:right]
                if face_crop.size == 0:
                    continue
                
                if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                    face_crop = cv2.resize(face_crop, (100, 100))
                
                # Get embedding
                face_encoding = get_deepface_embedding(face_crop)
                if face_encoding is None:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Embedding Failed", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue
                
                # SIMPLE RECOGNITION (Only against finalized vectors)
                face_id = recognize_face_simple(face_encoding, known_face_encodings, known_face_ids)
                
                display_text = ""
                display_color = (0, 255, 0) # Green
                
                # --- NEW FACE LOGIC (Sample Collection) ---
                if face_id is None:
                    # Look for a face that is currently being sampled (not yet complete)
                    current_tracker_id = None
                    for tid, tracker in face_trackers.items():
                        if not tracker['complete']:
                            # Simple assumption: only track one face being registered at a time
                            current_tracker_id = tid
                            break
                    
                    if current_tracker_id is None:
                        # Start tracking a NEW face
                        current_tracker_id = next_face_id
                        face_trackers[current_tracker_id] = {'samples_collected': 0, 'complete': False, 'best_samples': []}
                        next_face_id += 1
                        print(f"NEW FACE REGISTRATION STARTED: ID #{current_tracker_id}")
                        
                    face_id = current_tracker_id
                    tracker = face_trackers[face_id]
                    
                    # 1. Store the sample
                    if tracker['samples_collected'] < SAMPLES_TO_COLLECT:
                        tracker['best_samples'].append({
                            'vector': face_encoding,
                            'quality': pose_quality 
                        })
                        tracker['samples_collected'] += 1
                        
                        display_text = f"NEW FACE: Capturing {tracker['samples_collected']}/{SAMPLES_TO_COLLECT}"
                        display_color = (255, 165, 0) # Orange
                        
                        # 2. Check for finalization
                        if tracker['samples_collected'] >= SAMPLES_TO_COLLECT:
                            finalize_face_vector(face_id)
                            display_text = f"Face #{face_id} FINALIZED!"
                            display_color = (0, 255, 0) # Green
                            
                    else:
                         # Should not happen if finalize_face_vector marks complete, but as a fallback
                         display_text = f"Face #{face_id} FINALIZING..."
                         display_color = (255, 0, 255) # Magenta
                         
                else:
                    # Recognized face
                    display_text = f"Face #{face_id} Recognized"
                    display_color = (0, 255, 0) # Green

                # --- END NEW FACE LOGIC ---
                
                # Drawing
                cv2.rectangle(frame, (left, top), (right, bottom), display_color, 2)
                cv2.putText(frame, display_text, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
                face_detected = True
        
        # If no faces detected, show message
        if not face_detected:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Simple Face Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord(' '): # Spacebar to pause
            cv2.waitKey(0)

# Save data locally before closing
save_known_faces_locally()

cap.release()
cv2.destroyAllWindows()