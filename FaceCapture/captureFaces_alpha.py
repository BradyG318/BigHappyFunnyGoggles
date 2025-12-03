import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace
import math

# For database storage
import DB_Link

# For local storage
# import os
# import json

# --- CONFIGURATION ---
# Database storage
DB_Link.db_link.initialize()
#DB_Link.db_link.clear_db()

# Local storage
# LOCAL_DB_FOLDER = "rm_db"
# LOCAL_DB_FILE = os.path.join(LOCAL_DB_FOLDER, "face_vectors.json")
# os.makedirs(LOCAL_DB_FOLDER, exist_ok=True)

# Camera
CAMERA_INDEX = 0 #ndi plugin, 7 for glasses on my laptop

# Face Collection Config
SAMPLES_TO_COLLECT = 30 
BEST_SAMPLES_TO_AVERAGE = 10 

# Models
mp_face_mesh = mp.solutions.face_mesh
DEEPFACE_MODEL = 'Facenet512'

# Recongition Threshold 
RECOGNITION_THRESHOLD = 0.85

# Overall threshold remains low to allow maximum ROM based on reduced penalties below.
POSE_QUALITY_THRESHOLD_ID = 0.50
POSE_QUALITY_THRESHOLD_CAPTURE = 0.89

# Sharpness: Laplacian Variance. < 50 is usually very blurry.
MIN_SHARPNESS_THRESHOLD = 50.0

# --- DATA STRUCTURES ---
next_face_id = 1
known_face_encodings = []
known_face_ids = []
face_trackers = {} 
currently_tracked_faces = set()

# --- HELPER FUNCTIONS ---
def get_pose_quality(landmarks):
    """
    Robust score (0.0 to 1.0) checking Roll, Yaw, and Pitch.
    """
    # Store landmarks
    lm = landmarks.landmark
    l_eye = np.array([lm[33].x, lm[33].y])
    r_eye = np.array([lm[263].x, lm[263].y])
    nose = np.array([lm[1].x, lm[1].y])
    lip = np.array([lm[13].x, lm[13].y])
    
    # 1. Calculate roll (Head Tilt Sideways)
    dY = r_eye[1] - l_eye[1]; dX = r_eye[0] - l_eye[0]
    angle = math.degrees(math.atan2(dY, dX)) 
    roll_penalty = (abs(angle) / 60.0) * 1.5 #Tuning
    
    # 2. Calculate yaw (Head Turn Left/Right)
    eye_center_x = (l_eye[0] + r_eye[0]) / 2
    eye_width = np.linalg.norm(r_eye - l_eye)
    yaw_deviation = abs(nose[0] - eye_center_x) / eye_width
    yaw_penalty = yaw_deviation * 1.8 #Tuning
    
    # 3. Calculate pitch (Head Look Up/Down)
    eye_line_y = (l_eye[1] + r_eye[1]) / 2
    nose_to_eye = abs(nose[1] - eye_line_y)
    nose_to_lip = abs(lip[1] - nose[1])
    if nose_to_lip == 0: nose_to_lip = 0.001
    ratio = nose_to_eye / nose_to_lip
    
    pitch_penalty = 0
    if ratio < 0.4: pitch_penalty = (0.4 - ratio) # Look up penalty, tuning
    elif ratio > 2.5: pitch_penalty = (ratio - 2.5) # Look down penalty, tuning
    
    # Calculate and return total penalty
    total_penalty = roll_penalty + yaw_penalty + pitch_penalty
    score = max(0, 1.0 - total_penalty)
    return score

def get_image_sharpness(image):
    """Returns the variance of the Laplacian (sharpness score). Higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def conservative_lighting_normalization(face_crop):
    """
    Conservative lighting normalization that preserves facial features
    Only applies correction when absolutely necessary
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        
        # Calculate lighting statistics
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Only apply correction in extreme cases
        if mean_brightness > 200 and std_brightness < 40:  # Severe overexposure
            # Gentle gamma correction instead of aggressive normalization
            gamma = 1.3
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            corrected = cv2.LUT(face_crop, table)
            return corrected
            
        elif mean_brightness < 40:  # Severe underexposure
            # Mild brightness boost
            alpha = 1.2  # Contrast control
            beta = 30    # Brightness control
            corrected = cv2.convertScaleAbs(face_crop, alpha=alpha, beta=beta)
            return corrected
            
        else:
            return face_crop
        #     # For moderate lighting issues, apply very mild normalization
        #     l_normalized = cv2.normalize(l_channel, None, 50, 200, cv2.NORM_MINMAX)
        #     lab[:,:,0] = l_normalized
        #     normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        #     return normalized
            
    except Exception as e:
        return face_crop

def get_deepface_embedding(face_crop):
    """
    Uses DeepFace to encode the cropped face image into a feature vector (embedding).
    """
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
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors (range -1 to 1)"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def recognize_face(face_encoding, known_face_encodings, known_ids, recognition_threshold = RECOGNITION_THRESHOLD):
    """Performs recognition using weighted cosine similarity."""
    if face_encoding is None or not known_face_encodings:
        return None
    
    # Initialize best similarity and match id
    best_similarity = -1
    best_match_id = None

    # Loop through known faces
    for face_id, known_encoding in zip(known_ids, known_face_encodings):
        base_similarity = cosine_similarity(face_encoding, known_encoding)
        
        # Check if this is the best match so far
        if base_similarity > best_similarity and base_similarity >= recognition_threshold:
            best_similarity = base_similarity
            best_match_id = face_id

    return best_match_id

# def save_known_faces_locally():
#     """Saves only complete face vectors to a JSON file."""
#     try:
#         # Filter to only include complete faces
#         complete_faces = []
#         for i in range(len(known_face_ids)):
#             face_id = known_face_ids[i]
#             # Check if this face has a tracker and is complete
#             if face_id in face_trackers and face_trackers[face_id].get('complete', False):
#                 complete_faces.append({
#                     "id": face_id,
#                     "vector": known_face_encodings[i].tolist() 
#                 })
        
#         data = {
#             "next_face_id": next_face_id,
#             "faces": complete_faces
#         }
        
#         with open(LOCAL_DB_FILE, 'w') as f:
#             json.dump(data, f, indent=4)
#         print(f"Saved {len(complete_faces)} complete faces locally to {LOCAL_DB_FILE}")
#     except Exception as e:
#         print(f"Error saving local file: {e}")

def save_data_to_database(face_id, encoding):
    """
    Saves the vector to PostgreSQL database.
    """
    print(f"\n--- DATABASE SAVE START: Face ID #{face_id} ---")

    # Save the final vector to database synchronously
    success = DB_Link.db_link.save_face_vector(face_id, encoding.tolist())
    
    if not success:
        print(f"!!! ERROR saving vector to database for face #{face_id}")
        return False
    
    print(f"Vector saved to database for Face ID #{face_id}")
    print(f"--- DATABASE SAVE COMPLETE: Face ID #{face_id} ---\n")
    return True

def finalize_face_vector(temp_face_id):
    """
    Averages collected samples.
    Sorts by a combined score of Pose Quality AND Sharpness to pick the best ones.
    """
    global known_face_encodings, known_face_ids
    
    if temp_face_id not in face_trackers: return

    tracker = face_trackers[temp_face_id]
    
    # Sort samples by combined quality score
    sorted_samples = sorted(
        tracker['best_samples'], 
        key=lambda x: x['quality'] + (min(x['sharpness'], 500) / 1000.0), 
        reverse=True
    )
    
    best_vectors = [s['vector'] for s in sorted_samples[:BEST_SAMPLES_TO_AVERAGE]]
    
    if not best_vectors: return
    
    # Average and Normalize
    avg_vector = np.mean(best_vectors, axis=0)
    final_vector = avg_vector / np.linalg.norm(avg_vector)

    # Update memory
    if temp_face_id in known_face_ids:
        idx = known_face_ids.index(temp_face_id)
        known_face_encodings[idx] = final_vector
    
    tracker['complete'] = True
    tracker['best_samples'] = [] 
    
    print(f"*** FACE {temp_face_id} FINALIZED! Averaged {BEST_SAMPLES_TO_AVERAGE} sharpest samples. ***")
    save_data_to_database(temp_face_id, final_vector)

# --- MAIN ---
# Load existing vectors from database
try:
    vectors_dict = DB_Link.db_link.get_all_vectors()
    for face_id_str, vector_list in vectors_dict.items():
        face_id = int(face_id_str)
        known_face_ids.append(face_id)
        known_face_encodings.append(np.array(vector_list))
        next_face_id = max(next_face_id, face_id + 1)
        
        face_trackers[face_id] = {
            'samples': [],
            'complete': True 
        }

    print(f"Loaded {len(known_face_ids)} existing faces from database. Next ID will be {next_face_id}.")
    
except Exception as e:
    print(f"Error loading from database: {e}. Starting fresh.")

# Load existing vectors (modified for local file loading)
# try:
#     if os.path.exists(LOCAL_DB_FILE):
#         with open(LOCAL_DB_FILE, 'r') as f:
#             data = json.load(f)
        
#         # Load global tracking variables
#         next_face_id = data.get("next_face_id", 1)
        
#         for face_data in data.get("faces", []):
#             face_id = face_data['id']
#             vector = np.array(face_data['vector'])
            
#             known_face_ids.append(face_id)
#             known_face_encodings.append(vector)
#             # Ensure tracked faces are marked complete at load
#             face_trackers[face_id] = {'samples_collected': 0, 'complete': True, 'best_samples': []} 
            
#         print(f"Loaded {len(known_face_ids)} existing faces from {LOCAL_DB_FILE}")
#     else:
#         print("No local face data found. Starting fresh.")
# except Exception as e:
#     print(f"Error loading: {e}")

# Start video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

# Initialize Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.01) as face_mesh:

    print("Starting face capture. Press 'Esc' to exit.")

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, mark the frame as not writeable to pass by reference
        frame.flags.writeable = False

        # --- DETECTION & CAPTURE ---

        # Use detection frame for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Reset data structures for tracking current detection info
        current_frame_data = []
        currently_tracked_faces.clear()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]

                left, right = int(min(x_coords)), int(max(x_coords))
                top, bottom = int(min(y_coords)), int(max(y_coords))
                
                # Padding
                pad = 20
                left = max(0, left-pad); right = min(w, right+pad)
                top = max(0, top-pad); bottom = min(h, bottom+pad)

                # Skip small detections
                face_width = right - left
                face_height = bottom - top
                if face_width < 60 or face_height < 60:
                    print(f"Skipped small face detection: {face_width}x{face_height}px")
                    continue
                
                # Crop face region
                face_crop = frame[top:bottom, left:right]
                
                # Apply lighting normalization ONLY in extreme cases
                face_crop = conservative_lighting_normalization(face_crop)

                # Set default status
                status = "Processing..."
                color = (0, 0, 255) # Red (Default)

                # Check if face sharp enough
                sharpness = get_image_sharpness(face_crop)

                if sharpness > MIN_SHARPNESS_THRESHOLD:
                    # Check level of pose validity (either capture mode or id mode)
                    pose_score = get_pose_quality(face_landmarks)

                    #print("DEBUG PS: ", pose_score)
                    
                    if pose_score > POSE_QUALITY_THRESHOLD_CAPTURE: # CAPTURE MODE
                        #print("DEBUG CAPTURE MODE")
                        # Image is Sharp & Good Angle -> Get Embedding
                        face_encoding = get_deepface_embedding(face_crop)

                        # Identify
                        if face_encoding is not None:
                            face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids)
                                  
                            # Check-In New Person
                            if face_id is None:
                                face_id = next_face_id
                                next_face_id += 1
                                face_trackers[face_id] = {'samples_collected': 0, 'complete': False, 'best_samples': []}
                                known_face_ids.append(face_id)
                                known_face_encodings.append(face_encoding)
                                print(f"NEW PERSON: ID #{face_id}")

                                # Tracker Logic
                                tracker = face_trackers[face_id]
                                
                                if tracker:
                                    if not tracker['complete']:
                                        if tracker['samples_collected'] < SAMPLES_TO_COLLECT:
                                            # Store Vector + Pose Score + Sharpness Score
                                            tracker['best_samples'].append({
                                                'vector': face_encoding, 
                                                'quality': pose_score,
                                                'sharpness': sharpness
                                            })
                                            
                                            tracker['samples_collected'] += 1
                                            color = (255, 165, 0) # Orange
                                            status = f"ID #{face_id}: {tracker['samples_collected']}/{SAMPLES_TO_COLLECT}"
                                        
                                        else:
                                            finalize_face_vector(face_id)
                                            color = (0, 255, 0) # Green
                                            status = f"ID #{face_id} SAVED"
                                    
                                    else:
                                        color = (0, 255, 0) # Green
                                        status = f"ID #{face_id}" 
                            
                            # Face recognized
                            else:
                                # Tracker Logic
                                tracker = face_trackers[face_id]
                                
                                if tracker:
                                    if not tracker['complete']:
                                        if tracker['samples_collected'] < SAMPLES_TO_COLLECT:
                                            # Store Vector + Pose Score + Sharpness Score
                                            tracker['best_samples'].append({
                                                'vector': face_encoding, 
                                                'quality': pose_score,
                                                'sharpness': sharpness
                                            })
                                            
                                            tracker['samples_collected'] += 1
                                            color = (255, 165, 0) # Orange
                                            status = f"ID #{face_id}: {tracker['samples_collected']}/{SAMPLES_TO_COLLECT}"
                                        
                                        else:
                                            finalize_face_vector(face_id)
                                            color = (0, 255, 0) # Green
                                            status = f"ID #{face_id} SAVED"
                                    
                                    else:
                                        color = (0, 255, 0) # Green
                                        status = f"ID #{face_id}"
                                           
                        else:
                            status = "Embed Fail"
                            color = (0, 0, 255)

                    elif pose_score > POSE_QUALITY_THRESHOLD_ID: # ID MODE
                        #print("DEBUG ID MODE")
                        # Image is Sharp & Good Angle -> Get Embedding
                        face_encoding = get_deepface_embedding(face_crop)

                        if face_encoding is not None:
                            # Identify
                            face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids)

                            if face_id is not None:    
                                # Tracker Logic
                                tracker = face_trackers.get(face_id)

                                if tracker:
                                    if not tracker['complete']:
                                            color = (255, 165, 0) # Orange
                                            status = f"ID #{face_id}: {tracker['samples_collected']}/{SAMPLES_TO_COLLECT}"

                                    else:
                                        color = (0, 255, 0) # Green
                                        status = f"ID #{face_id}"
                                else:
                                    status = "Embed Fail"
                                    color = (0, 0, 255)

                    else:
                        status = f"Bad Angle ({int(pose_score*100)}%)"
                        color = (100, 100, 100) # Grey

                else:
                    status = f"Blurry ({int(sharpness)})"
                    color = (0, 0, 255) # Red

                # --- DRAWING ---
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, status, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        cv2.imshow('Face Capture Alpha', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()