import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mediapipe as mp
from deepface import DeepFace
import os
import json
import math

# --- CONFIGURATION ---
# Local storage
LOCAL_DB_FOLDER = "rm_db"
LOCAL_DB_FILE = os.path.join(LOCAL_DB_FOLDER, "face_vectors.json")
os.makedirs(LOCAL_DB_FOLDER, exist_ok=True)

# Face Collection Config
SAMPLES_TO_COLLECT = 30 
BEST_SAMPLES_TO_AVERAGE = 10 

# Models
DEEPFACE_MODEL = 'Facenet512'
mp_face_mesh = mp.solutions.face_mesh

#Recongition Threshold 
RECOGNITION_THRESHOLD = 0.90

# Overall threshold remains low to allow maximum ROM based on reduced penalties below.
POSE_QUALITY_THRESHOLD = 0.30 

# Sharpness: Laplacian Variance. < 50 is usually very blurry.
MIN_SHARPNESS_THRESHOLD = 60.0 

# DATA STRUCTURES
next_face_id = 1
known_face_encodings = []
known_face_ids = []
face_trackers = {} 

# --- HELPER FUNCTIONS ---

def save_known_faces_locally():
    """Saves known_face_ids and known_face_encodings to a JSON file."""
    try:
        data = {
            "next_face_id": next_face_id,
            "faces": [
                {
                    "id": known_face_ids[i],
                    "vector": known_face_encodings[i].tolist() 
                }
                for i in range(len(known_face_ids))
            ]
        }
        with open(LOCAL_DB_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(known_face_ids)} faces locally to {LOCAL_DB_FILE}")
    except Exception as e:
        print(f"Error saving local file: {e}")

def get_image_sharpness(image):
    """Returns the variance of the Laplacian (sharpness score). Higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

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
    save_known_faces_locally()

def get_robust_pose_quality(landmarks):
    """
    Robust score (0.0 to 1.0) checking Roll, Yaw, and Pitch.
    """
    lm = landmarks.landmark
    l_eye = np.array([lm[33].x, lm[33].y])
    r_eye = np.array([lm[263].x, lm[263].y])
    nose = np.array([lm[1].x, lm[1].y])
    lip = np.array([lm[13].x, lm[13].y])
    
    # 1. ROLL (Head Tilt Sideways)
    dY = r_eye[1] - l_eye[1]; dX = r_eye[0] - l_eye[0]
    angle = math.degrees(math.atan2(dY, dX)) 
    # Divisor increased from 45.0 to 60.0 for more tilt freedom
    roll_penalty = abs(angle) / 60.0 
    
    # 2. YAW (Head Turn Left/Right)
    eye_center_x = (l_eye[0] + r_eye[0]) / 2
    eye_width = np.linalg.norm(r_eye - l_eye)
    yaw_deviation = abs(nose[0] - eye_center_x) / eye_width
    # Multiplier reduced from 2.5 to 1.8 for maximum side-to-side freedom
    yaw_penalty = yaw_deviation * 1.8 
    
    # 3. PITCH (Head Look Up/Down)
    eye_line_y = (l_eye[1] + r_eye[1]) / 2
    nose_to_eye = abs(nose[1] - eye_line_y)
    nose_to_lip = abs(lip[1] - nose[1])
    if nose_to_lip == 0: nose_to_lip = 0.001
    ratio = nose_to_eye / nose_to_lip
    
    pitch_penalty = 0
    # Limits widened aggressively: 0.5 -> 0.4 and 2.0 -> 2.5
    if ratio < 0.4: pitch_penalty = (0.4 - ratio) # Look up penalty
    elif ratio > 2.5: pitch_penalty = (ratio - 2.5) # Look down penalty
    
    total_penalty = roll_penalty + yaw_penalty + pitch_penalty
    score = max(0, 1.0 - total_penalty)
    return score

def recognize_face_simple(face_encoding, known_encodings, known_ids):
    if face_encoding is None or not known_encodings: return None
    best_sim = 0
    best_id = None
    for fid, k_enc in zip(known_ids, known_encodings):
        sim = np.dot(face_encoding, k_enc)
        if sim > best_sim and sim >= RECOGNITION_THRESHOLD:
            best_sim = sim
            best_id = fid
    return best_id

def get_deepface_embedding(face_crop):
    try:
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        emb = DeepFace.represent(img_path=rgb, model_name=DEEPFACE_MODEL, align=True, enforce_detection=False)
        return np.array(emb[0]['embedding']) if emb else None
    except: return None

# --- LOAD DATA ---
try:
    if os.path.exists(LOCAL_DB_FILE):
        with open(LOCAL_DB_FILE, 'r') as f:
            data = json.load(f)
        next_face_id = data.get("next_face_id", 1)
        for fd in data.get("faces", []):
            known_face_ids.append(fd['id'])
            known_face_encodings.append(np.array(fd['vector']))
            face_trackers[fd['id']] = {'samples_collected': SAMPLES_TO_COLLECT, 'complete': True, 'best_samples': []}
        print(f"Loaded {len(known_face_ids)} existing faces.")
except Exception as e: print(f"Load error: {e}")

# ----------------------------------------------------------------------
# --- MAIN LOOP ---
# ----------------------------------------------------------------------
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    print("System Ready. Press 'Esc' to exit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = frame.shape[:2]
                x_c = [lm.x * w for lm in face_landmarks.landmark]
                y_c = [lm.y * h for lm in face_landmarks.landmark]
                left, right = int(min(x_c)), int(max(x_c))
                top, bottom = int(min(y_c)), int(max(y_c))
                
                # Padding
                pad = 20
                left = max(0, left-pad); right = min(w, right+pad)
                top = max(0, top-pad); bottom = min(h, bottom+pad)
                
                # --- STATUS VARIABLES ---
                status = "Processing..."
                color = (0, 0, 255) # Red (Default)
                
                # 1. Size Check
                if (right-left) < 50 or (bottom-top) < 50:
                    status = "Too Small"
                    color = (100, 100, 100) # Grey
                
                else:
                    # 2. Robust Pose Check (MAX ROM)
                    pose_score = get_robust_pose_quality(face_landmarks)
                    
                    if pose_score < POSE_QUALITY_THRESHOLD:
                        status = f"Bad Angle ({int(pose_score*100)}%)"
                        color = (100, 100, 100) # Grey
                    else:
                        face_crop = frame[top:bottom, left:right]
                        if face_crop.size > 0:
                            
                            # 3. SHARPNESS CHECK
                            sharpness = get_image_sharpness(face_crop)
                            
                            if sharpness < MIN_SHARPNESS_THRESHOLD:
                                status = f"Blurry ({int(sharpness)})"
                                color = (0, 0, 255) # Red
                            else:
                                # Image is Sharp & Good Angle -> Get Embedding
                                face_encoding = get_deepface_embedding(face_crop)
                                
                                if face_encoding is not None:
                                    # 4. Identify / Check-In
                                    face_id = recognize_face_simple(face_encoding, known_face_encodings, known_face_ids)
                                    
                                    # Check-In New Person
                                    if face_id is None:
                                        face_id = next_face_id
                                        next_face_id += 1
                                        face_trackers[face_id] = {'samples_collected': 0, 'complete': False, 'best_samples': []}
                                        known_face_ids.append(face_id)
                                        known_face_encodings.append(face_encoding)
                                        print(f"NEW PERSON: ID #{face_id}")

                                    # Tracker Logic
                                    tracker = face_trackers.get(face_id)
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

                # --- DRAWING ---
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, status, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Maximum ROM Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

save_known_faces_locally()
cap.release()
cv2.destroyAllWindows()