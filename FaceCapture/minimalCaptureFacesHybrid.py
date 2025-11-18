import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import mediapipe as mp
from deepface import DeepFace
import DB_Link

# --- CONFIGURATION ---
DB_Link.db_link.initialize()
DB_Link.db_link.clear_db()

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
face_trackers = {}

# Load existing vectors
try:
    vectors_dict = DB_Link.db_link.get_all_vectors()
    for face_id_str, vector_list in vectors_dict.items():
        face_id = int(face_id_str)
        known_face_ids.append(face_id)
        known_face_encodings.append(np.array(vector_list))
        next_face_id = max(next_face_id, face_id + 1)
        face_trackers[face_id] = {'samples': [], 'complete': True}
    print(f"Loaded {len(known_face_ids)} existing faces.")
except Exception as e:
    print(f"Error loading: {e}")

# SIMPLIFIED POSE QUALITY
def get_pose_quality_score(landmarks):
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    left_eye = points_3d[33]
    right_eye = points_3d[263]
    eye_vector = right_eye - left_eye
    roll_penalty = abs(eye_vector[1]) / np.linalg.norm(eye_vector)
    return max(0, 1.0 - roll_penalty * 2)

# SIMPLIFIED RECOGNITION
def recognize_face_simple(face_encoding, known_encodings, known_ids):
    if face_encoding is None or not known_encodings:
        return None
    
    best_similarity = 0
    best_match_id = None
    
    for face_id, known_encoding in zip(known_ids, known_encodings):
        similarity = np.dot(face_encoding, known_encoding) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding))
        
        if similarity > best_similarity and similarity >= RECOGNITION_THRESHOLD:
            best_similarity = similarity
            best_match_id = face_id
    
    if best_match_id:
        print(f"Simple match: Face #{best_match_id} (similarity: {best_similarity:.3f})")
    
    return best_match_id

def get_deepface_embedding(face_crop):
    try:
        # Convert BGR to RGB for DeepFace
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        embeddings = DeepFace.represent(img_path=face_crop_rgb, model_name=DEEPFACE_MODEL, align=True, enforce_detection=False)
        return np.array(embeddings[0]['embedding']) if embeddings else None
    except Exception as e:
        print(f"DeepFace error: {e}")
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
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        face_detected = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box
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
                
                # Skip small faces
                if (right - left) < 50 or (bottom - top) < 50:
                    print(f"Face too small: {right-left}x{bottom-top}")
                    continue
                
                # Check pose
                pose_quality = get_pose_quality_score(face_landmarks)
                print(f"Pose quality: {pose_quality:.3f}")
                
                if pose_quality < POSE_QUALITY_THRESHOLD:
                    print(f"Pose too poor: {pose_quality:.3f} < {POSE_QUALITY_THRESHOLD}")
                    # Draw yellow box for poor pose
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, "Poor Pose", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    continue
                
                # Extract face crop
                face_crop = frame[top:bottom, left:right]
                if face_crop.size == 0:
                    print("Empty face crop")
                    continue
                
                # Resize face crop if too small for DeepFace
                if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                    face_crop = cv2.resize(face_crop, (100, 100))
                
                print(f"Face crop size: {face_crop.shape}")
                
                # Get embedding
                face_encoding = get_deepface_embedding(face_crop)
                if face_encoding is None:
                    print("Failed to get embedding")
                    # Draw red box for embedding failure
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, "Embedding Failed", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue
                
                # SIMPLE RECOGNITION
                face_id = recognize_face_simple(face_encoding, known_face_encodings, known_face_ids)
                
                # New face logic
                if face_id is None:
                    face_id = next_face_id
                    face_trackers[face_id] = {'samples': [], 'complete': False}
                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(face_id)
                    
                    face_trackers[face_id]['samples'].append({
                        'vector': face_encoding,
                        'crop': face_crop
                    })
                    
                    print(f"NEW FACE: ID #{face_id}")
                    next_face_id += 1
                
                # Drawing - Green for successful recognition
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"Face #{face_id}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                face_detected = True
        
        # If no faces detected, show message
        if not face_detected:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Simple Face Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # Spacebar to pause
            cv2.waitKey(0)

DB_Link.db_link.close()
cap.release()
cv2.destroyAllWindows()