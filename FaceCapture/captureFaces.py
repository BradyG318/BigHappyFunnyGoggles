import os
import cv2
import time
import numpy as np
from datetime import datetime
import mediapipe as mp

# --- CONFIGURATION ---
SAVE_FOLDER = "captured_faces"
MAX_FRAMES_PER_PERSON = 20
# Max vector distance for a match (lower is stricter). Using Euclidean distance.
RECOGNITION_THRESHOLD = 0.48
# Set the ID of the specific face you want to save. Set to None to save all faces.
TARGET_FACE_ID = None
# --- END CONFIGURATION ---

# --- RECOGNITION DATABASE ---

next_face_id = 1
known_face_encodings = []
known_face_ids = []

face_trackers = {}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- HELPER FUNCTION: LANDMARK EMBEDDING ---

def get_face_embedding(landmarks):
    """
    Creates a rotation- and translation-normalized feature vector 
    from the 468 facial landmarks for robust recognition.
    """
    # 1. Calculate centroid (average position) for translation normalization
    all_x = [lm.x for lm in landmarks.landmark]
    all_y = [lm.y for lm in landmarks.landmark]
    centroid_x = np.mean(all_x)
    centroid_y = np.mean(all_y)
    
    # 2. Get normalized coordinates relative to the centroid
    normalized_landmarks = []
    for lm in landmarks.landmark:
        normalized_landmarks.append((lm.x - centroid_x, lm.y - centroid_y))

    # 3. Calculate rotation angle for tilt correction
    
    # Use key points around the eyes for a stable horizontal reference:
    # Landmark 33 (left outer eye corner) and 263 (right outer eye corner)
    p1 = normalized_landmarks[33]
    p2 = normalized_landmarks[263]
    
    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    
    # Calculate angle (in radians) of the line between p1 and p2 relative to the x-axis
    angle = np.arctan2(delta_y, delta_x)
    
    # The angle we need to rotate by is -angle to make the eyes horizontal
    cos_theta = np.cos(-angle)
    sin_theta = np.sin(-angle)
    
    # 4. Apply 2D Rotation to all normalized landmarks
    flat_vector = []
    for x, y in normalized_landmarks:
        # Rotation matrix: x' = x*cos - y*sin, y' = x*sin + y*cos
        rotated_x = x * cos_theta - y * sin_theta
        rotated_y = x * sin_theta + y * cos_theta
        flat_vector.extend([rotated_x, rotated_y])
        
    return np.array(flat_vector)

# --- MAIN EXECUTION ---

os.makedirs(SAVE_FOLDER, exist_ok=True)

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

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)

        current_frame_data = [] 

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                face_encoding = get_face_embedding(face_landmarks)
                face_id = None
                
                if known_face_encodings:
                    distances = [np.linalg.norm(known_enc - face_encoding) 
                                 for known_enc in known_face_encodings]
                    
                    best_match_index = np.argmin(distances)
                    
                    if distances[best_match_index] < RECOGNITION_THRESHOLD:
                        face_id = known_face_ids[best_match_index]
                
                if face_id is None:
                    face_id = next_face_id
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    folder_name = f"Face_{face_id}_{timestamp}"
                    person_folder_path = os.path.join(SAVE_FOLDER, folder_name)
                    os.makedirs(person_folder_path, exist_ok=True)
                    
                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(face_id)
                    
                    face_trackers[face_id] = {
                        'folder': person_folder_path,
                        'frames_saved': 0
                    }
                    next_face_id += 1
                    print(f"NEW FACE DETECTED: Face ID #{face_id}. Saving to: {person_folder_path}")

                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                top = int(min(y_coords))
                bottom = int(max(y_coords))
                left = int(min(x_coords))
                right = int(max(x_coords))
                
                buffer_x = int((right - left) * 0.1)
                buffer_y = int((bottom - top) * 0.1)
                
                top = max(0, top - buffer_y)
                bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x)
                right = min(w, right + buffer_x)


                current_frame_data.append((face_id, (top, right, bottom, left)))


        # --- DRAWING & SAVING LOGIC ---
        
        for face_id_to_draw, (top, right, bottom, left) in current_frame_data:
            
            is_target = TARGET_FACE_ID is None or face_id_to_draw == TARGET_FACE_ID
            
            tracker = face_trackers.get(face_id_to_draw)
            frames_saved = tracker.get('frames_saved', 0) if tracker else 0
            save_limit = MAX_FRAMES_PER_PERSON
            
            
            if is_target:
                if frames_saved < save_limit:
                    save_status = f"SAVING ({frames_saved}/{save_limit})"
                    color = (0, 255, 0)
                    
                    face_crop = frame[top:bottom, left:right].copy()
                    if face_crop.size > 0 and tracker:
                        save_path = os.path.join(tracker['folder'], f"frame_{frames_saved:03d}.jpg")
                        cv2.imwrite(save_path, face_crop)
                        tracker['frames_saved'] += 1
                else:
                    save_status = "SAVE COMPLETE"
                    color = (0, 0, 255)
            else:
                save_status = "NON-TARGET (Skip Save)"
                color = (255, 165, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.putText(frame, f"Face #{face_id_to_draw}", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{save_status}", (left, top - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # --- DISPLAY STATUS ---
        
        num_faces = len(current_frame_data)
        
        if num_faces > 0:
            status = f"RECOGNIZING {num_faces} FACES"
        else:
            status = "NO FACE DETECTED"
            
        info_text = f"Total Unique People: {next_face_id - 1}. Target: {'ALL' if TARGET_FACE_ID is None else f'#{TARGET_FACE_ID}'}"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.namedWindow('Landmark-Based Face Recognition Tracker', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Landmark-Based Face Recognition Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Landmark-Based Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
 
cv2.destroyAllWindows()
cap.release()
