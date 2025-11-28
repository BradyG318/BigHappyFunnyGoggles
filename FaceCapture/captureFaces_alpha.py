import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace
import math

# For database storage
#import DB_Link

# For local storage
import os
import json

# --- CONFIGURATION ---
# DATABASE INITIALIZATION
# DB_Link.db_link.initialize()
# DB_Link.db_link.clear_db()

# LOCAL STORAGE INITIALIZATION
LOCAL_DB_FOLDER = "rm_db"
LOCAL_DB_FILE = os.path.join(LOCAL_DB_FOLDER, "face_vectors.json")
os.makedirs(LOCAL_DB_FOLDER, exist_ok=True)

# AI MODELS
DEEPFACE_MODEL = 'Facenet512'
mp_face_mesh = mp.solutions.face_mesh

# PARAMETERS
CAMERA_INDEX = 0

ID_THRESHOLD = 0.80 # Cosine similarity threshold for ID case (0 - 1)
CAPTURE_THRESHOLD = 0.85 # Cosine similarity threshold for capture case (0 - 1)

MIN_SAMPLES_FOR_AVERAGE = 30 # Minimum samples required to compute average vector

# POSE THRESHOLDS
# ID is looser, CAPTURE is tighter

# PITCH THRESHOLDS (looking up/down)
PITCH_RATIO_LOW_ID = 0.3    # Looking up threshold ID - - - - - - - - - - - |: ID WINDOW (0.3 - 2.0)
PITCH_RATIO_LOW_CAPTURE = 0.9    # Looking up threshold CAPTURE - - - |     |
#                                                                     |     |
PITCH_RATIO_HIGH_ID = 2.0   # Looking down threshold ID - - - - - - - | - - |
PITCH_RATIO_HIGH_CAPTURE = 1.5   # Looking down threshold CAPTURE - - |: CAPTURE WINDOW (0.9 - 1.5)

# TILT THRESHOLDS (head rotation - ear to shoulder)
TILT_ANGLE_THRESHOLD_ID = 15 # degrees ID
TILT_ANGLE_THRESHOLD_CAPTURE = 10 # degrees CAPTURE

# YAW THRESHOLD (head turning - left/right)
YAW_RATIO_THRESHOLD_ID = 0.32  # Absolute value for left/right turning ID
YAW_RATIO_THRESHOLD_CAPTURE = 0.25  # Absolute value for left/right turning CAPTURE

# DATA STRUCTURES
next_face_id = 1
known_face_encodings = [] 
known_face_ids = []
face_trackers = {}

# --- FUNCTIONS ---
def calculate_3d_angle(point1, point2, point3):
    """Calculate angle between three 3D points in degrees"""
    # Convert to vectors
    v1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    v2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
    
    # Dot product
    dot_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    
    # Magnitudes
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
    
    # Avoid division by zero
    if mag1 * mag2 == 0:
        print("ERROR: DIVIDE BY 0")
        return 0
    
    # Calculate angle in radians and convert to degrees
    angle_rad = math.acos(max(-1, min(1, dot_product / (mag1 * mag2))))
    return math.degrees(angle_rad)

def pitch_limit(face_landmarks, is_Capture=False):
    """
    Simple pitch detection using vertical positions
    Returns True if pitch exceeds threshold
    """
    # Key vertical landmarks
    left_eye = face_landmarks.landmark[33]      # Left eye corner
    nose_tip = face_landmarks.landmark[1]       # Nose tip  
    bottom_lip = face_landmarks.landmark[14]    # Bottom lip
    
    # Calculate vertical distances (normalized coordinates)
    eye_to_nose = nose_tip.y - left_eye.y
    nose_to_mouth = bottom_lip.y - nose_tip.y
    
    # The ratio should be relatively constant for straight head
    ratio = eye_to_nose / (nose_to_mouth + 0.0001)
    
    # print(f"Eye-nose-mouth ratio: {ratio:.2f}")
    
    # If checking capture threshold
    if is_Capture:
        PITCH_RATIO_LOW = PITCH_RATIO_LOW_CAPTURE
        PITCH_RATIO_HIGH = PITCH_RATIO_HIGH_CAPTURE

    # Otherwise check id threshold
    else:
        PITCH_RATIO_LOW = PITCH_RATIO_LOW_ID
        PITCH_RATIO_HIGH = PITCH_RATIO_HIGH_ID

    # If ratio is too small (looking up) or too large (looking down)
    if ratio < PITCH_RATIO_LOW or ratio > PITCH_RATIO_HIGH:
        #print(f"Pitch limit exceeded - ratio: {ratio:.2f}")
        return True
    
    return False

def tilt_limit(face_landmarks, is_Capture=False):
    """
    Tilt detection using the angle of the line between eyes
    Returns True if tilt exceeds threshold
    """
    # Use outer eye corners
    left_eye_outer = face_landmarks.landmark[33]    # Left eye outer corner
    right_eye_outer = face_landmarks.landmark[263]  # Right eye outer corner
    
    # Calculate the angle of the eye line
    delta_y = right_eye_outer.y - left_eye_outer.y
    delta_x = right_eye_outer.x - left_eye_outer.x
    
    # Calculate angle in degrees (-90 to +90)
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    
    #print(f"Tilt angle: {angle_deg:.1f}°")
    
    # If checking capture threshold
    if is_Capture:
        TILT_ANGLE_THRESHOLD = TILT_ANGLE_THRESHOLD_CAPTURE

    # Otherwise check id threshold
    else:
        TILT_ANGLE_THRESHOLD = TILT_ANGLE_THRESHOLD_ID

    # If the eye line is not horizontal enough
    if abs(angle_deg) > TILT_ANGLE_THRESHOLD:  # Allow ±TILT_ANGLE_THRESHOLD degrees of tilt
        # print(f"Tilt limit exceeded - angle: {angle_deg:.1f}°")
        return True
    
    return False

def yaw_limit(face_landmarks, is_Capture=False):
    """
    Simple yaw detection using nose position relative to eyes
    Returns True if yaw exceeds threshold
    """
    # Key landmarks for yaw detection
    left_eye = face_landmarks.landmark[33]      # Left eye corner
    right_eye = face_landmarks.landmark[263]    # Right eye corner
    nose_tip = face_landmarks.landmark[1]       # Nose tip
    
    # Calculate eye center
    eye_center_x = (left_eye.x + right_eye.x) / 2
    
    # Calculate how far nose is from center (normalized)
    nose_center_ratio = (nose_tip.x - eye_center_x) / (abs(right_eye.x - left_eye.x) + 0.0001)
    
    #print(f"Yaw ratio: {nose_center_ratio:.2f}")
    
    # If checking capture threshold
    if is_Capture:
        YAW_RATIO_THRESHOLD = YAW_RATIO_THRESHOLD_CAPTURE
    
    # Otherwise check id threshold
    else:
        YAW_RATIO_THRESHOLD = YAW_RATIO_THRESHOLD_ID

    # If nose is too far from center
    if abs(nose_center_ratio) > YAW_RATIO_THRESHOLD:
        #print(f"Yaw limit exceeded - ratio: {nose_center_ratio:.2f}")
        return True
    
    return False

def is_pose_valid_ID(face_landmarks):
    # Check if pose violates any pitch/tilt/yaw ID threshold
    if (pitch_limit(face_landmarks) or tilt_limit(face_landmarks) or yaw_limit(face_landmarks)):
        return False
    
    return True

def is_pose_valid_CAPTURE(face_landmarks):
    # Check if pose violates any pitch/tilt/yaw CAPTURE threshold
    if (pitch_limit(face_landmarks, True) or tilt_limit(face_landmarks, True) or yaw_limit(face_landmarks, True)):
        return False
    
    return True

def is_quality_face(face_crop):
    """
    Check if face image has acceptable quality for recognition
    """
    if face_crop is None:
        return False
    
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Reject overexposed or underexposed faces
    if mean_brightness > 220 or mean_brightness < 30:
        return False
    
    # Reject low contrast faces
    if std_brightness < 20:
        return False
    
    return True

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
            # For moderate lighting issues, apply very mild normalization
            l_normalized = cv2.normalize(l_channel, None, 50, 200, cv2.NORM_MINMAX)
            lab[:,:,0] = l_normalized
            normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return normalized
            
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
            # enforce_detection=False,
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

def recognize_face(face_encoding, known_face_encodings, known_ids, recognition_threshold):
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

# def save_data_to_database(face_id, samples):
#     """
#     Finalizes the data collection, calculates the final average vector,
#     RENORMALIZES and saves the vector to PostgreSQL database.
#     """
#     print(f"\n--- DATABASE SAVE START: Face ID #{face_id} ---")
    
#     # Calculate the final, averaged face vector
#     vectors = np.array([s['vector'] for s in samples])
#     avg_vector = np.mean(vectors, axis=0)
    
#     # Re normalize to correct averaging error
#     final_vector = avg_vector / np.linalg.norm(avg_vector)

#     # Save the final vector to database synchronously
#     success = DB_Link.db_link.save_face_vector(face_id, final_vector.tolist())
    
#     if not success:
#         print(f"!!! ERROR saving vector to database for face #{face_id}")
#         return False
    
#     print(f"Vector saved to database for Face ID #{face_id}")
#     print(f"--- DATABASE SAVE COMPLETE: Face ID #{face_id} ---\n")
#     return True

def save_known_faces_locally():
    """Saves known_face_ids and known_face_encodings to a JSON file."""
    try:
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
        return True
    except Exception as e:
        print(f"Error saving local faces: {e}")
        return False

def save_data_locally(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    RENORMALIZES and saves the vector to local JSON file.
    """
    print(f"\n--- LOCAL SAVE START: Face ID #{face_id} ---")
    
    # Calculate the final, averaged face vector
    vectors = np.array([s['vector'] for s in samples])
    avg_vector = np.mean(vectors, axis=0)
    
    # Re normalize to correct averaging error
    final_vector = avg_vector / np.linalg.norm(avg_vector)

    # Update the in-memory encoding with the final averaged vector
    idx = known_face_ids.index(face_id)
    known_face_encodings[idx] = final_vector

    # Save all known faces to local JSON file
    success = save_known_faces_locally()
    
    if not success:
        print(f"!!! ERROR saving vector locally for face #{face_id}")
        return False
    
    print(f"Vector saved locally for Face ID #{face_id}")
    print(f"--- LOCAL SAVE COMPLETE: Face ID #{face_id} ---\n")
    return True

# --- MAIN ---
# Load existing vectors from database
# try:
#     vectors_dict = DB_Link.db_link.get_all_vectors()
#     for face_id_str, vector_list in vectors_dict.items():
#         face_id = int(face_id_str)
#         known_face_ids.append(face_id)
#         known_face_encodings.append(np.array(vector_list))
#         next_face_id = max(next_face_id, face_id + 1)
        
#         face_trackers[face_id] = {
#             'samples': [],
#             'complete': True 
#         }

#     print(f"Loaded {len(known_face_ids)} existing faces from database. Next ID will be {next_face_id}.")
    
# except Exception as e:
#     print(f"Error loading from database: {e}. Starting fresh.")

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

        # Copy frames for detection and recognition, while keeping original for display
        detection_frame = frame.copy()
        recognition_frame = frame.copy()

        # Convert the BGR frame to RGB for mediapipe
        detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find face landmarks
        results = face_mesh.process(detection_frame)

        # Initialize / reset current data in frame
        current_frame_data = []

        # If there are face landmarks detected
        if results.multi_face_landmarks:
            # Loop through face detections
            for face_landmarks in results.multi_face_landmarks:
                # Initialize capture boolean to false
                capture = False

                # Get bounding box (using MediaPipe landmarks)
                h, w = detection_frame.shape[:2]
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                top = int(min(y_coords)); bottom = int(max(y_coords))
                left = int(min(x_coords)); right = int(max(x_coords))

                # Skip small detections - uncomment later
                face_width = right - left
                face_height = bottom - top
                if face_width < 80 or face_height < 80:
                    print(f"Skipped small face detection: {face_width}x{face_height}px")
                    continue
                
                # Check level of pose validity
                if is_pose_valid_CAPTURE(face_landmarks):
                    capture = True

                elif is_pose_valid_ID(face_landmarks):
                    capture = False

                else:
                    print(f"Skipped invalid pose.")
                    continue

                # Add buffer for cropping and update coords
                buffer_x = int(face_width * 0.15)
                buffer_y = int(face_height * 0.15)
                
                top = max(0, top - buffer_y); bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x); right = min(w, right + buffer_x)

                # Crop face region
                face_crop = recognition_frame[top:bottom, left:right]

                # Check if lighting quality target met
                if not is_quality_face(face_crop):
                    print("Skipping low quality face due to lighting")
                    continue
                
                # Apply lighting normalization
                face_crop_final = conservative_lighting_normalization(face_crop)

                # Generate embedding using DeepFace
                face_encoding = get_deepface_embedding(face_crop_final)

                if face_encoding is None:
                    continue
                
                # Store original encoding for potential consistency tracking - currently scrapped this approach again but maybe something to it for the future
                original_encoding = face_encoding.copy()

                # Initialize the data dictionary for this face detection
                data = {
                    'id': None, # Placeholder
                    'box': (top, right, bottom, left),
                    'tracker': None, # Placeholder
                    'crop': face_crop_final,
                    'encoding': face_encoding,
                    'capture mode' : capture
                }

                # Initialize face_id
                face_id = None
                
                # Check if capture threshold passed
                if data['capture mode']:
                    #DEBUG
                    print("CAPTURE THRESHOLD CASE")

                    # Check to ID like usual
                    if known_face_encodings:
                        face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, CAPTURE_THRESHOLD)

                    # New face (or just not recognized)
                    if face_id is None:
                        # Assign new id
                        face_id = next_face_id
                        next_face_id += 1

                        # Initialize face_trackers dictionary entry at new id
                        face_trackers[face_id] = {
                            'samples': [],
                            'complete': False
                        }

                        # Add new encoding and id to known lists
                        known_face_encodings.append(face_encoding)
                        known_face_ids.append(face_id)
                    
                        print(f"NEW FACE: ID #{face_id}")
                        
                        # Add encoding and crop to the samples list from the face_trackers dictionart entry at new id
                        face_trackers[face_id]['samples'].append({
                            'vector': face_encoding,
                            'crop': face_crop_final,
                        })

                        # Assign tracker and id for new entry
                        data['tracker'] = face_trackers[face_id]
                        data['id'] = face_id

                        # Add detection data to current frame data
                        current_frame_data.append(data)
                    
                    # Known faces which have no tracker or id
                    elif (data['tracker'] is None or data['id'] is None):
                        # Assign tracker and id for new entry
                        data['tracker'] = face_trackers[face_id]
                        data['id'] = face_id

                        # Add detection data to current frame data
                        current_frame_data.append(data)

                # We already checked earlier to ensure ID threshold met
                else:
                    #DEBUG
                    print("ID THRESHOLD CASE")

                    # Attempt to identify with a lower threshold, if still fails so be it
                    if known_face_encodings:
                        face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, ID_THRESHOLD)
                        
                        # If face identified but tracker and id not yet initialized
                        if face_id is not None:
                            if (data['tracker'] is None or data['id'] is None):
                                # Assign tracker and id for new entry
                                data['tracker'] = face_trackers[face_id]
                                data['id'] = face_id
                        
                            # Add detection data to current frame data
                            current_frame_data.append(data)
                        
                        else:
                            continue # Do not draw/sample/anything in this ID only case

        # --- SAMPLING ---
        # Loop through all current data
        for data in current_frame_data:
            # Unpack data for a detection
            face_id = data['id']
            top, right, bottom, left = data['box']

            tracker = data['tracker']

            capture_mode = data['capture mode']

            # Initialize status variables
            color = (0, 255, 0)  # Default: Green (collecting)
            status = f"COLLECTING"

            if not tracker['complete'] and capture_mode:
                # If all samples not collected
                if len(tracker['samples']) < MIN_SAMPLES_FOR_AVERAGE:
                    # Add vector and crop to samples
                    tracker['samples'].append({
                    'vector': data['encoding'],
                    'crop': data['crop']
                    })

                    color = (0, 255, 0) # Green: Collecting
                    status = f"COLLECTING ({len(tracker['samples'])}/{MIN_SAMPLES_FOR_AVERAGE})" 
                # If all samples collected
                else:
                    # Save possible
                    if save_data_locally(face_id, tracker['samples']):
                        # Mark sample collection complete
                        tracker['complete'] = True

                        color = (0, 0, 255) # Red: Complete (Successful save)
                        status = "SAVE SUCCESS"

                        # Update with final vector
                        idx = known_face_ids.index(face_id)
                        final_vector = np.mean(np.array([s['vector'] for s in tracker['samples']]), axis=0)
                        known_face_encodings[idx] = final_vector

                        print(f"Finalized Face ID #{face_id}")
                    # Save failed
                    else:
                        color = (0, 165, 255) # Orange: Complete (Failed save)
                        status = "SAVE FAILED"
            # If tracker complete
            else:
                # If face actually complete
                if (tracker['complete']):
                    color = (0, 0, 255) # Red: Complete (Recognized)
                    status = "RECOGNIZED"
                
                # If not in capture mode but still incomplete (ID threshold only)
                else:
                    color = (255, 255, 0)  # Yellow: ID only, not collecting
                    status = f"ID ONLY ({len(tracker['samples'])}/{MIN_SAMPLES_FOR_AVERAGE})"
        
            # --- DRAWING ---
            display_id = f"Face #{face_id}"

            # Update flag to allow drawing
            frame.flags.writeable = True
            
            # Draw detection
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, display_id, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{status}", (left, top - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # --- DISPLAY STATUS ---
        info_text = f"Total Unique People: {next_face_id - 1}. Model: {DEEPFACE_MODEL}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_text = f"Threshold: {ID_THRESHOLD} | Samples: {MIN_SAMPLES_FOR_AVERAGE}"
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('DeepFace + MediaPipe Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Close resources on exit
# DB_Link.db_link.close()
cap.release()
cv2.destroyAllWindows()