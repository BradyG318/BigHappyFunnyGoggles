import os
import cv2
import numpy as np
import json
import warnings

# deals with deprecation warnings from mediapipe
warnings.filterwarnings("ignore", message=".*GetPrototype.*")

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
RECOGNITION_THRESHOLD = 0.63
TARGET_FACE_ID = None # Set to None to save all unique faces

# Pose filtering parameters
MAX_POSE_ANGLE = 25  # Maximum acceptable head turn angle
POSE_FILTERING = True  # Enable/disable pose-based filtering

# Sampling parameters for robust data collection
MIN_SAMPLES_FOR_AVERAGE = 50     # Minimum frames needed before we calculate the final vector
NUM_BEST_FRAMES_TO_SEND = 10      # Number of sharpest frames to save locally
MIN_SHARPNESS_THRESHOLD = 15.0   # Minimum image quality score to consider a frame

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

# --- 3D EMBEDDING FUNCTION ---

def get_face_embedding(landmarks):
    """
    Creates a fully pose-normalized 3D feature vector 
    that handles roll, pitch, and yaw rotations.
    from the 468 facial landmarks for robust recognition.
    """
    # Extract all 3D coordinates
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # 1. CENTER THE FACE
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid
    
    # Define reference points for coordinate system
    # Nose tip - origin
    nose_tip = centered_points[1]  # Landmark 1 is nose tip
    
    # Right eye corner - X-axis direction
    right_eye = centered_points[33]  # Right eye outer corner
    
    # Left eye corner - for calculating X-axis
    left_eye = centered_points[263]  # Left eye outer corner
    
    # Chin - Y-axis direction (downward)
    chin = centered_points[152]  # Chin bottom
    
    # Forehead - for better Y-axis estimation
    forehead = centered_points[10]  # Forehead
    
    # 3. CREATE ROBUST COORDINATE SYSTEM
    try:
        # X-axis: horizontal direction between eyes
        x_axis = right_eye - left_eye
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Temporary Y-axis: from nose to forehead (upward) DEV: maybe chin if needed
        temp_y = forehead - nose_tip
        temp_y = temp_y / np.linalg.norm(temp_y)
        
        # Z-axis: cross product of X and temporary Y
        z_axis = np.cross(x_axis, temp_y)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Recalculate Y-axis from cross product of Z and X
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        # Ensure it's a proper rotation matrix (orthogonal)
        if np.linalg.det(rotation_matrix) < -1e-6:
            # Fix handedness if needed
            z_axis = -z_axis
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
    except (ValueError, np.linalg.LinAlgError):
        # Fallback: use simple eye-based rotation (your original method)
        print("Warning: Using fallback rotation")
        # return get_face_embedding_fallback(landmarks)
    
    # 4. ROTATE ALL POINTS TO CANONICAL POSE
    rotated_points = centered_points @ rotation_matrix.T  # Matrix multiplication
    
    # 5. ADD ROBUST FEATURE SELECTION
    feature_vector = []
    
    # Relative distances between key points (scale-invariant)
    key_distances = {
        (33, 263),   # Eye corners width
        (1, 152),    # Nose to chin
        (10, 152),   # Forehead to chin  
        (33, 1),     # Right eye to nose
        (263, 1),    # Left eye to nose
        (61, 291),   # Mouth corners
        (159, 386),  # Eye heights
    }
    
    for idx1, idx2 in key_distances:
        if idx1 < len(rotated_points) and idx2 < len(rotated_points):
            dist = np.linalg.norm(rotated_points[idx1] - rotated_points[idx2])
            feature_vector.append(dist)
    
    # Add ratios for better scale invariance
    eye_width = np.linalg.norm(rotated_points[33] - rotated_points[263])
    face_height = np.linalg.norm(rotated_points[10] - rotated_points[152])
    if face_height > 0:
        feature_vector.append(eye_width / face_height)
    
    # C. Add region-specific features with different weighting
    high_weight_regions = {
        'eyes': [33, 133, 157, 158, 159, 160, 161, 246, 7, 163],  # Most stable features
        'nose_bridge': [1, 2, 3, 4, 5, 6],  # Nose shape is very distinctive
    }

    medium_weight_regions = {
        'mouth': [61, 84, 85, 78, 191, 80, 81, 82],
        'eyebrows': [70, 63, 105, 66, 107]
    }

    # Add high-weight regions (repeat to give more importance)
    for region_name, indices in high_weight_regions.items():
        for idx in indices:
            if idx < len(rotated_points):
                feature_vector.extend(rotated_points[idx] * 2.0)  # Double weight

    # Add medium-weight regions
    for region_name, indices in medium_weight_regions.items():
        for idx in indices:
            if idx < len(rotated_points):
                feature_vector.extend(rotated_points[idx])

    # D. Add facial contour but with lower weight
    face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]
    for idx in face_contour:
        if idx < len(rotated_points):
            feature_vector.extend(rotated_points[idx] * 0.5)  # Half weight
    
    return np.array(feature_vector)

    # --- POSE ESTIMATION FOR VISUALIZATION ---

def estimate_head_pose(landmarks, image_size):
    """
    Estimate head pose angles for visualization and filtering
    """
    # 3D model points for basic head pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])
    
    # Corresponding 2D image points
    image_points = np.array([
        [landmarks.landmark[1].x * image_size[1], landmarks.landmark[1].y * image_size[0]],  # Nose tip
        [landmarks.landmark[152].x * image_size[1], landmarks.landmark[152].y * image_size[0]],  # Chin
        [landmarks.landmark[33].x * image_size[1], landmarks.landmark[33].y * image_size[0]],  # Left eye
        [landmarks.landmark[263].x * image_size[1], landmarks.landmark[263].y * image_size[0]],  # Right eye
        [landmarks.landmark[61].x * image_size[1], landmarks.landmark[61].y * image_size[0]],  # Left mouth
        [landmarks.landmark[291].x * image_size[1], landmarks.landmark[291].y * image_size[0]]   # Right mouth
    ], dtype=np.float64)
    
    # Camera internals (approximate)
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    
    try:
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            pose_angles = rotationMatrixToEulerAngles(rotation_matrix)
            
            return pose_angles, rotation_vector, translation_vector
    except:
        pass
    
    return None, None, None

def rotationMatrixToEulerAngles(R):
    """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / np.pi  # Convert to degrees

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def recognize_face(face_encoding, known_encodings, known_ids, threshold=0.85):
    """Enhanced recognition using cosine similarity"""
    if not known_encodings:
        return None
    
    best_similarity = -1
    best_match_id = None
    
    for face_id, known_encoding in zip(known_ids, known_encodings):
        similarity = cosine_similarity(face_encoding, known_encoding)
        
        if similarity > best_similarity and similarity > threshold:
            best_similarity = similarity
            best_match_id = face_id
    
    return best_match_id

# --- ENHANCED SAMPLING STRATEGY ---

def get_pose_quality_score(pose_angles):
    """Calculate a quality score based on head pose (0-1 scale)"""
    if pose_angles is None:
        return 0.0
    
    pitch, yaw, roll = pose_angles
    
    # Ideal pose is frontal (all angles near 0)
    pitch_score = max(0, 1 - abs(pitch) / 30)  # 30 degrees max
    yaw_score = max(0, 1 - abs(yaw) / 40)      # 40 degrees max  
    roll_score = max(0, 1 - abs(roll) / 20)    # 20 degrees max
    
    # Combined quality score (weighted average)
    quality = 0.4 * yaw_score + 0.3 * pitch_score + 0.3 * roll_score
    return quality

def calculate_weighted_average(samples):
    """Calculate weighted average based on sharpness and pose quality"""
    if not samples:
        return None
    
    vectors = np.array([s['vector'] for s in samples])
    weights = np.array([s['sharpness'] * s.get('pose_quality', 1.0) for s in samples])
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        weighted_avg = np.average(vectors, axis=0, weights=weights)
    else:
        weighted_avg = np.mean(vectors, axis=0)
    
    return weighted_avg

def save_data_locally(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    and saves the vector and best frames to local files.
    """
    print(f"\n--- LOCAL SAVE START: Face ID #{face_id} ---")
    
    # 1. Calculate the final, averaged face vector
    final_vector = calculate_weighted_average(samples)  # Uses sharpness + pose quality
    
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
    min_detection_confidence=0.35,
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
                    # Use cosine similarity function for recognition
                    face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, threshold=0.85)  # Adjust this threshold as needed
                
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
                    
                    pose_angles, _, _ = estimate_head_pose(face_landmarks, (h, w))
                    pose_quality = get_pose_quality_score(pose_angles)

                    # Check sharpness threshold before adding sample
                    if data['sharpness'] >= MIN_SHARPNESS_THRESHOLD:
                        tracker['samples'].append({
                            'vector': data['encoding'],
                            'crop': data['crop'],
                            'sharpness': data['sharpness'],
                            'pose_quality' : pose_quality
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