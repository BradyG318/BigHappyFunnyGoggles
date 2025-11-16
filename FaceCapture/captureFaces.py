import os
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp

# --- CONFIGURATION ---

# This is where your collected data will be stored locally.
DATABASE_FOLDER = "remote_database"
VECTORS_FILE = os.path.join(DATABASE_FOLDER, "vectors.json")
FRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "best_frames")

# Ensure necessary folders exist
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Recognition parameters
RECOGNITION_THRESHOLD = 0.95
TARGET_FACE_ID = None # Set to None to save all unique faces

# Sampling parameters for data collection
MIN_SAMPLES_FOR_AVERAGE = 50     # Minimum frames needed before we calculate the final vector
NUM_BEST_FRAMES_TO_SEND = 10      # Number of sharpest frames to save locally
MIN_SHARPNESS_KNOWN = 15.0   # Minimum image quality score to consider a frame of a known face
MIN_SHARPNESS_UNKNOWN = 25.0   # Minimum image quality score to consider a frame of an unknown face

# --- DATA STRUCTURES ---
next_face_id = 1
known_face_encodings = [] # List of final, averaged vectors
known_face_ids = []       # List of corresponding Face IDs
face_trackers = {} # Tracks state and collected samples for each detected face
currently_tracked_faces = set() # Track currently visible faces to prevent duplicates

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

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

# --- HELPER FUNCTIONS ---

# Function to preprocess the frame for better face detection
def preprocess_frame(image):
    # Reduce compression artifacts
    image = cv2.medianBlur(image, 5)  # Reduce noise aggressively for longer range
    
    # Enhance contrast aggressively for longer range (helps with detection)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Scale image up for better detection of smaller faces
    scale_factor = 1.5  # Increase this if needed (1.5 = 150% size)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

    return image

def get_sharpness_score(image):
    """Calculates the image sharpness using the variance of the Laplacian."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_pose_normalized_embedding(landmarks):
    """
    Essential pose normalization for handling tilted faces
    """
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # 1. Center the face
    centroid = np.mean(points_3d, axis=0)
    centered_points = points_3d - centroid
    
    # 2. Simple pose normalization using eyes and nose
    left_eye = centered_points[33]
    right_eye = centered_points[263]
    nose_tip = centered_points[1]
    forehead = centered_points[10]
    
    # Create basic coordinate system
    try:
        x_axis = right_eye - left_eye
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        temp_y = forehead - nose_tip
        temp_y = temp_y / np.linalg.norm(temp_y)
        
        z_axis = np.cross(x_axis, temp_y)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
    except:
        # Fallback: no rotation
        rotation_matrix = np.eye(3)
    
    # 3. Rotate to canonical pose
    rotated_points = centered_points @ rotation_matrix.T
    
    # 4. Simplified feature selection
    feature_vector = []
    
    # Core facial points (pose-normalized coordinates)
    core_points = [1, 33, 263, 133, 362, 61, 291, 4, 5, 152, 10]  # Key landmarks
    
    for idx in core_points:
        if idx < len(rotated_points):
            feature_vector.extend(rotated_points[idx])
    
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

    #TUNING
    # Add high-weight regions (repeat to give more importance)
    for region_name, indices in high_weight_regions.items():
        for idx in indices:
            if idx < len(rotated_points):
                feature_vector.extend(rotated_points[idx] * 1.15)  # Higher weight

    # Add medium-weight regions
    for region_name, indices in medium_weight_regions.items():
        for idx in indices:
            if idx < len(rotated_points):
                feature_vector.extend(rotated_points[idx])

    # D. Add facial contour but with lower weight
    face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365]
    for idx in face_contour:
        if idx < len(rotated_points):
            feature_vector.extend(rotated_points[idx] * 0.15)  # Lower weight
    
    return np.array(feature_vector)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def recognize_face(face_encoding, known_encodings, known_ids, face_trackers, threshold=RECOGNITION_THRESHOLD):
    """Simple recognition with debug output"""
    if not known_encodings:
        return None
    
    best_similarity = -1
    best_match_id = None
    
    for face_id, known_encoding in zip(known_ids, known_encodings):
        base_similarity = cosine_similarity(face_encoding, known_encoding)

        # Apply weighting based on face status
        if face_id in face_trackers:
            if face_trackers[face_id]['complete']:
                # Completed faces are trusted as is
                weighted_similarity = base_similarity
            else:
                # Incomplete faces - check how many samples they have
                samples_collected = len(face_trackers[face_id]['samples'])
                progress_bonus = min(samples_collected / MIN_SAMPLES_FOR_AVERAGE * 0.06, 0.06) #TUNING
                weighted_similarity = base_similarity + progress_bonus
        else:
            weighted_similarity = base_similarity
        
        if weighted_similarity > best_similarity and weighted_similarity > threshold:
            best_similarity = weighted_similarity
            best_match_id = face_id
    
    # Show similarity for debugging
    if best_match_id:
        print(f"Matched Face #{best_match_id} (similarity: {best_similarity:.3f})")
    else:
        print(f"No match (best similarity: {best_similarity:.3f})")
    
    return best_match_id

def get_sharpness_threshold(face_id):
    """Return appropriate sharpness threshold based on face status"""
    if face_id in face_trackers and face_trackers[face_id]['complete']:
        return MIN_SHARPNESS_KNOWN  # Lower threshold for known faces
    else:
        return MIN_SHARPNESS_UNKNOWN  # Higher threshold for new/uncompleted faces

# --- ENHANCED SAMPLING STRATEGY ---

def save_data_locally(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    and saves the vector and best frames to local files.
    """
    print(f"\n--- LOCAL SAVE START: Face ID #{face_id} ---")
    
    # 1. Calculate the final, averaged face vector
    vectors = np.array([s['vector'] for s in samples])
    final_vector = np.mean(vectors, axis=0)
    
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

cap = cv2.VideoCapture(2)

# Initialize Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.01) as face_mesh:

    print("Starting face capture. Press 'Esc' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip the frame horizontally for a later selfie-view display
        #frame = cv2.flip(frame, 1)
        
        # Preprocess frame
        frame = preprocess_frame(frame)
        frame.flags.writeable = False #improve performance

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Reset current frame data
        current_frame_data = []
        currently_tracked_faces.clear()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                top = int(min(y_coords)); bottom = int(max(y_coords))
                left = int(min(x_coords)); right = int(max(x_coords))
                
                # Skip small detections
                face_width = right - left
                face_height = bottom - top
                
                # Add buffer
                buffer_x = int(face_width * 0.15)
                buffer_y = int(face_height * 0.15)
                
                top = max(0, top - buffer_y); bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x); right = min(w, right + buffer_x)

                face_crop = frame[top:bottom, left:right]
                sharpness = get_sharpness_score(face_crop)
                
                # Generate pose-normalized embedding
                face_encoding = get_pose_normalized_embedding(face_landmarks)
                
                # Recognize
                face_id = None
                if known_face_encodings:
                    face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, face_trackers)
                
                # New face
                if face_id is None:
                    face_id = next_face_id
                    face_trackers[face_id] = {'samples': [], 'complete': False}
                    known_face_encodings.append(face_encoding)
                    known_face_ids.append(face_id)
                    next_face_id += 1
                    print(f"NEW FACE: ID #{face_id}")

                if face_id not in currently_tracked_faces:
                    currently_tracked_faces.add(face_id)

                else:
                    # Already processed this face in current frame
                    continue

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
            sharpness = data['sharpness']
            
            is_target = TARGET_FACE_ID is None or face_id == TARGET_FACE_ID
            
            if is_target and not tracker['complete']:
                
                # 1. Sample collection is in progress
                samples_collected = len(tracker['samples'])
                
                if samples_collected < MIN_SAMPLES_FOR_AVERAGE:
                    required_sharpness = get_sharpness_threshold(face_id)

                    # Check sharpness threshold before adding sample
                    if sharpness >= required_sharpness:
                        tracker['samples'].append({
                            'vector': data['encoding'],
                            'crop': data['crop'],
                            'sharpness': sharpness
                        })
                        color = (0, 255, 0) # Green: Collecting
                        status = f"COLLECTING ({samples_collected + 1}/{MIN_SAMPLES_FOR_AVERAGE})"
                    else:
                        color = (255, 255, 0) # Yellow: Too blurry
                        status = f"BLURRY ({data['sharpness']:.1f}/{required_sharpness:.1f})"
                        
                # 2. Collection is complete, save data once
                else:
                    if save_data_locally(face_id, tracker['samples']):
                        tracker['complete'] = True
                        color = (0, 0, 255) # Red: Complete (Success)
                        status = "SAVE SUCCESS"
                        
                        # Update with final vector
                        idx = known_face_ids.index(face_id)
                        vectors = np.array([s['vector'] for s in tracker['samples']])
                        final_vector = np.mean(np.array([s['vector'] for s in tracker['samples']]), axis=0)
                        
                        known_face_encodings[idx] = final_vector
                        print(f"Finalized Face ID #{face_id}")
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

            frame.flags.writeable = True

            # --- Drawing on Frame ---
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"Face #{face_id}", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{status}", (left, top - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # --- DISPLAY STATUS ---        
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