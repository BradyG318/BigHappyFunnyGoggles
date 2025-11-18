import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace
import DB_Link

# --- CONFIGURATION ---

# Uncomment for local storage option (instead of database)
# NOTE: Will need to reimplement old local saving method for this to work
# DATABASE_FOLDER = "rm_db"
# VECTORS_FILE = os.path.join(DATABASE_FOLDER, "vectors.json")
# FRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "best_frames")
# os.makedirs(FRAMES_FOLDER, exist_ok=True)
# os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Initialize database connection
DB_Link.db_link.initialize()

# Uncomment to clear database (for testing purposes)
DB_Link.db_link.clear_db()

# Using Facenet for a balance of speed and accuracy
DEEPFACE_MODEL = 'Facenet512'

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Camera index
CAMERA_INDEX = 1

# Recognition parameters
RECOGNITION_THRESHOLD = 0.91 # Cosine similarity threshold for recognition (0 to 1 scale)
TARGET_FACE_ID = None 

# Sampling parameters for data collection
MIN_SAMPLES_FOR_AVERAGE = 75
NUM_BEST_FRAMES_TO_SEND = 30

# Misc weighting parameters
KNOWN_BONUS = 0.03  # Bonus added to similarity for known faces
PROGRESS_BONUS = 0.02 # Bonus added to similarity based on collection progress

MIN_SHARPNESS_KNOWN = 15.0  # Lower sharpness allowed for known faces
MIN_SHARPNESS_UNKNOWN = 35.0 # Higher sharpness required for new (unknown) faces

POSE_QUALITY_THRESHOLD = 0.30  # Minimum pose quality score to consider a face

# --- DATA STRUCTURES ---
next_face_id = 1
known_face_encodings = [] 
known_face_ids = [] 

face_trackers = {} 
currently_tracked_faces = set()

previous_face_positions = {} # face_id -> last known position

# --- LOAD EXISTING VECTORS ---
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

# --- HELPER FUNCTIONS ---

def dual_preprocess_frame(image):
    """ #1 Preprocess the frame to enhance face detection.
        #2 Neutral frame for recognition.
    """

    #1

    # Copy original to avoid modifying it directly
    detection_image = image.copy()

    # Reduce compression artifacts
    detection_image = cv2.GaussianBlur(detection_image, (3, 3), 0)

    # Enhance contrast
    lab = cv2.cvtColor(detection_image, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:,:,0])  # Reduced from 3.0
    detection_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Scale image up for better detection of smaller faces
    scale_factor = 1.5  # Increase this if needed (1.5 = 150% size)
    height, width = detection_image.shape[:2]
    detection_image = cv2.resize(detection_image, (int(width * scale_factor), int(height * scale_factor)))

    #2

    # Copy original to avoid modifying it directly
    recognition_image = image.copy()

    # Slight blur to reduce noise for recognition
    recognition_image = cv2.GaussianBlur(recognition_image, (3, 3), 0)

    return detection_image, recognition_image

def normalize_lighting(face_crop):
    """
    Apply lighting normalization to make face recognition more consistent
    across different lighting conditions
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    
    try:
        # Convert to LAB color space
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        
        # Normalize the L channel (lightness)
        l_channel = lab[:,:,0]
        
        # Apply simple histogram normalization (better than CLAHE for recognition)
        l_normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        
        # Merge back
        lab[:,:,0] = l_normalized
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
        
    except Exception as e:
        # If normalization fails, return original
        return face_crop

def adaptive_lighting_compensation(face_crop):
    """
    Smart lighting compensation that only activates when needed
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    
    # Calculate image statistics to detect lighting issues
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Only apply compensation if lighting is problematic
    if mean_brightness < 50 or mean_brightness > 200 or contrast < 25:
        return normalize_lighting(face_crop)
    else:
        # Good lighting, return as-is
        return face_crop

def get_sharpness_score(image):
    """Calculates the image sharpness using the variance of the Laplacian."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_sharpness_threshold(face_id):
    """Return appropriate sharpness threshold based on face status"""
    if face_id in face_trackers and face_trackers[face_id]['complete']:
        return MIN_SHARPNESS_KNOWN
    else:
        return MIN_SHARPNESS_UNKNOWN

def get_pose_quality_score(landmarks):
    """Score face pose quality (0-1 scale)"""
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # Key landmarks
    left_eye = points_3d[33]
    right_eye = points_3d[263]
    nose_tip = points_3d[1]
    chin = points_3d[152]
    
    # 1. Roll - eye level alignment
    eye_vector = right_eye - left_eye
    roll_penalty = abs(eye_vector[1]) / np.linalg.norm(eye_vector)
    
    # 2. Pitch - vertical face alignment  
    face_vertical = chin - nose_tip
    pitch_penalty = abs(face_vertical[1]) / np.linalg.norm(face_vertical)
    
    # 3. Yaw - face symmetry
    left_dist = np.linalg.norm(left_eye - nose_tip)
    right_dist = np.linalg.norm(right_eye - nose_tip)
    yaw_penalty = abs(left_dist - right_dist) / max(left_dist, right_dist)
    
    # Combine penalties (lower is better)
    total_penalty = roll_penalty * 0.5 + pitch_penalty * 0.3 + yaw_penalty * 0.2
    
    # Convert to quality score (higher is better)
    pose_score = max(0, 1.0 - total_penalty * 2)
    
    return pose_score

def get_pose_aware_threshold(pose_quality, base_threshold=RECOGNITION_THRESHOLD):
    """Adjust threshold based on pose quality"""
    if pose_quality < 0.4:
        return base_threshold + 0.05  # Much harder for bad poses
    elif pose_quality < 0.6:
        return base_threshold + 0.02  # Slightly harder
    else:
        return base_threshold  # Normal for good poses

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

def save_data_to_database(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    and saves the vector to PostgreSQL database.
    """
    print(f"\n--- DATABASE SAVE START: Face ID #{face_id} ---")
    
    # 1. Calculate the final, averaged face vector
    vectors = np.array([s['vector'] for s in samples])
    final_vector = np.mean(vectors, axis=0)
    
    # 2. Save the final vector to database - synchronous call!
    success = DB_Link.db_link.save_face_vector(face_id, final_vector.tolist())
    
    if not success:
        print(f"!!! ERROR saving vector to database for face #{face_id}")
        return False
    
    print(f"Vector saved to database for Face ID #{face_id}")
    print(f"--- DATABASE SAVE COMPLETE: Face ID #{face_id} ---\n")
    return True

def calculate_center_distance(box1, box2):
    """Calculate distance between centers of two bounding boxes"""
    top1, right1, bottom1, left1 = box1
    top2, right2, bottom2, left2 = box2
    
    center1_x = (left1 + right1) / 2
    center1_y = (top1 + bottom1) / 2
    center2_x = (left2 + right2) / 2
    center2_y = (top2 + bottom2) / 2
    
    return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def get_stable_face_id(current_face, current_frame_data):
    """Simple consistency based on position and appearance"""
    current_id = current_face['id']
    if current_id == -1:  # Don't stabilize blurry faces
        return current_id
        
    current_pos = current_face['box']
    current_encoding = current_face['encoding']
    
    # Check if any recent face was in a similar position with similar appearance
    for prev_id, prev_pos in previous_face_positions.items():
        if prev_id == -1:  # Skip blurry faces
            continue
            
        distance = calculate_center_distance(current_pos, prev_pos)
        if distance < 100:  # 100 pixel threshold
            # Check if it's the same person by appearance
            if prev_id in known_face_ids:
                idx = known_face_ids.index(prev_id)
                prev_encoding = known_face_encodings[idx]
                similarity = cosine_similarity(current_encoding, prev_encoding)
                if similarity > 0.85:  # Same person
                    print(f"Stabilized: {current_id} -> {prev_id} (distance: {distance:.1f}, similarity: {similarity:.3f})")
                    return prev_id
    
    # Update position for this ID
    previous_face_positions[current_id] = current_pos
    return current_id

def recognize_face(face_encoding, known_encodings, known_ids, face_trackers, pose_quality, track_history=None):
    """Performs recognition using weighted cosine similarity with temporal consistency."""
    if face_encoding is None or not known_encodings:
        return None
    
    # Initialize best similarity and match id
    best_similarity = -1
    best_match_id = None

    # Get pose-aware threshold
    effective_threshold = get_pose_aware_threshold(pose_quality)

    # Loop through known faces
    for face_id, known_encoding in zip(known_ids, known_encodings):
        base_similarity = cosine_similarity(face_encoding, known_encoding)
        weighted_similarity = base_similarity

        # PROGRESS BONUS
        if face_id in face_trackers and not face_trackers[face_id]['complete']:
            samples_collected = len(face_trackers[face_id]['samples'])
            progress_ratio = samples_collected / MIN_SAMPLES_FOR_AVERAGE
            
            if progress_ratio < 0.5:
                progress_bonus = PROGRESS_BONUS * 0.1 * np.log(1 + 10 * progress_ratio)
            else:
                progress_bonus = PROGRESS_BONUS * progress_ratio

            weighted_similarity += progress_bonus
        
        # KNOWN BONUS
        elif face_id in face_trackers and face_trackers[face_id]['complete']:
            weighted_similarity += KNOWN_BONUS

        if weighted_similarity > best_similarity and weighted_similarity >= effective_threshold:
            best_similarity = weighted_similarity
            best_match_id = face_id
    
    if best_match_id:
        print(f"Matched Face #{best_match_id} (similarity: {best_similarity:.3f})")
    
    return best_match_id

def find_similar_ongoing_collection(face_encoding, current_face_id):
    """Check if there's already a collection for this face"""
    if current_face_id in face_trackers and not face_trackers[current_face_id]['complete']:
        return current_face_id  # This face is already being collected
    
    # Check all ongoing collections
    for face_id, tracker in face_trackers.items():
        if not tracker['complete'] and tracker['samples']:
            # Compare with the first sample of ongoing collection
            first_sample_vector = tracker['samples'][0]['vector']
            similarity = cosine_similarity(face_encoding, first_sample_vector)
            
            if similarity > 0.88:  # Very high threshold - likely same person
                print(f"Found similar ongoing collection: {face_id} (similarity: {similarity:.3f})")
                return face_id
    
    return None

# --- MAIN EXECUTION ---

# Initialize video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

# Initialize Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.01) as face_mesh:

    print("Starting face capture. Press 'Esc' to exit.")

    # Initialize frame counter for tracking passage of time
    frame_count = 0

    while cap.isOpened():
        # Increment frame count
        frame_count += 1

        success, frame = cap.read()
        
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        #frame = cv2.flip(frame, 1)

        # Preprocess frame
        detection_frame, recognition_frame = dual_preprocess_frame(frame)
        frame.flags.writeable = False #improve performance

        # Use detection frame for Mediapipe
        rgb_frame = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        current_frame_data = []
        currently_tracked_faces.clear()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box (using MediaPipe landmarks)
                h, w = detection_frame.shape[:2]
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                top = int(min(y_coords)); bottom = int(max(y_coords))
                left = int(min(x_coords)); right = int(max(x_coords))
                
                # Skip small detections
                face_width = right - left
                face_height = bottom - top
                if face_width < 80 or face_height < 80:
                    print(f"Skipped small face detection: {face_width}x{face_height}px")
                    continue
                
                # Check if pose quality is acceptable
                pose_quality = get_pose_quality_score(face_landmarks)
                if pose_quality < POSE_QUALITY_THRESHOLD:
                    # frame.flags.writeable = True #debugging
                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # cv2.putText(frame, f"BAD POSE: {pose_quality:.2f}", (left, top - 10), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # frame.flags.writeable = False
                    print(f"Skipped face due to poor pose quality: {pose_quality:.2f}")
                    continue

                # Add buffer for cropping
                buffer_x = int(face_width * 0.15)
                buffer_y = int(face_height * 0.15)
                
                top = max(0, top - buffer_y); bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x); right = min(w, right + buffer_x)

                # Convert coordinates back to RECOGNITION frame (original scale)
                scale_factor = 1.5
                top_recog = int(top / scale_factor)
                bottom_recog = int(bottom / scale_factor)
                left_recog = int(left / scale_factor) 
                right_recog = int(right / scale_factor)

                # Extract the face crop for DeepFace from neutral recognition frame
                face_crop = recognition_frame[top_recog:bottom_recog, left_recog:right_recog]
                
                # Calculate true sharpness from original frame
                sharpness = get_sharpness_score(frame[top_recog:bottom_recog, left_recog:right_recog])

                # Apply adaptive lighting normalization
                face_crop_final = adaptive_lighting_compensation(face_crop)

                # Generate embedding using DeepFace
                face_encoding = get_deepface_embedding(face_crop_final) 

                if face_encoding is None:
                    continue

                # Prepare the data dictionary
                data = {
                    'id': None, # Placeholder
                    'box': (top_recog, right_recog, bottom_recog, left_recog),
                    'tracker': None, # Placeholder
                    'crop': face_crop,
                    'sharpness': sharpness,
                    'encoding': face_encoding,
                    'pose_quality': pose_quality
                }
                
                # Recognize against known faces OR locked identity from track
                face_id = None

                if known_face_encodings:
                    # Only do recognition if no locked identity
                    face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, face_trackers, pose_quality)
                
                # New face or not recognized
                if face_id is None:
                    # --- NEW SHARPNESS CHECK FOR UNKNOWN FACES ---
                    required_sharpness = MIN_SHARPNESS_UNKNOWN
                    if sharpness < required_sharpness:
                        face_id = -1 
                        data['color'] = (0, 100, 255)
                        data['status'] = f"BLURRY START ({sharpness:.1f}/{required_sharpness:.1f})"
                    else:
                        # Try to find existing collection first
                        existing_collection_id = find_similar_ongoing_collection(face_encoding, None)
                        
                        if existing_collection_id and existing_collection_id in face_trackers:
                            # Use existing collection
                            face_id = existing_collection_id
                            print(f"Using existing collection {face_id} instead of creating new ID")
                            data['tracker'] = face_trackers[face_id]
                        else:
                            # Create new collection
                            face_id = next_face_id
                            face_trackers[face_id] = {'samples': [], 'complete': False}
                            known_face_encodings.append(face_encoding)
                            known_face_ids.append(face_id)
                            next_face_id += 1
                            print(f"NEW FACE: ID #{face_id}")
                            
                            face_trackers[face_id]['samples'].append({
                                'vector': face_encoding,
                                'crop': face_crop,
                                'sharpness': sharpness
                            })
                            data['tracker'] = face_trackers[face_id]

                if face_id == -1: # Skip blurry faces
                    # Blurry face - set placeholder values
                    data['id'] = -1
                    data['tracker'] = None
                    data['color'] = (0, 100, 255)  # Dark Orange
                    data['status'] = f"BLURRY START ({sharpness:.1f}/{MIN_SHARPNESS_UNKNOWN:.1f})"
                    current_frame_data.append(data)
                    continue
                
                if face_id not in currently_tracked_faces:
                    currently_tracked_faces.add(face_id)

                # If face_id is not -1, assign the tracker
                if face_id != -1:
                    data['id'] = face_id
                    data['tracker'] = face_trackers[face_id]
                    current_frame_data.append(data)
                else:
                    continue # Skip duplicate tracking in the same frame

        # --- DRAWING & SAMPLING LOGIC ---
        
        for data in current_frame_data:
            # Apply simple stabilization
            data['id'] = get_stable_face_id(data, current_frame_data)

            face_id = data['id']
            top, right, bottom, left = data['box']
            sharpness = data['sharpness']
            
            # If the face was too blurry to start, use the temporary drawing info
            if face_id == -1:
                color = data['color']
                status = data['status']
            else:
                tracker = data['tracker']
                is_target = TARGET_FACE_ID is None or face_id == TARGET_FACE_ID
                
                if is_target and not tracker['complete']:
                    # 1. Sample collection is in progress
                    samples_collected = len(tracker['samples'])
                    
                    if samples_collected < MIN_SAMPLES_FOR_AVERAGE:
                        required_sharpness = get_sharpness_threshold(face_id)

                        if sharpness >= required_sharpness:
                            # Add sample if it's sharp enough
                            tracker['samples'].append({
                                'vector': data['encoding'],
                                'crop': data['crop'],
                                'sharpness': sharpness
                            })
                            color = (0, 255, 0) # Green: Collecting
                            status = f"COLLECTING ({len(tracker['samples'])}/{MIN_SAMPLES_FOR_AVERAGE})" 
                        else:
                            color = (255, 255, 0) # Yellow: Too blurry
                            status = f"BLURRY ({sharpness:.1f}/{required_sharpness:.1f})"
                            
                    # 2. Collection is complete, save data once
                    else:
                        if save_data_to_database(face_id, tracker['samples']):
                            tracker['complete'] = True
                            color = (0, 0, 255) # Red: Complete (Success)
                            status = "SAVE SUCCESS"
                            
                            # Update with final vector
                            idx = known_face_ids.index(face_id)
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

            # --- Drawing on Frame ---
            display_id = f"Face #{face_id}" if face_id != -1 else "Unknown"

            frame.flags.writeable = True

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, display_id, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{status}", (left, top - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # --- DISPLAY STATUS --- 		
        info_text = f"Total Unique People: {next_face_id - 1}. Model: {DEEPFACE_MODEL}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_text = f"Threshold: {RECOGNITION_THRESHOLD} | Samples: {MIN_SAMPLES_FOR_AVERAGE}"
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('DeepFace + MediaPipe Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
DB_Link.db_link.close()
cv2.destroyAllWindows()
cap.release()