import os
import cv2
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace 

# --- CONFIGURATION ---
# Using Facenet for a balance of speed and accuracy
DEEPFACE_MODEL = 'Facenet'

# This is where your collected data will be stored locally.
DATABASE_FOLDER = "rm_db"
VECTORS_FILE = os.path.join(DATABASE_FOLDER, "vectors.json")
FRAMES_FOLDER = os.path.join(DATABASE_FOLDER, "best_frames")

# Ensure necessary folders exist
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Recognition parameters
RECOGNITION_THRESHOLD = 0.85 # Cosine similarity threshold for recognition (0 to 1 scale)
TARGET_FACE_ID = None 

# Sampling parameters for data collection
MIN_SAMPLES_FOR_AVERAGE = 50    
NUM_BEST_FRAMES_TO_SEND = 25

# Weighting parameters
PROGRESS_BONUS = 0.08 # Bonus added to similarity based on collection progress
MIN_SHARPNESS_KNOWN = 10.0  # Lower sharpness allowed for known faces
MIN_SHARPNESS_UNKNOWN = 37.5 # Higher sharpness required for new (unknown) faces
POSE_QUALITY_THRESHOLD = 0.55  # Minimum pose quality score to consider a face

# --- DATA STRUCTURES ---
next_face_id = 1
known_face_encodings = [] 
known_face_ids = [] 
face_trackers = {} 
currently_tracked_faces = set() 

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# --- LOAD EXISTING VECTORS (Persistence) ---
try:
    if os.path.exists(VECTORS_FILE):
        with open(VECTORS_FILE, 'r') as f:
            loaded_vectors = json.load(f)
            for face_id_str, vector_list in loaded_vectors.items():
                face_id = int(face_id_str)
                known_face_ids.append(face_id)
                known_face_encodings.append(np.array(vector_list))
                next_face_id = max(next_face_id, face_id + 1)
                
                face_trackers[face_id] = {
                    'samples': [],
                    'complete': True 
                }
            print(f"Loaded {len(known_face_ids)} existing faces. Next ID will be {next_face_id}.")
            
except FileNotFoundError:
    print("No existing vectors file found. Starting fresh.")
except Exception as e:
    print(f"Error loading persistence files: {e}. Starting fresh.")


# --- HELPER FUNCTIONS ---

# Function to preprocess the frame for better face detection
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

def get_pose_quality_score(landmarks):
    """Score face pose quality (0-1 scale)"""
    points_3d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
    
    # Check if face is frontal (similar to v1 logic)
    left_eye = points_3d[33]
    right_eye = points_3d[263]
    nose_tip = points_3d[1]
    
    # Calculate eye alignment
    eye_vector = right_eye - left_eye
    eye_alignment = abs(eye_vector[1]) / np.linalg.norm(eye_vector)
    
    # Lower score = more frontal (better)
    pose_score = 1.0 - min(eye_alignment * 3, 1.0)
    
    return pose_score

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

def recognize_face(face_encoding, known_encodings, known_ids, face_trackers, threshold=RECOGNITION_THRESHOLD):
    """Performs recognition using weighted cosine similarity."""
    if face_encoding is None or not known_encodings:
        return None
    
    best_similarity = -1
    best_match_id = None
    
    for face_id, known_encoding in zip(known_ids, known_encodings):
        if face_encoding.shape != known_encoding.shape:
            continue
            
        base_similarity = cosine_similarity(face_encoding, known_encoding)

        # Apply weighting based on face status
        if face_id in face_trackers:
            if face_trackers[face_id]['complete']:
                weighted_similarity = base_similarity + PROGRESS_BONUS  # Small bonus for known faces
            else:
                samples_collected = len(face_trackers[face_id]['samples'])
                # Reduced bonus to prevent merging two different people
                progress_bonus = min((samples_collected / MIN_SAMPLES_FOR_AVERAGE) * PROGRESS_BONUS, PROGRESS_BONUS) 
                weighted_similarity = base_similarity + progress_bonus
        else:
            weighted_similarity = base_similarity
        
        if weighted_similarity > best_similarity and weighted_similarity > threshold:
            best_similarity = weighted_similarity
            best_match_id = face_id
    
    if best_match_id:
        print(f"Matched Face #{best_match_id} (similarity: {best_similarity:.3f})")
    
    return best_match_id

def get_sharpness_threshold(face_id):
    """Return appropriate sharpness threshold based on face status"""
    if face_id in face_trackers and face_trackers[face_id]['complete']:
        return MIN_SHARPNESS_KNOWN
    else:
        return MIN_SHARPNESS_UNKNOWN

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

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Face Mesh model
with mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.01) as face_mesh:

    print("Starting face capture. Press 'Esc' to exit.")

    while cap.isOpened():
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
                    continue
                
                # Check if pose quality is acceptable
                pose_quality = get_pose_quality_score(face_landmarks)
                if pose_quality < POSE_QUALITY_THRESHOLD:
                    # frame.flags.writeable = True #debugging
                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    # cv2.putText(frame, f"BAD POSE: {pose_quality:.2f}", (left, top - 10), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # frame.flags.writeable = False
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
                    'box': (top, right, bottom, left),
                    'tracker': None, # Placeholder
                    'crop': face_crop,
                    'sharpness': sharpness,
                    'encoding': face_encoding,
                    'pose_quality': pose_quality
                }
                
                # Recognize against known faces
                face_id = None
                if known_face_encodings:
                    face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, face_trackers)
                
                # New face or not recognized
                if face_id is None:
                    # --- NEW SHARPNESS CHECK FOR UNKNOWN FACES ---
                    required_sharpness = MIN_SHARPNESS_UNKNOWN
                    if sharpness < required_sharpness:
                        # Temporarily use a placeholder ID (-1) to draw a box but skip tracking
                        face_id = -1 
                        data['color'] = (0, 100, 255) # Dark Orange: Too blurry to track/start new
                        data['status'] = f"BLURRY START ({sharpness:.1f}/{required_sharpness:.1f})"
                    # ---------------------------------------------
                    
                    else: # Sharpness is acceptable, proceed with new ID assignment
                        face_id = next_face_id
                        
                        if face_id not in face_trackers or face_trackers[face_id]['complete']:
                            face_trackers[face_id] = {'samples': [], 'complete': False}
                            known_face_encodings.append(face_encoding)
                            known_face_ids.append(face_id)
                            next_face_id += 1
                            print(f"NEW FACE: ID #{face_id}")
                            
                            # Immediately add the first sample for a high-quality start
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
            face_id = data['id']
            top, right, bottom, left = data['box']
            sharpness = data['sharpness']
            
            # Convert coordinates back to scale of original frame
            scale_factor = 1.5
            top_orig = int(top / scale_factor)
            bottom_orig = int(bottom / scale_factor)
            left_orig = int(left / scale_factor)
            right_orig = int(right / scale_factor)
            
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
                            # Only add sample if it's sharp enough and hasn't been added yet (first sample handled above)
                            if samples_collected == 0 or samples_collected > 0: 
                                tracker['samples'].append({
                                    'vector': data['encoding'],
                                    'crop': data['crop'],
                                    'sharpness': sharpness
                                })
                            color = (0, 255, 0) # Green: Collecting
                            # samples_collected is accurate after the append
                            status = f"COLLECTING ({len(tracker['samples'])}/{MIN_SAMPLES_FOR_AVERAGE})" 
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

            cv2.rectangle(frame, (left_orig, top_orig), (right_orig, bottom_orig), color, 2)
            cv2.putText(frame, display_id, (left_orig, top_orig - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{status}", (left_orig, top_orig - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


        # --- DISPLAY STATUS --- 		
        info_text = f"Total Unique People: {next_face_id - 1}. Model: {DEEPFACE_MODEL}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        status_text = f"Threshold: {RECOGNITION_THRESHOLD} | Samples: {MIN_SAMPLES_FOR_AVERAGE}"
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('DeepFace + MediaPipe Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
 
cv2.destroyAllWindows()
cap.release()