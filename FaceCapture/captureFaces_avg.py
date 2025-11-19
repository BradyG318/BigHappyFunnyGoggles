import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace
import DB_Link

# --- CONFIGURATION ---
# Initialize database connection
DB_Link.db_link.initialize()
DB_Link.db_link.clear_db()

# Initialize ai models
DEEPFACE_MODEL = 'Facenet512'
mp_face_mesh = mp.solutions.face_mesh

# PARAMETERS
CAMERA_INDEX = 0

RECOGNITION_THRESHOLD = 0.95 # Cosine similarity threshold for recognition (0 - 1)

MIN_SAMPLES_FOR_AVERAGE = 30 # Minimum samples required to compute average vector
NUM_BEST_FRAMES_TO_SEND = 30 # Number of best quality frames to use for average vector

# DATA STRUCTURES
next_face_id = 1
known_face_encodings = [] 
known_face_ids = []
face_trackers = {}

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

# --- FUNCTIONS ---

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

def recognize_face(face_encoding, known_face_encodings, known_ids, face_trackers):
    """Performs recognition using weighted cosine similarity with temporal consistency."""
    if face_encoding is None or not known_face_encodings:
        return None
    
    # Initialize best similarity and match id
    best_similarity = -1
    best_match_id = None

    # Loop through known faces
    for face_id, known_encoding in zip(known_ids, known_face_encodings):
        base_similarity = cosine_similarity(face_encoding, known_encoding)
        
        # Check if this is the best match so far
        if base_similarity > best_similarity and base_similarity >= RECOGNITION_THRESHOLD:
            best_similarity = base_similarity
            best_match_id = face_id

    return best_match_id

def save_data_to_database(face_id, samples):
    """
    Finalizes the data collection, calculates the final average vector,
    and saves the vector to PostgreSQL database.
    """
    print(f"\n--- DATABASE SAVE START: Face ID #{face_id} ---")
    
    # Calculate the final, averaged face vector
    vectors = np.array([s['vector'] for s in samples])
    final_vector = np.mean(vectors, axis=0)
    
    # Save the final vector to database synchronously
    success = DB_Link.db_link.save_face_vector(face_id, final_vector.tolist())
    
    if not success:
        print(f"!!! ERROR saving vector to database for face #{face_id}")
        return False
    
    print(f"Vector saved to database for Face ID #{face_id}")
    print(f"--- DATABASE SAVE COMPLETE: Face ID #{face_id} ---\n")
    return True

# --- MAIN ---
# Start video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

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
                # Get bounding box (using MediaPipe landmarks)
                h, w = detection_frame.shape[:2]
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                top = int(min(y_coords)); bottom = int(max(y_coords))
                left = int(min(x_coords)); right = int(max(x_coords))

                # Skip small detections - uncomment later
                face_width = right - left
                face_height = bottom - top
                # if face_width < 80 or face_height < 80:
                #     print(f"Skipped small face detection: {face_width}x{face_height}px")
                #     continue

                # Add buffer for cropping and update coords
                buffer_x = int(face_width * 0.15)
                buffer_y = int(face_height * 0.15)
                
                top = max(0, top - buffer_y); bottom = min(h, bottom + buffer_y)
                left = max(0, left - buffer_x); right = min(w, right + buffer_x)

                # Crop face region
                face_crop = recognition_frame[top:bottom, left:right]

                # Generate embedding using DeepFace
                face_encoding = get_deepface_embedding(face_crop)

                if face_encoding is None:
                    continue
                
                # Initialize the data dictionary for this face detection
                data = {
                    'id': None, # Placeholder
                    'box': (top, right, bottom, left),
                    'tracker': None, # Placeholder
                    'crop': face_crop,
                    'encoding': face_encoding,
                }

                # Initialize face_id
                face_id = None
                
                if known_face_encodings:
                    face_id = recognize_face(face_encoding, known_face_encodings, known_face_ids, face_trackers)

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
                        'crop': face_crop,
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
        
        # --- SAMPLING ---
        # Loop through all current data
        for data in current_frame_data:
            # Unpack data for a detection
            face_id = data['id']
            top, right, bottom, left = data['box']

            tracker = data['tracker']

            if not tracker['complete']:
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
                    if save_data_to_database(face_id, tracker['samples']):
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
                color = (0, 0, 255) # Red: Complete (Recognized)
                status = "RECOGNIZED"
        
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
        
        status_text = f"Threshold: {RECOGNITION_THRESHOLD} | Samples: {MIN_SAMPLES_FOR_AVERAGE}"
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('DeepFace + MediaPipe Face Recognition Tracker', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Close resources on exit
DB_Link.db_link.close()
cap.release()
cv2.destroyAllWindows()