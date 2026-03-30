import cv2
import numpy as np
import warnings
import mediapipe as mp
import math
import socket
import struct
import argparse
from typing import List, Optional
import time
import traceback
import threading
import queue
import ssl

warnings.filterwarnings("ignore")

# Packets 
from FacePacket import FacePacket
from IDPacket import IDPacket

# Detection Tracker
from face_tracker import SimpleFaceTracker

#Client Config

# Network
SERVER_HOST = '76.28.113.73' #'127.0.0.1'   
#SERVER_HOST = '10.0.0.172' #'127.0.0.1'   #Brady's gross yucky local IP (cuz I'm tired of switching it back every time and uncommenting is marginally easier)      
SERVER_PORT =  33060 #5000
TIMEOUT = 30.0

# Camera
CAMERA_INDEX = 0  #0 for webcam, 6 for virtual cam (OBS), 7 for glasses (usually)

# Face Collection Config (Used for Capture Mode)
BEST_SAMPLES_TO_AVERAGE = 10 # Send 10 crops for full enrollment packet.

# Models
mp_face_mesh = mp.solutions.face_mesh

# Pose/Quality Thresholds
POSE_QUALITY_THRESHOLD = 0.89
SHARPNESS_THRESHOLD = 50.0

# UI info dictionary - # Example: 1: {"fullname": "Alice Smith", "age": 30}
ID_INFO = {} # maybe move this to track object eventually

# Utility functions 

def get_pose_quality(landmarks) -> float:
    """Robust score (0.0 to 1.0) checking Roll, Yaw, and Pitch."""
    lm = landmarks.landmark
    l_eye = np.array([lm[33].x, lm[33].y]); r_eye = np.array([lm[263].x, lm[263].y])
    nose = np.array([lm[1].x, lm[1].y]); lip = np.array([lm[13].x, lm[13].y])
    
    dY = r_eye[1] - l_eye[1]; dX = r_eye[0] - l_eye[0]
    angle = math.degrees(math.atan2(dY, dX)); roll_penalty = (abs(angle) / 60.0) * 1.5 
    eye_center_x = (l_eye[0] + r_eye[0]) / 2
    eye_width = np.linalg.norm(r_eye - l_eye)
    yaw_deviation = abs(nose[0] - eye_center_x) / eye_width; yaw_penalty = yaw_deviation * 1.8 
    eye_line_y = (l_eye[1] + r_eye[1]) / 2
    nose_to_eye = abs(nose[1] - eye_line_y); nose_to_lip = abs(lip[1] - nose[1])
    if nose_to_lip == 0: nose_to_lip = 0.001
    ratio = nose_to_eye / nose_to_lip
    pitch_penalty = 0
    if ratio < 0.4: pitch_penalty = (0.4 - ratio) 
    elif ratio > 2.5: pitch_penalty = (ratio - 2.5) 
    
    total_penalty = roll_penalty + yaw_penalty + pitch_penalty
    score = max(0, 1.0 - total_penalty)
    return score

def get_image_sharpness(image: np.ndarray) -> float:
    """Returns the variance of the Laplacian (sharpness score)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

def conservative_lighting_normalization(face_crop: np.ndarray) -> np.ndarray:
    """Conservative lighting normalization that preserves facial features."""
    if face_crop is None or face_crop.size == 0: return face_crop
    
    try:
        lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        mean_brightness = np.mean(l_channel); std_brightness = np.std(l_channel)
        shadow_area = np.percentile(face_crop, 10) # Checking the shadows passed by the glasses 
        
        if mean_brightness > 200 and std_brightness < 40: #this is for too bright 
            gamma = 1.5         #; inv_gamma = 1.0 / gamma  |darken the overexposured image
            table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8") #inv_gamma changes to gamma
            return cv2.LUT(face_crop, table)
        elif mean_brightness < 60 or shadow_area < 35: # originally (40) checking for shadows casted by the glasses to make sure that they arent't too much 
            alpha = 1.3; beta = 45 # originally 1.2, 30 (hopefully 45 will lift the shadows)
            return cv2.convertScaleAbs(face_crop, alpha=alpha, beta=beta)
        else:
            return face_crop
    except Exception:
        return face_crop

def get_face_crop(frame: np.ndarray, face_landmarks):
    """Extracts and crops the face from the frame based on landmarks and padding."""
    h, w = frame.shape[:2]
    x_coords = [lm.x * w for lm in face_landmarks.landmark]
    y_coords = [lm.y * h for lm in face_landmarks.landmark]

    left, right = int(min(x_coords)), int(max(x_coords))
    top, bottom = int(min(y_coords)), int(max(y_coords))
    
    pad = 20
    left = max(0, left-pad); right = min(w, right+pad)
    top = max(0, top-pad); bottom = min(h, bottom+pad)

    if right - left < 60 or bottom - top < 60: return None, None
    
    return frame[top:bottom, left:right], [left, top, right, bottom]

class FaceCaptureClient:
    """
    Client application (running on glasses) to detect faces via MediaPipe,
    check quality, crop, and send them to the server for recognition/capture.
    """
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.sock = None
        
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open camera {CAMERA_INDEX}")
        
        self.seq_num = 0 #initialize at 0, increment after receiving response
        self.recent_face_ids = [None] * 5 # Last 5 recognized face IDs for context
        
        self.tracker = SimpleFaceTracker(iou_threshold=0.25, max_frames_missed=5, max_age_seconds = 30)
        
        #Threading stuff 
        self.request_queue = queue.Queue() #Stores the packets that are waiting to be sent to the server

        #Create/start the background thread (should close when main program closes)
        self.network_thread = threading.Thread(target=self._network_worker, daemon=True)
        self.network_thread.start()

        self._connect_to_server()
        
    # Networking functions 

    def _network_worker(self):
        """Background thread that handles sending packets and receiving responses."""
        while True:
            #Wait for a packet to be added to the queue
            task = self.request_queue.get()
            if task is None:
                print("[INFO] Network worker thread exiting...")
                break #Exit the thread if there is no task
            track_id, packet = task

            response = self._send_packet_and_receive_id(packet)
            current_time = time.time()

            #Update tracker with server's repsonse
            if track_id in self.tracker.get_active_tracks():
                track = self.tracker.get_active_tracks()[track_id]
                track.last_recognition_time = current_time
                if response:
                    if response.success:
                        track.server_id = response.face_id
                        track.pending_seq_num = None
                        track.recognition_cooldown = 0.0
                        track.failed_attempts = 0
                        
                        # Store and truncate recent IDs for context in future packets
                        self.recent_face_ids.insert(0, response.face_id)
                        self.recent_face_ids = self.recent_face_ids[:5]
                        
                        if ID_INFO.get(response.face_id) is None: # Only store info if we don't already have it for this ID
                            ID_INFO[response.face_id] = {"fullname": response.fullname, "age": response.age} # Store info for UI display
                    else:
                        track.failed_attempts += 1
                        cooldown = min(1.5 ** track.failed_attempts, 6)
                        track.server_id = None
                        track.pending_seq_num = None
                        track.recognition_cooldown = current_time + cooldown

                        if track.buffer_full:
                            track.buffer_full = False
                            track.crop_buffer.clear()

                        else:
                            track.recognition_cooldown = current_time + 1.0
                            track.pending_seq_num = None
                    self.request_queue.task_done()


    def _connect_to_server(self):
        """Establish or re-establish connection to server"""
        try:
            if self.sock:
                try:
                    self.sock.close()
                except:
                    pass
            
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(TIMEOUT)
            self.sock.connect((self.host, self.port))
            print(f"[INFO] Connected to server at {self.host}:{self.port}")
            
            # Wrap the socket with SSL
            # context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            # context.load_verify_locations('server.crt')  # Load server's certificate for verification
            # context.check_hostname = False  # Disable hostname checking
            # self.sock = context.wrap_socket(self.sock, server_hostname=self.host)
            print(f"[INFO] SSL handshake completed with server at {self.host}:{self.port}")
            
        except Exception as e:
            print(f"[ERROR] Failed to connect to server: {e}")
            self.sock = None
            return False
        return True

    def _recv_exactly(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket, handling fragmented reads."""
        data = b''
        while len(data) < n:
            try:
                chunk = self.sock.recv(n - len(data))
                if not chunk:
                    return None # Connection closed
                data += chunk
            except socket.timeout:
                raise socket.timeout("Timed out waiting for full packet.")
            except Exception:
                return None
        return data

    def _send_packet_and_receive_id(self, packet: FacePacket) -> Optional[IDPacket]:
        """Connects to server, sends packet, and receives IDPacket reliably."""
        if not self.sock:
            if not self._connect_to_server():
                return None
        
        try:
            serialized_packet = packet.serialize()
            self.sock.sendall(serialized_packet)
                
            # Receive IDPacket length (4 bytes)
            response_len_data = self._recv_exactly(4)
            
            # Handle connection loss and attempt to reconnect
            if not response_len_data:
                # Connection broken, try to reconnect
                print("[WARN] Connection lost, reconnecting...")
                if not self._connect_to_server():
                    return None
                # Retry sending the packet
                self.sock.sendall(serialized_packet)
                response_len_data = self._recv_exactly(4)
                if not response_len_data:
                    return None
                
            response_len = struct.unpack('>I', response_len_data)[0]
            
            # Receive the IDPacket payload
            response_payload = self._recv_exactly(response_len)
            if not response_payload: return None
            
            return IDPacket.deserialize(response_payload)
                
        except socket.timeout as e:
            print(f"[ERROR] Socket timeout: {e}")
            # Attempt to reconnect
            self._connect_to_server()
            return None
        except socket.error as e:
            print(f"[ERROR] Socket error: {e}")
            # Attempt to reconnect
            self._connect_to_server()
            return None
        except Exception as e:
            print(f"[ERROR] Network communication error: {e}")
            return None

    # Main capture loop 

    def run(self):
        """Main loop for face detection, quality check, and server communication."""
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        ) as face_mesh:

            print(f"Client running. Sending face data to {self.host}:{self.port}...")
            print("Press 'q' or 'Esc' to quit the application.")

            # Default status
            status = "Searching..."
            color = (255, 0, 0) # Blue (Searching)
            
            while self.cap.isOpened():
                # Get frame
                success, frame = self.cap.read()
                if not success: continue
                
                # Initialize current frame data lists
                current_frame_boxes = []
                quality_list = []
                face_crops_for_boxes = []

                # Process frame for face landmarks
                frame.flags.writeable = False
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                frame.flags.writeable = True
                
                # Handle detected faces
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Pre-processing and quality checks 
                        raw_face_crop, border = get_face_crop(frame, face_landmarks)
                        track_box = (border[0], border[1], border[2], border[3])  # (x1, y1, x2, y2)

                        if raw_face_crop is None: continue
                            
                        processed_face_crop = conservative_lighting_normalization(raw_face_crop)
                            
                        sharpness = get_image_sharpness(processed_face_crop)
                        pose_score = get_pose_quality(face_landmarks)
                            
                        is_sharp_enough = sharpness >= SHARPNESS_THRESHOLD                          
                            
                        is_pose_ok = pose_score >= POSE_QUALITY_THRESHOLD
                        
                        # Always append the most recent box for tracking
                        current_frame_boxes.append(track_box)
                        face_crops_for_boxes.append(processed_face_crop)
                        
                        # Check quality and attach to list
                        if is_sharp_enough and is_pose_ok:
                            quality_list.append(True) # Mark as good quality
                        else:
                            quality_list.append(False) # Mark as poor quality
                    
                    # Update tracker: get persistent track_ids for this frame's boxes
                    tracker_results = self.tracker.update(current_frame_boxes)
                    
                    # Get active tracks
                    active_tracks = self.tracker.get_active_tracks()
                    
                    # Process each tracked face
                    for track_id, current_box in tracker_results.items():
                        # Find which crop index corresponds to this box
                        try:
                            box_index = current_frame_boxes.index(current_box)
                            current_crop = face_crops_for_boxes[box_index]
                            current_quality = quality_list[box_index]
                        except ValueError:
                            continue # Box not found, skip
                        
                        # Get track object
                        track = active_tracks[track_id]
                        
                        current_time = time.time()
                        
                        # If already recognized, just display
                        if track.server_id is not None:
                            display_id = track.server_id
                            
                            # Get info related to this ID from the database
                            db_info = ID_INFO.get(display_id)
                            
                            if db_info is None or db_info.get("age") == 0 or db_info.get("fullname") == "": # Handle case where ID exists but no info found from DB
                                db_info = {"fullname": "Unknown", "age": "Unknown"}
                            
                            #status = f"ID: #{display_id} | {db_info.get('fullname')} | {db_info.get('age')} yrs"
                            nameLine = f"Name: {db_info.get('fullname')}"
                            ageLine = f"Age: {db_info.get('age')}"
                            idLine = f"ID: #{display_id}"

                            ##UI Crapola
                            color = (0, 255, 0)  # Green
                            x1, y1, x2, y2 = current_box
                            cv2.rectangle(frame, (current_box[0], current_box[1]), 
                                        (current_box[2], current_box[3]), color, 2)
                            cv2.putText(frame, nameLine, (x1, y1 - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            cv2.putText(frame, ageLine, (x1, y1 - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            cv2.putText(frame, idLine, (x1, y1 - 1),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                            continue  # Skip server query for this face

                        # Check if we're in cooldown after a failed attempt
                        elif current_time < track.recognition_cooldown:
                            cooldown_left = track.recognition_cooldown - current_time
                            status = f"Retry in {cooldown_left:.1f}s"
                            color = (255, 165, 0)  # Orange
                           
                        
                        # Check if there's a pending request
                        elif track.pending_seq_num is not None:
                            status = "Recognizing..."
                            color = (0, 255, 255)  # Yellow
                        
                        # Check quality before sending
                        elif not current_quality:
                            status = "Poor Quality"
                            color = (0, 0, 255)  # Red
                            
                        else:
                            #First attempt or no recent IDs (ID CASE with 10 crops)
                            if (track.failed_attempts > 0 or self.recent_face_ids[0] is None): 
                                if not track.buffer_full:
                                    track.crop_buffer.append(current_crop)
                                    if len(track.crop_buffer) >= BEST_SAMPLES_TO_AVERAGE:
                                        track.buffer_full = True
                                    else:
                                        #Still gathering crops, set status and skip sending
                                        status = f"Gathering {len(track.crop_buffer)}/{BEST_SAMPLES_TO_AVERAGE}"
                                        color = (255, 255, 0)  # Cyan
                                        
                                #If the buffer just filled up, prep the packet and send
                                if track.buffer_full:
                                    packet = FacePacket(self.seq_num, track.crop_buffer, [None]*5)
                                    
                                    track.pending_seq_num = self.seq_num  
                                    self.request_queue.put((track_id, packet))
                                    self.seq_num += 1
                                    
                                    status = "Recognizing..."
                                    color = (0, 255, 255)  # Yellow
                                    
                            #First attempt failed or recent IDs available (RE-ID CASE with 1 crop + recent IDs)
                            else:
                                packet = FacePacket(self.seq_num, [current_crop], self.recent_face_ids)
                                
                                track.pending_seq_num = self.seq_num  
                                self.request_queue.put((track_id, packet))
                                self.seq_num += 1
                                
                                status = "Recognizing..."
                                color = (0, 255, 255)  # Yellow

                        # Draw the box and status
                        cv2.rectangle(frame, (current_box[0], current_box[1]), 
                                    (current_box[2], current_box[3]), color, 2)
                        cv2.putText(frame, status, (current_box[0], current_box[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                # Drawing the frame                
                cv2.imshow('Face Capture Client (Glasses)', frame)
                
                # Handle keyboard inputs (only for quitting)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: 
                    break
        
        # Cleanup
        if self.sock: self.sock.close()
        self.cap.release()
        cv2.destroyAllWindows()

# Main
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Face Capture Client (Glasses)')
    parser.add_argument('--host', default=SERVER_HOST, help='Server Host IP')
    parser.add_argument('--port', type=int, default=SERVER_PORT, help='Server Port')
    
    args = parser.parse_args()
    
    try:
        client = FaceCaptureClient(host=args.host, port=args.port)
        client.run()
    except IOError as e:
        print(f"Failed to start client: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(traceback.format_exc())