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

warnings.filterwarnings("ignore")

# Packets 
from FacePacket import FacePacket
from IDPacket import IDPacket

# Detection Tracker
from face_tracker import SimpleFaceTracker

#Client Config

# Network
SERVER_HOST = '76.28.113.73' #'127.0.0.1'        
SERVER_PORT =  33060 #5000
TIMEOUT = 30.0

# Camera
CAMERA_INDEX = 0

# Face Collection Config (Used for Capture Mode)
BEST_SAMPLES_TO_AVERAGE = 10 # Send 10 crops for full enrollment packet.

# Models
mp_face_mesh = mp.solutions.face_mesh

# Pose/Quality Thresholds
POSE_QUALITY_THRESHOLD_RE_ID = 0.50
POSE_QUALITY_THRESHOLD_ID = 0.89
SHARPNESS_THRESHOLD = 50.0

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
        
        if mean_brightness > 200 and std_brightness < 40: #this is for too bright so dont mess with this 
            gamma = 1.3; inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
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

    if right - left < 60 or bottom - top < 60: return None
    
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
        
        self.last_send_time = 0.0
        self.SEND_INTERVAL = 0.1 # Send at most 10 packets per second
        
        self.seq_num = 0 #initialize at 0, increment after receiving response
        self.recent_face_ids = [None] * 5 # Last 5 recognized face IDs for context
        
        # ID Mode State
        self.is_new_id = False
        self.capture_crops: List[np.ndarray] = [] # Accumulates the 10 crops
        
        self.tracker = SimpleFaceTracker(iou_threshold=0.3, max_frames_missed=5)
        self.track_id_to_data = {}  # track_id -> face_id
        
        self._connect_to_server()
        
    # Networking functions 

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
            
            clone = response_len_data
            
            # Handle connection loss and attempt to reconnect TODO: TEST THIS
            # if not response_len_data:
            #     # Connection broken, try to reconnect
            #     print("[WARN] Connection lost, reconnecting...")
            #     if not self._connect_to_server():
            #         return None
            #     # Retry sending the packet
            #     self.sock.sendall(serialized_packet)
            #     response_len_data = self._recv_exactly(4)
            #     if not response_len_data:
            #         return None
                
            response_len = struct.unpack('>I', response_len_data)[0]
            
            # Receive the IDPacket payload
            response_payload = self._recv_exactly(response_len) #accounting for seq num
            if not response_payload: return None
            
            return IDPacket.deserialize(clone + response_payload) #TODO: remove length prefix from deserialize method because this is so fucking stupid
                
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
            max_num_faces=2,
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
                        
                        # Use tighter threshold for ID, looser for re-ID
                        quality_threshold = POSE_QUALITY_THRESHOLD_ID if self.is_new_id else POSE_QUALITY_THRESHOLD_RE_ID
                        is_pose_ok = pose_score >= quality_threshold
                        
                        if is_sharp_enough and is_pose_ok:
                            # Store for tracking
                            if raw_face_crop is not None:
                                current_frame_boxes.append(track_box)
                                face_crops_for_boxes.append(processed_face_crop)
                        else:
                            # if status[:2] == "ID":
                            #     continue
                            
                            # # Quality check failed
                            # status = f"Poor Quality (S:{int(sharpness)} P:{int(pose_score*100)}%)"
                            # color = (100, 100, 100) # Grey
                            continue #TODO: reimplement UI for poor quality

                    # Update tracker: get persistent track_ids for this frame's boxes
                    tracker_results = self.tracker.update(current_frame_boxes)
                    
                    # Process each tracked face
                    for track_id, current_box in tracker_results.items():
                        # Find which crop index corresponds to this box
                        try:
                            box_index = current_frame_boxes.index(current_box)
                            current_crop = face_crops_for_boxes[box_index]
                        except ValueError:
                            continue # Box not found, skip
                        
                        track_data = {} # initialize empty
                        
                        # Check if this track_id already has a server-assigned face_id
                        if track_id in self.track_id_to_data:
                            track_data = self.track_id_to_data[track_id]
                            if track_data.get('server_id'):  # Already recognized
                                display_id = track_data['server_id']
                                status = f"ID #{display_id}"
                                
                                # Draw box with this ID
                                status = f"ID: #{display_id}"
                                color = (0, 255, 0) # Green
                            
                                # Draw bounding box and status
                                cv2.rectangle(frame, (current_box[0], current_box[1]), (current_box[2], current_box[3]), color, 2) #left, top, right, bottom
                                cv2.putText(frame, status, (current_box[0], current_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                continue  # Skip server query for this face

                        # If unknown, check if a request is pending for this track
                        if track_data != -1 and track_data.get('pending_seq_num'):
                            status = f"Verifying Track {track_id}..."
                        else:
                            # Send to server, get seq_num
                            packet = FacePacket(self.seq_num, [current_crop], self.recent_face_ids)
                            response = self._send_packet_and_receive_id(packet)

                            if response:
                                # Update your maps
                                if response.success:
                                    self.track_id_to_data[track_id] = {
                                        'server_id': response.face_id,
                                        'pending_seq_num': None
                                    }
                                else:
                                    self.track_id_to_data[track_id] = {
                                        'server_id': None,
                                        'pending_seq_num': self.seq_num  # Mark as pending
                                    }
                                self.seq_num += 1
                        
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