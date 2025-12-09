import cv2
import numpy as np
import warnings
import mediapipe as mp
import math
import socket
import struct
import time
import argparse
from typing import List, Optional, Tuple, Any

warnings.filterwarnings("ignore") 

#Packets 
from FacePacket import FacePacket
from IDPacket import IDPacket 

#Client Config 

# Network
SERVER_HOST = '127.0.0.1' 
SERVER_PORT = 5000 
TIMEOUT = 30.0 

# Camera
CAMERA_INDEX = 0 
FPS = 30 

# Face Collection Config (Used for Capture Mode)
BEST_SAMPLES_TO_AVERAGE = 10 # Send 10 crops for full enrollment packet.

# Models
mp_face_mesh = mp.solutions.face_mesh

# Pose/Quality Thresholds
POSE_QUALITY_THRESHOLD_ID = 0.50
POSE_QUALITY_THRESHOLD_CAPTURE = 0.89
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
        
        if mean_brightness > 200 and std_brightness < 40:
            gamma = 1.3; inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(face_crop, table)
        elif mean_brightness < 40:
            alpha = 1.2; beta = 30
            return cv2.convertScaleAbs(face_crop, alpha=alpha, beta=beta)
        else:
            return face_crop
    except Exception:
        return face_crop

def get_face_crop(frame: np.ndarray, face_landmarks) -> Optional[np.ndarray]:
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
    
    return frame[top:bottom, left:right]

class FaceCaptureClient:
    """
    Client application (running on glasses) to detect faces via MediaPipe,
    check quality, crop, and send them to the server for recognition/capture.
    """
    def __init__(self, host=SERVER_HOST, port=SERVER_PORT):
        self.host = host
        self.port = port
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.recent_face_ids: List[Optional[int]] = [None] * 5
        self.seq_num = 0 #initialize at 0, increment where needed
        
        # Capture Mode State
        self.is_capturing_new_face = False
        self.capture_crops: List[np.ndarray] = [] # Accumulates the 10 crops
        
        if not self.cap.isOpened():
            raise IOError(f"Error: Could not open camera {CAMERA_INDEX}")
        
    # Networking functions 

    def _recv_exactly(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """Receive exactly n bytes from socket, handling fragmented reads."""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk: return None
                data += chunk
            except socket.timeout:
                raise socket.timeout("Timed out waiting for full packet.")
            except Exception:
                return None
        return data

    def _send_packet_and_receive_id(self, packet: FacePacket) -> Optional[IDPacket]:
        """Connects to server, sends packet, and receives IDPacket reliably."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(TIMEOUT)
                s.connect((self.host, self.port))
                
                serialized_packet = packet.serialize()
                s.sendall(serialized_packet)
                
                # Receive IDPacket length (4 bytes)
                response_len_data = self._recv_exactly(s, 4)
                if not response_len_data: return None
                
                response_len = struct.unpack('I', response_len_data)[0]
                
                # Receive the IDPacket payload
                response_payload = self._recv_exactly(s, response_len)
                if not response_payload: return None
                
                return IDPacket.deserialize(response_len_data + response_payload)
                
        except socket.timeout as e:
            print(f"[ERROR] Socket timeout: {e}")
            return None
        except socket.error as e:
            print(f"[ERROR] Socket error: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Network communication error: {e}")
            return None

    # Main capture loop 

    def run(self):
        """Main loop for face detection, quality check, and server communication."""
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

            print(f"Client running. Sending face data to {self.host}:{self.port}...")
            print("Press 'q' or 'Esc' to quit the application.")

            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success: continue

                frame.flags.writeable = False
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.flags.writeable = True

                status = "Searching..."
                color = (255, 0, 0) # Blue (Searching)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        
                        # Pre-processing and quality checks 
                        raw_face_crop = get_face_crop(frame, face_landmarks)
                        if raw_face_crop is None: continue 

                        processed_face_crop = conservative_lighting_normalization(raw_face_crop)
                        
                        sharpness = get_image_sharpness(processed_face_crop)
                        pose_score = get_pose_quality(face_landmarks)
                        
                        is_sharp_enough = sharpness >= SHARPNESS_THRESHOLD
                        
                        # Use tighter threshold for capture, looser for ID
                        quality_threshold = POSE_QUALITY_THRESHOLD_CAPTURE if self.is_capturing_new_face else POSE_QUALITY_THRESHOLD_ID
                        is_pose_ok = pose_score >= quality_threshold

                        # Sending data 
                        
                        if is_sharp_enough and is_pose_ok:
                            
                            if self.is_capturing_new_face:
                                # MODE: CAPTURE (Accumulate and send 10 crops)
                                
                                self.capture_crops.append(processed_face_crop)
                                
                                if len(self.capture_crops) >= BEST_SAMPLES_TO_AVERAGE:
                                    # Accumulation complete: Send the final 10-crop packet
                                    
                                    packet = FacePacket(self.seq_num, self.capture_crops, self.recent_face_ids)
                                    response = self._send_packet_and_receive_id(packet)
                                    
                                    # Reset mode immediately after sending the packet
                                    self.is_capturing_new_face = False
                                    self.capture_crops = []
                                    
                                    if response and response.success:
                                        # Enrollment Successful
                                        status = f"Enrollment Complete! ID: {response.face_id}"
                                        color = (0, 255, 0) # Green
                                        
                                        self.seq_num += 1 #WARNING/DEBUG: THIS IS NOT CORRECT PLACEMENT, WILL NEED TO BE UPDATED TO ASSOCIATE WITH A SPECIFIC DETECTION
                                    else:
                                        status = "Enrollment Failed (Server Error)"
                                        color = (0, 0, 255) # Red

                                else:
                                    status = f"CAPTURING... {len(self.capture_crops)}/{BEST_SAMPLES_TO_AVERAGE}"
                                    color = (255, 165, 0) # Orange

                            else:
                                # MODE: IDENTIFICATION (Send 1 crop)
                                packet = FacePacket(self.seq_num, [processed_face_crop], self.recent_face_ids)
                                response = self._send_packet_and_receive_id(packet)
                                
                                if response and response.success and response.face_id:
                                    # KNOWN FACE (Re-ID successful)
                                    face_id = response.face_id
                                    self.recent_face_ids.insert(0, face_id)
                                    self.recent_face_ids = self.recent_face_ids[:5]
                                    
                                    status = f"ID #{face_id} Found"
                                    color = (0, 255, 0) 
                                else:
                                    # UNKNOWN FACE (Failed Re-ID): Automatically initiate Capture Mode
                                    self.is_capturing_new_face = True
                                    self.capture_crops = [processed_face_crop] # Add the first quality crop
                                    
                                    status = f"Unknown Face. Starting Capture (1/{BEST_SAMPLES_TO_AVERAGE})"
                                    color = (255, 255, 0) # Yellow
                        else:
                            # Quality check failed
                            status = f"Poor Quality (S:{int(sharpness)} P:{int(pose_score*100)}%)"
                            color = (100, 100, 100) # Grey
                        
                # Drawing the frame 
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.imshow('Face Capture Client (Glasses)', frame)
                
                # Handle keyboard inputs (only for quitting)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27: 
                    break

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