import socket
import time
import struct
import cv2
import logging
import argparse
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress TensorFlow logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging
from deepface import DeepFace

from FacePacket import FacePacket #receive
from IDPacket import IDPacket #send
import DB_Link

# Demo version of the Face Recognition Server - id mode only

class FaceRecognitionServer:
    # ~~~ CONSTANTS ~~~
    # Model
    DEEPFACE_MODEL = 'Facenet512'

    # Recognition Threshold 
    RECOGNITION_THRESHOLD = 0.85

    # Pose Thresholds (ID is looser than Capture)
    POSE_QUALITY_THRESHOLD_ID = 0.50
    POSE_QUALITY_THRESHOLD_CAPTURE = 0.89

    # Sharpness: Laplacian Variance
    MIN_SHARPNESS_THRESHOLD = 50.0
    
    # To avoid duplicate tracking in one session TODO: implement
    currently_tracked_faces = set()
    
    # ~~~ SERVER FUNCTIONS ~~~
    def __init__(self, host='10.111.104.220', port=5000):
        """
        TCP Server for receiving face packets
        Args:
            host: Listen address
            port: TCP port
        """
        self.host = host
        self.port = port
        
        # TCP Server
        self.server_socket = None
        self.running = False
        
        # Data structures
        self.known_face_encodings = {} # face_id -> embedding
        self.known_face_ids = []
        self.next_face_id = 1
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FaceRecognitionServer')
        
    def _start(self):
        """Start TCP server"""
        try:
            # Create TCP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1) # Only allow 1 connection
            
            self.server_socket.settimeout(1.0)  # Set accept timeout to allow graceful shutdown
            
            self.running = True
            
            self.logger.info(f"TCP Server started on {self.host}:{self.port}")
            
            # Main thread
            while self.running:
                try:
                    client_socket, client_addr = self.server_socket.accept()
                    self.logger.info(f"Accepted connection from {client_addr}")
                    
                    # Handle connection
                    self._accept_connection(client_socket, client_addr)
                    
                    # Connection closed, wait for new one
                    self.logger.info("Connection closed, waiting for new connection...")
                
                except KeyboardInterrupt:
                    self.logger.info("Shutdown requested...")
                    break
                
                except socket.timeout:
                    continue  # Timeout occurred, loop back to check self.running
                
                except Exception as e:
                    if self.running:
                        self.logger.error(f"Connection error: {e}")
                    continue
                
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            
        finally:
            self._stop()
    
    def _accept_connection(self, client_socket, client_addr): 
        """Accept incoming TCP connection from glasses"""
        try:
            client_socket.settimeout(60.0)  # Set socket timeout
                        
            # Process packets in loop
            while self.running:
                try:
                    # Read packet length prefix (4 bytes)
                    length_data = self._recv_exactly(client_socket, 4)
                    
                    if not length_data:
                        break  # Connection closed
                    
                    # Unpack length
                    packet_length = struct.unpack('<I', length_data)[0]
                    
                    # Read the actual packet
                    packet_data = self._recv_exactly(client_socket, packet_length)
                    
                    #DEBUG might wanna wait for full packet
                    
                    if not packet_data:
                        break
                    
                    # Process full packet
                    seq_num, response = self._process_packet(length_data + packet_data, client_addr)
                    
                    self.logger.debug("DEBUG: seq_num =", {seq_num}, "response =", {response})
                    
                    # Send back response with recognition result
                    self.send_result(client_socket, seq_num, response)
                
                except socket.timeout:
                    self.logger.debug(f"Connection from {client_addr} timed out")
                    continue # Continue to wait for new packets
                
                except ConnectionResetError:
                    self.logger.info(f"Connection from {client_addr} reset by peer")
                    break
                
        except Exception as e:
            self.logger.error(f"Error accepting connection: {e}")
            
        finally:
            client_socket.close()
            self.logger.info(f"Connection from {client_addr} closed")
    
    def _recv_exactly(self, sock, n):
        """Receive exactly n bytes from socket"""
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None  # Connection closed
                data += chunk
                
            except socket.timeout:
                return None
            
            except Exception as e:
                self.logger.error(f"Receive error: {e}")
                return None
            
        return data

    def _process_packet(self, packet_data, client_addr):
        """Process a single face packet"""
        start_time = time.time() #debug
        
        try:
            # Deserialize packet
            packet = FacePacket.deserialize(packet_data)
            
            if packet is None:
                self.logger.warning(f"Invalid packet from {client_addr}")
                return None, None
            
            # Extract data
            face_crops = packet.face_crops
            recent_ids = packet.recent_ids
            seq_num = packet.seq_num
            
            self.logger.debug(f"Processing packet from {client_addr}: {len(face_crops)} faces")
            
            # Recognize face/faces
            result = self.recognize_face(face_crops, recent_ids)
            
            self.logger.debug(f"Packet from {client_addr} processed in {time.time() - start_time:.2f}s")
            return seq_num, result
            
        except Exception as e:
            self.logger.error(f"Packet processing error from {client_addr}: {e}")
            return None, None

    def _stop(self):
        """Stop server gracefully"""
        self.logger.info("Stopping TCP server...")
        self.running = False
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        cv2.destroyAllWindows()
        self.logger.info("TCP Server stopped")
    
    # ~~~ FACE RECOGNITION FUNCTIONS ~~~
    def get_deepface_embedding(self, face_crop):
        """
        Uses DeepFace to encode the cropped face image into a feature vector (embedding).
        """
        if face_crop is None or face_crop.size == 0:
            return None
        
        try:
            # Convert from Unity's RGB to BGR for OpenCV
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            #DEBUG show image
            cv2.imshow("Face Crop", face_crop)
            
            cv2.waitKey(1)
            
            embeddings = DeepFace.represent(
                img_path=face_crop, 
                model_name=self.DEEPFACE_MODEL, 
                enforce_detection=False,
                align=True 			    
            )
            
            if embeddings:
                return np.array(embeddings[0]['embedding'])
            else:
                return None

        except Exception as e:
            self.logger.debug(f"DeepFace embedding error: {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors (range -1 to 1)"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def recognize_by_range(self, embedding, face_ids):
        """
        Recognize a face embedding against a provided list of face ids.
        Returns the best matching face id or None.
        """
        # Initialize best similarity and match id
        best_similarity = -1
        best_match_id = None
        
        # Check ids in provided range
        for face_id in face_ids:
            if face_id is None:
                continue
            
            if face_id in self.known_face_ids:
                similarity = self.cosine_similarity(embedding, self.known_face_encodings[face_id])
                        
                if similarity > best_similarity and similarity >= self.RECOGNITION_THRESHOLD:
                    best_similarity = similarity
                    best_match_id = face_id
        
        # Return best match or None if no match found
        return best_match_id
    
    def recognize_face(self, face_crops, recent_ids):
        """
        Creates an embedding for the single face sent
        Checks it against recent IDs
        Returns recognized face ID or None
        """
        try:
            # Check number of faces sent (check ID vs Capture)
            num_faces = len(face_crops)
            
            self.logger.debug(f"Recognizing {num_faces} face(s)")
            
            # Get encoding for single face
            embedding = self.get_deepface_embedding(face_crops[0])

            #DEBUG chew rest of empty list?
            
            if embedding is None:
                self.logger.info("No valid embedding generated for face")
                return None
            
            # Check identity against all known faces
            match_id = self.recognize_by_range(embedding, self.known_face_ids) #DEBUG we could later consider adding a bonus for recent ids ONLY IN capture case re-id where they previously failed
                
            if match_id is not None:
                self.logger.info(f"Face recognized as ID #{match_id}")
                    
                return match_id
                
            else:
                self.logger.info("Face not recognized")
                    
                return None
            
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
    
    def send_result(self, client_socket, seq_num, result):
        """Send recognition result back to client by IDPacket"""
        try:
            # Create IDPacket based on result
            if result is not None:
                response_packet = IDPacket(True, seq_num, result)
            else:
                response_packet = IDPacket(False, seq_num)
            
            response_data = response_packet.serialize()
            
            # Send the 4-byte length prefix FIRST
            length_prefix = struct.pack('<I', len(response_data))
            client_socket.sendall(length_prefix)
            
            # Then send the actual packet data
            client_socket.sendall(response_data)
            
            self.logger.info(f"Sent response for seq_num {seq_num}: success={response_packet.success}")
            
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
    
    def load_data_from_database(self):
        # Load existing vectors from database
        try:            
            vectors_dict = DB_Link.db_link.get_all_vectors()
            for face_id_str, vector_list in vectors_dict.items():
                face_id = int(face_id_str) # Convert string key to int
                self.known_face_ids.append(face_id)
                self.known_face_encodings[face_id] = np.array(vector_list)
                self.next_face_id = max(self.next_face_id, face_id + 1)

            self.logger.info(f"Loaded {len(self.known_face_ids)} existing faces from database. Next ID will be {self.next_face_id}.")
        
        except Exception as e:
            self.logger.error(f"Error loading from database: {e}. Starting fresh.")
    
    def save_data_to_database(self, face_id, encoding):
        """
        Saves the vector to PostgreSQL database.
        """
        # Save the final vector to database synchronously
        success = DB_Link.db_link.save_face_vector(face_id, encoding.tolist())
    
        if not success:
            self.logger.error(f"!!! ERROR saving vector to database for face #{face_id}")
            return False

        self.logger.info(f"Added face ID {face_id} to database")
        
        return True

# ~~~ MAIN ~~~
if __name__ == "__main__":    
    # --- CONFIGURATION ---
    # Database initialization
    DB_Link.db_link.initialize()
    #DB_Link.db_link.clear_db() # For testing, clear on startup
    
    parser = argparse.ArgumentParser(description='Face Recognition TCP Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    #if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start server
    server = FaceRecognitionServer(
        host=args.host,
        port=args.port
    )
    
    # Load known faces from database
    server.load_data_from_database()
    
    try:
        server._start() #_start() -> _accept_connection() -> _process_packet() -> recognize_face()
    except KeyboardInterrupt:
        server.logger.info("Server shutdown requested by user")
    except Exception as e:
        server.logger.error(f"Server error: {e}")
    finally:
        server._stop()