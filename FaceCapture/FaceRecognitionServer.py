import socket
import threading
import time
import struct
import cv2
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore") # deals with deprecation warnings from mediapipe
import mediapipe as mp
from deepface import DeepFace
import math

from FacePacket import FacePacket
from IDPacket import IDPacket
import DB_Link

# Reworked captureFaces_alpha.py into a TCP server structure

class FaceRecognitionServer:
    # ~~~ SERVER FUNCTIONS ~~~
    def __init__(self, host='0.0.0.0', port=5000, max_connections=1):
        """
        TCP Server for receiving face packets
        Args:
            host: Listen address
            port: TCP port
            max_connections: Maximum concurrent connections
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections
        
        # TCP Server
        self.server_socket = None
        self.running = False
        
        # Connection tracking
        self.active_connections = set()
        self.connection_lock = threading.Lock()
        
        # Processing thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # Statistics
        self.stats = {
            'connections_total': 0,
            'connections_active': 0,
            'packets_received': 0,
        }
        
        # Known faces database
        self.known_faces = {}  # face_id -> embedding
        self.face_encodings = []
        self.face_ids = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FaceRecognitionServer')
        
    def start(self):
        """Start TCP server"""
        try:
            # Create TCP socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            self.server_socket.settimeout(5.0)  # For graceful shutdown
            
            self.running = True
            
            self.logger.info(f"TCP Server started on {self.host}:{self.port}")
            self.logger.info(f"Maximum connections: {self.max_connections}")
            
            # Start connection acceptor thread
            acceptor_thread = threading.Thread(target=self._accept_connections, daemon=True)
            acceptor_thread.start()
            
            # Start monitor thread
            monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            monitor_thread.start()
            
            # Keep main thread alive
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested...")
                
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
        finally:
            self.stop()
    
    def _accept_connections(self):
        """Accept incoming TCP connections"""
        while self.running:
            try:
                client_socket, client_addr = self.server_socket.accept()
                client_socket.settimeout(5.0)  # Set socket timeout
                
                # Update stats
                self.stats['connections_total'] += 1
                
                # Create connection handler thread
                conn_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket, client_addr),
                    daemon=True
                )
                conn_thread.start()
                
                # Track connection
                with self.connection_lock:
                    self.active_connections.add(client_socket)
                    self.stats['connections_active'] = len(self.active_connections)
                
                self.logger.info(f"New connection from {client_addr}. Active: {self.stats['connections_active']}")
                
            except socket.timeout:
                continue  # Normal for accept with timeout
            except Exception as e:
                if self.running:
                    self.logger.error(f"Accept error: {e}")
    
    def _handle_connection(self, client_socket, client_addr):
        """Handle a single TCP connection"""
        connection_id = f"{client_addr[0]}:{client_addr[1]}"
        
        try:
            self.logger.debug(f"Handling connection {connection_id}")
            
            while self.running:
                # Read packet length (4 bytes)
                length_data = self._recv_exactly(client_socket, 4)
                if not length_data:
                    break  # Connection closed
                
                packet_length = struct.unpack('I', length_data)[0]
                
                # Read the actual packet
                packet_data = self._recv_exactly(client_socket, packet_length)
                if not packet_data:
                    break
                
                # Update stats
                self.stats['packets_received'] += 1
                
                # Submit for processing
                self.thread_pool.submit(
                    self._process_packet,
                    length_data + packet_data, client_addr, connection_id
                )
                
        except socket.timeout:
            self.logger.warning(f"Connection {connection_id} timed out")
        except Exception as e:
            self.logger.error(f"Connection {connection_id} error: {e}")
        finally:
            # Cleanup
            client_socket.close()
            with self.connection_lock:
                if client_socket in self.active_connections:
                    self.active_connections.remove(client_socket)
                    self.stats['connections_active'] = len(self.active_connections)
            
            self.logger.info(f"Connection {connection_id} closed. Active: {self.stats['connections_active']}")
    
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

    def _monitor_loop(self):
        """Monitor server performance"""
        while self.running:
            time.sleep(10)  # Log every 10 seconds
            
            self.logger.info(
                f"Connections: {self.stats['connections_active']}/{self.stats['connections_total']} | "
                f"Packets: {self.stats['packets_received']} | "
            )

    def stop(self):
        """Stop server gracefully"""
        self.logger.info("Stopping TCP server...")
        self.running = False
        
        # Close all active connections
        with self.connection_lock:
            for sock in list(self.active_connections):
                try:
                    sock.close()
                except:
                    pass
            self.active_connections.clear()
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        cv2.destroyAllWindows()
        self.logger.info("TCP Server stopped")

    def _process_packet(self, packet_data, client_addr, connection_id):
        """Process a single face packet"""
        start_time = time.time()
        
        try:
            # Deserialize packet (note: includes length prefix in packet_data)
            packet = FacePacket.deserialize(packet_data)
            
            if packet is None:
                self.logger.warning(f"Invalid packet from {connection_id}")
                return
            
            # Extract data
            face_crops = packet.face_crops
            recent_ids = packet.recent_face_ids
            
            self.logger.debug(f"Processing packet from {connection_id}: {len(face_crops)} faces")
            
            # Process each face
            for i, crop in enumerate(face_crops):
                if crop is not None:
                    self._recognize_face(crop, recent_ids, client_addr, i)
            
            # Optional: Send response
            # self._send_response(client_socket, result)
            
        except Exception as e:
            self.logger.error(f"Packet processing error from {connection_id}: {e}")
        finally:
            processing_time = time.time() - start_time
            if processing_time > 0.5:  # Log slow processing
                self.logger.warning(f"Slow processing: {processing_time:.2f}s for packet from {connection_id}")
    
    # ~~~ FACE RECOGNITION FUNCTIONS ~~~
    def _recognize_face(self, face_crop, recent_ids, client_addr, face_index): #TBD
        """
        Face recognition logic
        TODO: Implement DeepFace recognition here
        """
        try:
            # Example: Display face (debug only)
            if self.logger.level <= logging.DEBUG:
                window_name = f"Face from {client_addr[0]}:{face_index}"
                cv2.imshow(window_name, face_crop)
                cv2.waitKey(1)
            
            # Log face details
            self.logger.debug(f"Face {face_index} from {client_addr[0]}: {face_crop.shape}")
            
            # TODO: Implement recognition
            # 1. Use DeepFace to get embedding
            # 2. Check recent_ids first (temporal locality)
            # 3. Check all known faces
            # 4. Update tracking
            
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
    
    def _send_response(self, client_socket, result): #TBD
        """Send recognition result back to client BY IDPacket"""
        try:
            # Example response format
            response = {
                'status': 'processed',
                'timestamp': time.time(),
                'matches': result
            }
            response_data = pickle.dumps(response)
            response_length = struct.pack('I', len(response_data))
            client_socket.sendall(response_length + response_data)
        except Exception as e:
            self.logger.error(f"Failed to send response: {e}")
    
    def add_known_face(self, face_id, embedding): #TBD
        """Add a known face to database"""
        self.known_faces[face_id] = embedding
        self.face_ids.append(face_id)
        self.face_encodings.append(embedding)
        self.logger.info(f"Added face ID {face_id} to database")

# ~~~ MAIN ~~~
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Face Recognition TCP Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--max-conn', type=int, default=10, help='Maximum connections')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start server
    server = FaceRecognitionServer(
        host=args.host,
        port=args.port,
        max_connections=args.max_conn
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.logger.info("Server interrupted by user")
    except Exception as e:
        server.logger.error(f"Server error: {e}")
    finally:
        server.stop()