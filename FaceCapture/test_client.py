import socket
import struct
import numpy as np
import cv2
import time
from FacePacket import FacePacket

def test_tcp_server(host='127.0.0.1', port=5000):
    """Simple test client for FaceRecognitionTCPServer that uses FacePacket"""
    
    def create_test_packet_using_facepacket():
        """Create a valid packet using FacePacket class"""
        # Create a dummy face image
        dummy_face = np.zeros((160, 160, 3), dtype=np.uint8)
        dummy_face[:80, :, 0] = 255  # Blue top half
        dummy_face[80:, :, 2] = 255  # Red bottom half
        
        # Add some text for identification
        cv2.putText(dummy_face, "TEST FACE", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Create recent IDs (None will become -1 in serialization)
        recent_ids = [101, None, 102, None, None]
        
        # Create FacePacket
        packet = FacePacket([dummy_face], recent_ids)
        
        # Serialize it
        serialized_data = packet.serialize()
        
        # Debug: Print packet structure
        print(f"\n=== Packet Structure Analysis ===")
        print(f"Total serialized bytes: {len(serialized_data)}")
        
        # Parse the serialized data to verify format
        if len(serialized_data) >= 4:
            total_length = struct.unpack('I', serialized_data[:4])[0]
            print(f"Length prefix (4 bytes): {total_length}")
            
            # Header: 1 byte num_faces
            num_faces = struct.unpack('B', serialized_data[4:5])[0]
            print(f"Num faces (1 byte): {num_faces}")
            
            # 5 recent IDs (20 bytes)
            pos = 5
            ids = []
            for i in range(5):
                face_id = struct.unpack('i', serialized_data[pos:pos+4])[0]
                ids.append(face_id)
                pos += 4
            print(f"Recent IDs (20 bytes): {ids}")
            
            # Crop size (4 bytes)
            crop_size = struct.unpack('I', serialized_data[pos:pos+4])[0]
            pos += 4
            print(f"Crop size (4 bytes): {crop_size}")
            
            # Crop data (variable)
            print(f"Crop data bytes: {len(serialized_data) - pos}")
            print(f"Total calculated: 4 + 1 + 20 + 4 + crop_data = {4 + 1 + 20 + 4 + crop_size}")
            print(f"Actual total: {len(serialized_data)}")
            print("===============================\n")
        
        return serialized_data
    
    try:
        print(f"Connecting to {host}:{port}...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        sock.connect((host, port))
        
        print("Connected! Creating test packet...")
        
        # Create and send first test packet
        packet1 = create_test_packet_using_facepacket()
        print(f"Sending {len(packet1)} bytes...")
        
        # Send the packet
        sock.sendall(packet1)
        print("First packet sent successfully!")
        
        # Try to receive any response
        try:
            sock.settimeout(2)
            response = sock.recv(1024)
            if response:
                print(f"Server response (first {min(100, len(response))} bytes): {response[:100]}...")
            else:
                print("No response received (connection closed)")
        except socket.timeout:
            print("No response (server may not be sending responses)")
        
        # Wait and send a second packet with different data
        time.sleep(1)
        print("\nCreating and sending second packet...")
        
        # Create second face with different properties
        dummy_face2 = np.ones((160, 160, 3), dtype=np.uint8) * 128
        dummy_face2[40:120, 40:120, :] = [0, 255, 0]  # Green square in center
        cv2.putText(dummy_face2, "TEST 2", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        recent_ids2 = [103, 104, None, 105, None]
        packet2 = FacePacket([dummy_face2], recent_ids2).serialize()
        
        sock.sendall(packet2)
        print(f"Second packet sent ({len(packet2)} bytes)!")
        
        # Test with multiple faces in one packet
        time.sleep(1)
        print("\nCreating and sending third packet with 2 faces...")
        
        dummy_face3a = np.zeros((160, 160, 3), dtype=np.uint8)
        dummy_face3a[:, :80, 0] = 255  # Blue left half
        
        dummy_face3b = np.zeros((160, 160, 3), dtype=np.uint8)
        dummy_face3b[:, 80:, 2] = 255  # Red right half
        
        recent_ids3 = [106, 107, 108, None, None]
        packet3 = FacePacket([dummy_face3a, dummy_face3b], recent_ids3).serialize()
        
        sock.sendall(packet3)
        print(f"Third packet with 2 faces sent ({len(packet3)} bytes)!")
        
    except ConnectionRefusedError:
        print(f"Connection refused. Is the server running on {host}:{port}?")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'sock' in locals():
            sock.close()
            print("\nConnection closed")

if __name__ == "__main__":
    # Test local server
    test_tcp_server('127.0.0.1', 5000)