# test_final_working.py
import socket
import struct
import numpy as np
import cv2
import time

from FacePacket import FacePacket

# run server first obviously

class WorkingClient:
    def __init__(self, host='127.0.0.1', port=5000):
        self.host = host
        self.port = port
        self.client_socket = None
        self.seq_num = 0
    
    def connect(self):
        """Connect to the server"""
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.settimeout(10.0)
        self.client_socket.connect((self.host, self.port))
        print(f"✅ Connected to {self.host}:{self.port}")
    
    def disconnect(self):
        """Disconnect from server"""
        if self.client_socket:
            self.client_socket.close()
        print("✅ Disconnected")
    
    def send_and_receive(self, face_crops, recent_ids=None):
        """Send packet and receive response"""
        if recent_ids is None:
            recent_ids = []
        
        self.seq_num += 1
        
        # Create FacePacket
        packet = FacePacket(seq_num=self.seq_num, face_crops=face_crops, recent_ids=recent_ids)
        packet_data = packet.serialize()
        
        # Add TCP length prefix
        tcp_length = len(packet_data)
        tcp_length_prefix = struct.pack('I', tcp_length)
        
        print(f"\n📤 Sending packet #{self.seq_num} ({len(face_crops)} faces)")
        
        # Send
        self.client_socket.sendall(tcp_length_prefix + packet_data)
        
        # Receive response
        return self._receive_response()
    
    def _receive_response(self):
        """Receive and parse response"""
        try:
            # Read TCP length
            tcp_length_data = self._recv_exactly(4)
            if not tcp_length_data:
                return None
            
            tcp_length = struct.unpack('I', tcp_length_data)[0]
            
            # Read IDPacket data
            idpacket_data = self._recv_exactly(tcp_length)
            if not idpacket_data:
                return None
            
            # Parse based on observed format
            # From debug: responses are 9 bytes: [success][face_id][seq_num]
            if len(idpacket_data) == 9:
                success = struct.unpack('?', idpacket_data[0:1])[0]
                face_id = struct.unpack('I', idpacket_data[1:5])[0]
                seq_num = struct.unpack('I', idpacket_data[5:9])[0]
                
                print(f"📥 Response: success={success}, face_id={face_id}, seq={seq_num}")
                
                if success:
                    print(f"✅ Recognized as Face ID #{face_id}")
                    return face_id
                else:
                    print("❌ Face not recognized")
                    return None
            else:
                print(f"⚠️ Unexpected response format: {len(idpacket_data)} bytes")
                return None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def _recv_exactly(self, n):
        """Receive exactly n bytes"""
        data = b''
        while len(data) < n:
            try:
                chunk = self.client_socket.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                return None
        return data
    
    def create_test_image(self, seed=0):
        """Create test image"""
        np.random.seed(seed)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        color = (np.random.randint(0, 255), 
                np.random.randint(0, 255), 
                np.random.randint(0, 255))
        cv2.rectangle(img, (20, 20), (80, 80), color, -1)
        return img

def main():
    print("="*60)
    print("Face Recognition Server Test - FINAL")
    print("="*60)
    
    client = WorkingClient('127.0.0.1', 5000)
    
    try:
        client.connect()
        
        # Wait for server to be ready
        time.sleep(1)
        
        # Test 1: Should fail (no faces in DB)
        print("\n" + "-"*40)
        print("Test 1: Recognize face (should fail)")
        img1 = client.create_test_image(1)
        result1 = client.send_and_receive([img1], [])
        
        time.sleep(1)
        
        # Test 2: Capture new face
        print("\n" + "-"*40)
        print("Test 2: Capture new face")
        images = [client.create_test_image(i) for i in range(3)]
        result2 = client.send_and_receive(images, [])
        
        time.sleep(1)
        
        # Test 3: Recognize the new face
        print("\n" + "-"*40)
        print("Test 3: Recognize existing face")
        img3 = client.create_test_image(1)  # Same as test 1
        result3 = client.send_and_receive([img3], [1])
        
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.disconnect()

if __name__ == "__main__":
    main()