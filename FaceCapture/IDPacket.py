import cv2
import struct
import numpy as np

# PROTOCOL DESIGN:

# Structure: [length (4 bytes)][seq_num (4 bytes)][success (1 byte)][if success: face_id (4 bytes)]

# Protocol Size = 13 bytes if success, else 5 bytes

class IDPacket:
    def __init__(self, success, seq_num, face_id=None):
        self.success = success
        self.seq_num = seq_num
        self.face_id = face_id
        
    def serialize(self):
        packet_data = struct.pack('<?', self.success)
        
        if self.success:
            # Add face ID
            packet_data += struct.pack('<I', self.face_id)
        
        # Store length prefix
        total_length = len(packet_data)
        
        # Construct complete packet with header (total length + seq_num)
        return struct.pack('<I', total_length) + struct.pack('<I', self.seq_num) +packet_data
    
    @staticmethod
    def deserialize(data):
        """Deserializes IDPacket with length prefix"""
        try:
            # First 4 bytes are total packet length
            if len(data) < 4:
                return None
            
            # Read length prefix
            total_length = struct.unpack('<I', data[:4])[0]
            
            # Verify we have enough data
            if len(data) < 4 + total_length:
                return None
            
            # Skip length prefix
            packet_data = data[4:4 + total_length]
            current_pos = 0
            
            # Read sequence number
            seq_num = struct.unpack('<I', packet_data[current_pos:current_pos + 4])[0]
            current_pos += 4
            
            # Read success flag
            success_flag = struct.unpack('<?', packet_data[current_pos:current_pos + 1])[0]
            current_pos += 1
            
            if not success_flag:
                return IDPacket(False, seq_num)
            
            else:
                # Read packet data
                face_id = struct.unpack('<I', packet_data[current_pos:current_pos + 4])[0]
                
                return IDPacket(True, seq_num, face_id)
        
        except Exception as e:
            print(f"Deserialization error: {e}")
            return None