import cv2
import struct
import numpy as np

# PROTOCOL DESIGN:

# Structure: [length (4 bytes)][seq_num (4 bytes)][success (1 byte)] if failure: end packet
# if success: [header (9 bytes)][face_id (4 bytes)][similarity (4 bytes)][age (4 bytes)][name_length (4 bytes)][fullname (variable length)]

# Protocol Size = 21 + variable bytes if success, else 5 bytes

class IDPacket:
    def __init__(self, success, seq_num, face_id=None, similarity=None, age=None, fullname=None):
        self.success = success #bool
        self.seq_num = seq_num #int
        self.face_id = face_id #int
        self.similarity = similarity #float
        self.age = age #int
        self.fullname = fullname #string

    def serialize(self):
        # Add success flag
        packet_data = struct.pack('>?', self.success)
        
        if self.face_id is None:
            self.face_id = 0
        
        if self.similarity is None:
            self.similarity = 0.0
            
        # Add face ID
        packet_data += struct.pack('>I', self.face_id)
        # Add similarity
        packet_data += struct.pack('>f', self.similarity)
        
        if self.success:
            # Add age
            if self.age is None:
                self.age = 0
            packet_data += struct.pack('>I', self.age)
            # Add fullname length and data - name field is not null
            if self.fullname is None:
                self.fullname = ""
            fullname_bytes = self.fullname.encode('utf-8')
            packet_data += struct.pack('>I', len(fullname_bytes))
            packet_data += fullname_bytes
    
        # Store length prefix
        total_length = len(packet_data) + 4 # 4 bytes for sequence number
        
        # Create header
        header = struct.pack('>I', total_length) + struct.pack('>I', self.seq_num)
        
        # Construct complete packet with header (total length + seq_num)
        return header + packet_data
    
    @staticmethod
    def deserialize(packet_data):
        """Deserializes IDPacket with length prefix"""
        try:
            current_pos = 0
            
            # Read sequence number
            seq_num = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
            current_pos += 4
            
            # Read success flag
            success_flag = struct.unpack('>?', packet_data[current_pos:current_pos + 1])[0]
            current_pos += 1
            
            if not success_flag:
                # Read packet data
                face_id = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
                current_pos += 4
                
                similarity = struct.unpack('>f', packet_data[current_pos:current_pos + 4])[0]
                
                print(f"Deserialized IDPacket: seq_num={seq_num}, closest face_id={face_id}, similarity={similarity}")
                return IDPacket(False, seq_num)
            
            else:
                # Read packet data
                face_id = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
                current_pos += 4
                
                similarity = struct.unpack('>f', packet_data[current_pos:current_pos + 4])[0]
                current_pos += 4
                
                age = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
                current_pos += 4
                
                name_length = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
                current_pos += 4
                
                fullname = packet_data[current_pos:current_pos + name_length].decode('utf-8')
                
                print(f"Deserialized IDPacket: seq_num={seq_num}, face_id={face_id}, similarity={similarity}, age={age}, fullname={fullname}")
                
                return IDPacket(True, seq_num, face_id, similarity, age, fullname)
        
        except Exception as e:
            print(f"Deserialization error: {e}")
            return None