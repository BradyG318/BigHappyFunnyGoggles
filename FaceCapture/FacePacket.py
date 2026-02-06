import cv2
import struct
import numpy as np

# PROTOCOL DESIGN:

# Header [TCP Packet Length (4 bytes) | Sequence Number (4 bytes)]
# Payload [NumFaces (1 byte) | RecentIDs (5 * 4 bytes) | CropSizes (1 or 10 * 4 bytes) | Crops (1 or 10 * variable size)]

# Protocol Size = 29 bytes + (1 or 10 * (variable size + 4 bytes))

# Sent from glasses to server in one of two cases:
# 1) Re-identify - send 1 face crop
# 2) Capture new face - send 10 face crops

class FacePacket:
    def __init__(self, seq_num, face_crops, recent_ids=None):
        self.seq_num = seq_num
        
        self.face_crops = face_crops if isinstance(face_crops, list) else [face_crops]
        
        # Process recent face IDs - always store 5 IDs, None becomes -1 later
        self.recent_ids = [None] * 5  # Initialize with 5 None values
        if recent_ids:
            for i, face_id in enumerate(recent_ids[:5]):
                self.recent_ids[i] = face_id
    
    def serialize(self):
        """Compact binary format with recent IDs"""
        num_faces = len(self.face_crops)
        
        # Compress crops
        compressed_crops = []
        crop_sizes = []
        
        for face_crop in self.face_crops:
            if face_crop is None or face_crop.size == 0:
                compressed_crops.append(b'')
                crop_sizes.append(0)
            else:
                _, encoded = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop_data = encoded.tobytes()
                compressed_crops.append(crop_data)
                crop_sizes.append(len(crop_data))
        
        # Build payload
        payload = b''
        
        # Pack num faces into payload
        payload += struct.pack('>B', num_faces)
        
        # Add recent face IDs (always 5 IDs, -1 for None)
        for face_id in self.recent_ids[:5]:
            id_value = face_id if face_id is not None else -1
            payload += struct.pack('>i', id_value)  # 'i' for signed int (allows -1 for None)
        
        # Add crop sizes for each face
        for crop_size in crop_sizes:
            payload += struct.pack('>I', crop_size)  # 4 bytes per crop size
        
        # Combine with all crops
        for crop_data in compressed_crops:
            payload += crop_data
        
        # Get length prefix for TCP
        total_length = len(payload) + 4 # 4 bytes for sequence number
        
        # Construct header
        header = struct.pack('>I', total_length) + struct.pack('>I', self.seq_num)
        
        # Contruct complete packet with header
        complete_packet = header + payload
        
        return complete_packet
    
    @staticmethod
    def deserialize(data):
        """Deserializes TCP packet with length prefix"""
        try:
            # Verify we have enough data
            if len(data) < 5:
                return None
            
            # Read length prefix
            total_length = struct.unpack('>I', data[:4])[0]
            
            # Skip length prefix
            packet_data = data[4:4 + total_length]
            current_pos = 0
            
            # Read sequence number
            seq_num = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
            current_pos += 4
            
            # Read num faces
            num_faces = struct.unpack('>B', packet_data[current_pos:current_pos + 1])[0]
            current_pos += 1
            
            # Read exactly 5 recent face IDs
            recent_ids = []
            for _ in range(5):
                face_id = struct.unpack('>i', packet_data[current_pos:current_pos + 4])[0]
                recent_ids.append(face_id if face_id != -1 else None)
                current_pos += 4
            
            # Read crop sizes
            crop_sizes = []
            for _ in range(num_faces):
                crop_size = struct.unpack('>I', packet_data[current_pos:current_pos + 4])[0]
                crop_sizes.append(crop_size)
                current_pos += 4
            
            # Read crops
            face_crops = []
            for crop_size in crop_sizes:
                if crop_size == 0:
                    face_crops.append(None)
                else:
                    crop_data = packet_data[current_pos:current_pos + crop_size]
                    crop_array = np.frombuffer(crop_data, dtype=np.uint8)
                    face_crop = cv2.imdecode(crop_array, cv2.IMREAD_COLOR)
                    face_crops.append(face_crop)
                    current_pos += crop_size
            
            return FacePacket(seq_num, face_crops, recent_ids)
            
        except Exception as e:
            print(f"Deserialization error: {e}")
            return None