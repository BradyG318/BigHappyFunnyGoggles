import cv2
import struct
import numpy as np

class BluetoothIdentityPacket:
    PACKET_TYPE_RECOGNIZED = 1

    def __init__(self, track_id, face_id, age, fullname, face_crop):
        self.track_id = track_id
        self.face_id = face_id
        self.age = 0 if age is None else age
        self.fullname = "" if fullname is None else fullname
        self.face_crop = face_crop

    def serialize(self):
        name_bytes = self.fullname.encode("utf-8")

        if self.face_crop is None or self.face_crop.size == 0:
            jpeg_bytes = b""
        else:
            ok, encoded = cv2.imencode(".jpg", self.face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                jpeg_bytes = b""
            else:
                jpeg_bytes = encoded.tobytes()

        payload = b""
        payload += struct.pack(">B", self.PACKET_TYPE_RECOGNIZED)
        payload += struct.pack(">I", self.track_id)
        payload += struct.pack(">I", self.face_id)
        payload += struct.pack(">I", self.age)
        payload += struct.pack(">I", len(name_bytes))
        payload += struct.pack(">I", len(jpeg_bytes))
        payload += name_bytes
        payload += jpeg_bytes

        return struct.pack(">I", len(payload)) + payload