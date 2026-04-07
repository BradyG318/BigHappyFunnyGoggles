import cv2
import struct
import numpy as np

class BluetoothSettingsPacket:
    PACKET_TYPE_RECOGNIZED = 2

    def __init__(self, numPeople, showPotential, showDisplay, uiTransparency):
        self.numPeople = numPeople
        self.showPotential = showPotential
        self.showDisplay = showDisplay
        self.uiTransparency = uiTransparency

    # def deserialize(self):
    #     name_bytes = self.fullname.encode("utf-8")

    #     if self.face_crop is None or self.face_crop.size == 0:
    #         jpeg_bytes = b""
    #     else:
    #         ok, encoded = cv2.imencode(".jpg", self.face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    #         if not ok:
    #             jpeg_bytes = b""
    #         else:
    #             jpeg_bytes = encoded.tobytes()

    #     payload = b""
    #     payload += struct.pack(">B", self.PACKET_TYPE_RECOGNIZED)
    #     payload += struct.pack(">I", self.track_id)
    #     payload += struct.pack(">I", self.face_id)
    #     payload += struct.pack(">I", self.age)
    #     payload += struct.pack(">I", len(name_bytes))
    #     payload += struct.pack(">I", len(jpeg_bytes))
    #     payload += name_bytes
    #     payload += jpeg_bytes

    #     return struct.pack(">I", len(payload)) + payload