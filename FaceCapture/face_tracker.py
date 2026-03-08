import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from filterpy.kalman import KalmanFilter

# ----------------------------------------------------------------------
# Kalman filter wrapper using filterpy
# ----------------------------------------------------------------------
class FilterPyBoxKalman:
    """
    A Kalman filter for a bounding box using filterpy's KalmanFilter.
    State: [cx, cy, w, h, vx, vy, vw, vh]  (8 dimensions)
    Measurement: [cx, cy, w, h]            (4 dimensions)
    """
    def __init__(self, bbox: Tuple[int, int, int, int], dt=1.0):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # State transition matrix (constant velocity)
        self.kf.F = np.array([[1, 0, 0, 0, dt, 0, 0, 0],
                              [0, 1, 0, 0, 0, dt, 0, 0],
                              [0, 0, 1, 0, 0, 0, dt, 0],
                              [0, 0, 0, 1, 0, 0, 0, dt],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1]])
        # Measurement matrix
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0]])
        # Covariance matrices
        self.kf.P *= 100.0                     # initial uncertainty
        self.kf.Q = np.eye(8) * 0.01            # process noise
        self.kf.R = np.eye(4) * 0.1             # measurement noise

        # Initialize state with first measurement
        self.kf.x[:4, 0] = np.array([cx, cy, w, h])

    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state and return predicted box."""
        self.kf.predict()
        cx, cy, w, h = self.kf.x[:4, 0]
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        return (x1, y1, x2, y2)

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update with a new measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        z = np.array([[cx], [cy], [w], [h]])
        self.kf.update(z)


# ----------------------------------------------------------------------
# Track class (updated with FilterPyBoxKalman)
# ----------------------------------------------------------------------
class Track:
    """Internal class to hold state for a single tracked face."""
    def __init__(self, initial_box: Tuple[int, int, int, int], track_id: int):
        # Tracking details
        self.track_id = track_id
        self.current_box = initial_box
        self.missed_count = 0
        self.last_seen_time = time.time()

        # Kalman filter for this track
        self.kalman = FilterPyBoxKalman(initial_box)

        # Recognition fields
        self.server_id = None
        self.pending_seq_num = None
        self.last_recognition_time = 0
        self.recognition_cooldown = 0
        self.failed_attempts = 0
        
        # Buffer for retry logic
        self.crop_buffer = []
        self.buffer_full = False

    def update(self, new_box: Tuple[int, int, int, int]):
        self.kalman.update(new_box)
        self.current_box = new_box
        self.missed_count = 0
        self.last_seen_time = time.time()

    def predict(self) -> Tuple[int, int, int, int]:
        return self.kalman.predict()

    def mark_missed(self):
        self.missed_count += 1


# ----------------------------------------------------------------------
# Tracker class to keep track of multiple face_mesh detections
# ----------------------------------------------------------------------
class SimpleFaceTracker:
    """
    A lightweight IOU-based tracker with Kalman filtering (filterpy version).
    """
    def __init__(self, iou_threshold: float, max_frames_missed: int, max_age_seconds: float):
        self.iou_thresh = iou_threshold
        self.max_missed = max_frames_missed
        self.max_age_seconds = max_age_seconds
        self.next_track_id = 0
        self.active_tracks: Dict[int, Track] = {}

    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        # Same as before
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def get_active_tracks(self) -> Dict[int, Track]:
        return self.active_tracks

    def _match_boxes(self, predicted_boxes: List[Tuple], current_boxes: List[Tuple]) -> Tuple:
        # Identical to custom version
        if not self.active_tracks:
            return [], list(range(len(current_boxes))), []

        iou_matrix = np.zeros((len(self.active_tracks), len(current_boxes)))
        for i, pred_box in enumerate(predicted_boxes):
            for j, curr_box in enumerate(current_boxes):
                iou_matrix[i, j] = self._calculate_iou(pred_box, curr_box)

        matches = []
        unmatched_tracks = list(range(len(self.active_tracks)))
        unmatched_current = list(range(len(current_boxes)))

        if iou_matrix.size > 0:
            for _ in range(min(iou_matrix.shape)):
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_val = iou_matrix[max_idx]
                if max_val < self.iou_thresh:
                    break
                matches.append((max_idx[0], max_idx[1]))
                unmatched_tracks.remove(max_idx[0])
                unmatched_current.remove(max_idx[1])
                iou_matrix[max_idx[0], :] = -1
                iou_matrix[:, max_idx[1]] = -1

        return matches, unmatched_current, unmatched_tracks

    def update(self, current_boxes: List[Tuple]) -> Dict[int, Tuple]:
        current_time = time.time()

        # 1. Predict for all active tracks
        predicted_boxes = []
        for track in self.active_tracks.values():
            pred = track.predict()
            predicted_boxes.append(pred)

        # 2. Match
        matches, unmatched_current, unmatched_tracks = self._match_boxes(predicted_boxes, current_boxes)

        # 3. Update matched tracks
        for track_idx, box_idx in matches:
            track_id = list(self.active_tracks.keys())[track_idx]
            self.active_tracks[track_id].update(current_boxes[box_idx])

        # 4. Create new tracks for unmatched boxes
        for box_idx in unmatched_current:
            new_id = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[new_id] = Track(current_boxes[box_idx], new_id)

        # 5. Mark unmatched tracks missed, delete if needed
        to_delete = []
        for track_idx in unmatched_tracks:
            track_id = list(self.active_tracks.keys())[track_idx]
            self.active_tracks[track_id].mark_missed()
            track = self.active_tracks[track_id]
            if (track.missed_count > self.max_missed or
                current_time - track.last_seen_time > self.max_age_seconds):
                to_delete.append(track_id)

        for track_id in to_delete:
            del self.active_tracks[track_id]

        return {tid: track.current_box for tid, track in self.active_tracks.items() if track.missed_count == 0}