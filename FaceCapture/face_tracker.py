import time
import numpy as np
from typing import List, Tuple, Dict, Optional

# Helper object to represent individual face_mesh tracks
class Track:
    """Internal class to hold state for a single tracked face."""
    def __init__(self, initial_box: Tuple, track_id: int):
        self.track_id = track_id
        self.current_box = initial_box
        self.missed_count = 0
        self.last_seen_time = time.time()
        
        # Recognition fields
        self.server_id = None # Recognized face ID from server, if any
        self.pending_seq_num = None # Current pending sequence number for recognition request, if any
        self.last_recognition_time = 0 # Last tried to recognize track
        self.recognition_cooldown = 0 # When we can try again
        self.failed_attempts = 0 # Count of failed recognition attempts

    def update(self, new_box: Tuple):
        self.current_box = new_box
        self.missed_count = 0  # Reset on successful match
        self.last_seen_time = time.time()

    def mark_missed(self):
        self.missed_count += 1

class SimpleFaceTracker:
    """
    A lightweight IOU-based tracker that assigns persistent IDs to face_mesh tracking
    """
    def __init__(self, iou_threshold, max_frames_missed, max_age_seconds):
        """
        Args:
            iou_threshold: Minimum IOU to consider boxes a match. Tune: (0.2 - 0.5)
            max_frames_missed: Frames to keep a 'stale' track alive before deleting it.
            max_age_seconds: Seconds to keep a track alive before deleting it.
        """
        self.iou_thresh = iou_threshold
        self.max_missed = max_frames_missed
        self.max_age_seconds = max_age_seconds
        self.next_track_id = 0
        self.active_tracks: Dict[int, Track] = {}  # track_id -> Track object

    @staticmethod
    def _calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """Calculates Intersection over Union for two boxes."""
        # Unpack coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection area
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def get_active_tracks(self):
        """Returns a dict of active track_id -> current_box for all non-stale tracks."""
        return self.active_tracks

    def _match_boxes(self, current_boxes) -> Tuple:
        """Performs IOU matching between active tracks and current boxes."""
        if not self.active_tracks:
            return [], list(range(len(current_boxes))), []

        # Calculate IOU matrix
        iou_matrix = np.zeros((len(self.active_tracks), len(current_boxes)))
        for i, track in enumerate(self.active_tracks.values()):
            for j, curr_box in enumerate(current_boxes):
                iou_matrix[i, j] = self._calculate_iou(track.current_box, curr_box)

        # Simple greedy matching: pair highest IOU above threshold
        matches = []
        unmatched_tracks = list(range(len(self.active_tracks)))
        unmatched_current = list(range(len(current_boxes)))

        # Sort by IOU descending for priority matching
        if iou_matrix.size > 0:
            for i in range(min(iou_matrix.shape)):
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_val = iou_matrix[max_idx]
                if max_val < self.iou_thresh:
                    break
                matches.append((max_idx[0], max_idx[1]))
                unmatched_tracks.remove(max_idx[0])
                unmatched_current.remove(max_idx[1])
                iou_matrix[max_idx[0], :] = -1  # Invalidate row
                iou_matrix[:, max_idx[1]] = -1  # Invalidate column

        return matches, unmatched_current, unmatched_tracks

    def update(self, current_boxes):
        """
        Main function. Call this each frame with detected face boxes.
        Args:
            current_boxes: Dict of new_track_id -> bounding boxes from current frame.
                           Format: [(x1, y1, x2, y2), ...]
        Returns:
            A dictionary mapping {track_id: current_box} for all matched and active tracks.
        """
        # 1. Predict current positions of active tracks (assume they stay relatively static)
        #    In a more advanced version, you could use a Kalman filter here.

        # 2. Match current boxes to active tracks using IOU
        matches, unmatched_current, unmatched_tracks = self._match_boxes(current_boxes)

        current_time = time.time()

        # 3. Update matched tracks with new box, reset missed count
        for track_idx, box_idx in matches:
            track_id = list(self.active_tracks.keys())[track_idx]
            self.active_tracks[track_id].update(current_boxes[box_idx])

        # 4. Create new tracks for unmatched current boxes
        for box_idx in unmatched_current:
            new_id = self.next_track_id
            self.next_track_id += 1
            self.active_tracks[new_id] = Track(current_boxes[box_idx], new_id)

        # 5. Mark unmatched tracks as missed and delete if too stale
        to_delete = []
        for track_idx in unmatched_tracks:
            track_id = list(self.active_tracks.keys())[track_idx]
            self.active_tracks[track_id].mark_missed()
            
            # Delete if missed too many frames or if track is too old
            if (self.active_tracks[track_id].missed_count > self.max_missed or
                current_time - self.active_tracks[track_id].last_seen_time > self.max_age_seconds):
                to_delete.append(track_id)
                
        for track_id in to_delete:
            del self.active_tracks[track_id]

        # 6. Return the current state: track_id -> current box
        return {tid: track.current_box for tid, track in self.active_tracks.items() if track.missed_count == 0}