"""
ScreenSafe AI Analysis Pipeline

Object Tracker using Norfair for tracking detections across video frames.
Used for the Magic Wand tool and consistent detection IDs.
"""

import numpy as np
from typing import Optional, Callable, List, Dict, Union, Any
import logging

from .models import Detection, BoundingBox

logger = logging.getLogger(__name__)


class ObjectTracker:
    """
    Object tracker using Norfair for tracking detections across frames.
    
    Features:
    - Track detections to maintain consistent IDs
    - Support manual region tracking (Magic Wand)
    - Efficient Kalman filter-based prediction
    """
    
    def __init__(self, 
                 distance_threshold: float = 50.0,
                 hit_counter_max: int = 15,
                 initialization_delay: int = 3):
        """
        Initialize tracker.
        
        Args:
            distance_threshold: Max distance for matching detections
            hit_counter_max: Frames to keep track without detection
            initialization_delay: Frames before track is confirmed
        """
        self.distance_threshold = distance_threshold
        self.hit_counter_max = hit_counter_max
        self.initialization_delay = initialization_delay
        
        self._tracker = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Initialize Norfair tracker"""
        if self._initialized:
            return
        
        try:
            from norfair import Tracker
            from norfair.distances import create_normalized_mean_euclidean_distance
            
            # Distance function for matching
            distance_function = create_normalized_mean_euclidean_distance(
                1920, 1080  # Normalize to HD resolution
            )
            
            self._tracker = Tracker(
                distance_function=distance_function,
                distance_threshold=self.distance_threshold,
                hit_counter_max=self.hit_counter_max,
                initialization_delay=self.initialization_delay,
            )
            
            self._initialized = True
            logger.info("Norfair tracker initialized")
            
        except ImportError:
            logger.error("Norfair not installed. Run: pip install norfair")
            raise
    
    def update(self, 
               detections: List[Detection],
               frame_width: int,
               frame_height: int) -> List[Detection]:
        """
        Update tracker with new detections and assign track IDs.
        
        Args:
            detections: List of detections for current frame
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Detections with track_id assigned
        """
        if not self._initialized:
            self.initialize()
        
        if not detections:
            # Update tracker with no detections
            self._tracker.update([])
            return []
        
        try:
            from norfair import Detection as NorfairDetection
            
            # Convert to Norfair format
            norfair_detections = []
            detection_map = {}
            
            for det in detections:
                # Convert normalized bbox to pixel center point
                x1, y1, x2, y2 = det.bbox.to_pixels(frame_width, frame_height)
                center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                
                norfair_det = NorfairDetection(
                    points=center,
                    scores=np.array([det.confidence])
                )
                norfair_detections.append(norfair_det)
                detection_map[id(norfair_det)] = det
            
            # Update tracker
            tracked_objects = self._tracker.update(
                detections=norfair_detections
            )
            
            # Match tracked objects back to our detections
            result = []
            for tracked_obj in tracked_objects:
                if tracked_obj.last_detection is not None:
                    original_det = detection_map.get(id(tracked_obj.last_detection))
                    if original_det:
                        # Assign track ID
                        original_det.track_id = str(tracked_obj.id)
                        result.append(original_det)
            
            return result
            
        except Exception as e:
            logger.warning(f"Tracking failed: {e}")
            return detections
    
    def reset(self) -> None:
        """Reset tracker state"""
        if self._tracker:
            # Reinitialize to clear state
            self._initialized = False
            self.initialize()
    
    def cleanup(self) -> None:
        """Release resources"""
        self._tracker = None
        self._initialized = False


class SimpleBBoxTracker:
    """
    Simple bounding box tracker for when Norfair is not available.
    Uses IoU-based matching with simple ID assignment.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 10):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[str, Any] = {}
        self.next_id = 1
    
    def update(self, 
               detections: List[Detection],
               frame_number: int) -> List[Detection]:
        """Update tracks with new detections"""
        
        if not detections:
            # Age out old tracks
            self._age_tracks(frame_number)
            return []
        
        # Match detections to existing tracks
        matched_dets = []
        unmatched_dets = list(detections)
        
        for track_id, track in list(self.tracks.items()):
            best_match = None
            best_iou = 0
            
            for det in unmatched_dets:
                iou = self._compute_iou(track['bbox'], det.bbox)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_match = det
            
            if best_match:
                best_match.track_id = track_id
                matched_dets.append(best_match)
                unmatched_dets.remove(best_match)
                
                # Update track
                track['bbox'] = best_match.bbox
                track['last_seen'] = frame_number
        
        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            track_id = f"track_{self.next_id}"
            self.next_id += 1
            
            det.track_id = track_id
            matched_dets.append(det)
            
            self.tracks[track_id] = {
                'bbox': det.bbox,
                'last_seen': frame_number
            }
        
        # Age out old tracks
        self._age_tracks(frame_number)
        
        return matched_dets
    
    def _age_tracks(self, current_frame: int) -> None:
        """Remove tracks that haven't been seen recently"""
        to_remove = []
        for track_id, track in self.tracks.items():
            if current_frame - track['last_seen'] > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _compute_iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Compute Intersection over Union of two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1 = bbox1.x, bbox1.y
        x2_1, y2_1 = bbox1.x + bbox1.width, bbox1.y + bbox1.height
        
        x1_2, y1_2 = bbox2.x, bbox2.y
        x2_2, y2_2 = bbox2.x + bbox2.width, bbox2.y + bbox2.height
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def reset(self) -> None:
        """Reset tracker state"""
        self.tracks = {}
        self.next_id = 1


def get_tracker(use_norfair: bool = True) -> Union[ObjectTracker, SimpleBBoxTracker]:
    """Get appropriate tracker based on availability"""
    if use_norfair:
        try:
            import norfair
            return ObjectTracker()
        except ImportError:
            logger.warning("Norfair not available, using simple tracker")
    
    return SimpleBBoxTracker()
