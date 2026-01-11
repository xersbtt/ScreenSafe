"""
ScreenSafe AI Analysis Pipeline

Motion Tracker - Tracks a selected region as it moves across frames,
outputting per-frame bounding box positions for dynamic blur application.

This is essential for:
- Scrolling content (text that moves up/down)
- Moving objects (dragged windows, etc.)
- Panning/zooming content
"""

import cv2
import numpy as np
import logging
from typing import Callable, Optional, Tuple, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MotionTrackResult:
    """Result of motion tracking with per-frame positions"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    # Per-frame positions: frame_number -> normalized (x, y, w, h)
    frame_positions: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)
    
    def get_position_at_frame(self, frame: int) -> Optional[Tuple[float, float, float, float]]:
        """Get interpolated position at a specific frame"""
        if frame in self.frame_positions:
            return self.frame_positions[frame]
        
        # Interpolate between known positions
        frames = sorted(self.frame_positions.keys())
        if not frames:
            return None
        
        if frame <= frames[0]:
            return self.frame_positions[frames[0]]
        if frame >= frames[-1]:
            return self.frame_positions[frames[-1]]
        
        # Find surrounding frames and interpolate
        for i, f in enumerate(frames[:-1]):
            if frames[i] <= frame <= frames[i + 1]:
                f1, f2 = frames[i], frames[i + 1]
                t = (frame - f1) / (f2 - f1)  # Interpolation factor
                
                p1 = self.frame_positions[f1]
                p2 = self.frame_positions[f2]
                
                # Linear interpolation
                return (
                    p1[0] + t * (p2[0] - p1[0]),
                    p1[1] + t * (p2[1] - p1[1]),
                    p1[2] + t * (p2[2] - p1[2]),
                    p1[3] + t * (p2[3] - p1[3])
                )
        
        return None


class MotionTracker:
    """
    Track a selected region as it moves through the video.
    
    Uses OpenCV CSRT tracker for accurate motion following, combined
    with content verification to detect when the object leaves the frame
    or becomes occluded.
    """
    
    # Every N frames we sample for tracking (balance between accuracy and speed)
    SAMPLE_INTERVAL = 2  # Track every 2nd frame
    SAFETY_BUFFER_SEC = 0.3  # Padding on start/end
    MAX_CONSECUTIVE_FAILURES = 5
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cancelled = False
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"MotionTracker: {self.total_frames} frames @ {self.fps:.1f} fps")
    
    def _create_tracker(self):
        """Create a tracker using available OpenCV API"""
        # Try different APIs for CSRT tracker
        try:
            # OpenCV 4.5.1+ with contrib
            return cv2.TrackerCSRT_create()
        except AttributeError:
            pass
        
        try:
            # OpenCV with legacy tracking module
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            pass
        
        try:
            # Try KCF as fallback (faster but less accurate)
            return cv2.TrackerKCF_create()
        except AttributeError:
            pass
        
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            pass
        
        # Last resort: use MOSSE (very fast, less accurate)
        try:
            return cv2.legacy.TrackerMOSSE_create()
        except AttributeError:
            logger.error("No compatible tracker found in OpenCV")
            return None
    
    def _normalize_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        """Convert pixel bbox to normalized (0-1) coordinates"""
        x, y, w, h = bbox
        return (
            x / self.width,
            y / self.height,
            w / self.width,
            h / self.height
        )
    
    def _denormalize_bbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates"""
        x, y, w, h = bbox
        return (
            int(x * self.width),
            int(y * self.height),
            int(w * self.width),
            int(h * self.height)
        )
    
    def track_motion(
        self,
        bbox: Tuple[float, float, float, float],  # Normalized
        start_frame: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> MotionTrackResult:
        """
        Track a region as it moves through the video.
        
        Returns positions for each frame where the object is visible,
        allowing the blur to follow the content as it moves.
        """
        self._cancelled = False
        pixel_bbox = self._denormalize_bbox(bbox)
        
        logger.info(f"Motion tracking: bbox={pixel_bbox}, start={start_frame}")
        
        frame_positions: Dict[int, Tuple[float, float, float, float]] = {}
        
        # Store initial position
        frame_positions[start_frame] = bbox
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Track forward
        if progress_callback:
            progress_callback(0, "Tracking forward motion...")
        
        forward_positions = self._track_direction(
            cap, pixel_bbox, start_frame, direction=1,
            progress_callback=progress_callback, progress_offset=0
        )
        frame_positions.update(forward_positions)
        
        # Track backward
        if progress_callback:
            progress_callback(50, "Tracking backward motion...")
        
        backward_positions = self._track_direction(
            cap, pixel_bbox, start_frame, direction=-1,
            progress_callback=progress_callback, progress_offset=50
        )
        frame_positions.update(backward_positions)
        
        cap.release()
        
        if not frame_positions:
            frame_positions[start_frame] = bbox
        
        # Find frame range
        all_frames = sorted(frame_positions.keys())
        first_frame = all_frames[0]
        last_frame = all_frames[-1]
        
        # Only apply safety buffer if we successfully tracked motion (multiple positions)
        # Otherwise keep the original selection time
        if len(frame_positions) > 3:  # More than just start + a couple fail positions
            buffer_frames = int(self.SAFETY_BUFFER_SEC * self.fps)
            first_frame = max(0, first_frame - buffer_frames)
            last_frame = min(self.total_frames - 1, last_frame + buffer_frames)
            logger.info(f"Applied {self.SAFETY_BUFFER_SEC}s buffer, {len(frame_positions)} positions tracked")
        else:
            # Keep original selection time - use start_frame as the base
            first_frame = start_frame
            last_frame = self.total_frames - 1  # Blur to end of video
            logger.info(f"Tracking had few positions ({len(frame_positions)}), using static blur from selection time")
        
        if progress_callback:
            progress_callback(100, "Motion tracking complete!")
        
        result = MotionTrackResult(
            start_frame=first_frame,
            end_frame=last_frame,
            start_time=first_frame / self.fps,
            end_time=last_frame / self.fps,
            confidence=0.9,
            frame_positions=frame_positions
        )
        
        logger.info(f"Motion tracking complete: {len(frame_positions)} positions, "
                    f"{result.start_time:.2f}s - {result.end_time:.2f}s")
        
        return result
    
    def _track_direction(
        self,
        cap,
        pixel_bbox: Tuple[int, int, int, int],
        start_frame: int,
        direction: int,  # 1 = forward, -1 = backward
        progress_callback: Optional[Callable[[float, str], None]] = None,
        progress_offset: float = 0
    ) -> Dict[int, Tuple[float, float, float, float]]:
        """
        Track in one direction, returning positions for each sampled frame.
        Uses CSRT tracker which handles motion well.
        """
        positions: Dict[int, Tuple[float, float, float, float]] = {}
        
        # Seek to start and initialize tracker
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame {start_frame}")
            return positions
        
        # Create tracker
        tracker = self._create_tracker()
        if tracker is None:
            logger.warning("No tracker available")
            return positions
        
        # Initialize tracker - convert tuple to proper format (x, y, w, h)
        x, y, w, h = pixel_bbox
        # Ensure valid bbox dimensions
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid bbox dimensions: {pixel_bbox}")
            return positions
        
        logger.info(f"Initializing tracker with bbox: ({x}, {y}, {w}, {h})")
        
        try:
            success = tracker.init(frame, (x, y, w, h))
        except Exception as e:
            logger.warning(f"Tracker init exception: {e}")
            success = False
        
        if not success:
            logger.warning(f"Failed to initialize tracker with bbox {pixel_bbox}")
            # Store at least the starting position
            positions[start_frame] = self._normalize_bbox(pixel_bbox)
            return positions
        
        current_bbox = pixel_bbox
        consecutive_failures = 0
        
        # Determine frame range
        if direction == 1:
            frame_range = range(start_frame + self.SAMPLE_INTERVAL, self.total_frames, self.SAMPLE_INTERVAL)
        else:
            frame_range = range(start_frame - self.SAMPLE_INTERVAL, -1, -self.SAMPLE_INTERVAL)
        
        total_frames = len(list(frame_range))
        frame_range = list(frame_range)  # Re-create since we exhausted it
        
        for i, target_frame in enumerate(frame_range):
            if self._cancelled:
                break
            
            # For backward tracking, we need to re-init tracker at each frame
            # because CSRT doesn't support backward tracking directly
            if direction == -1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Re-init tracker at previous known position
                tracker = self._create_tracker()
                tracker.init(frame, current_bbox)
                
                # Read next frame and update
                ret, next_frame = cap.read()
                if ret:
                    success, new_bbox = tracker.update(next_frame)
                else:
                    success = False
            else:
                # Forward: read sequential frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap.read()
                if not ret:
                    break
                
                success, new_bbox = tracker.update(frame)
            
            if success and self._is_valid_bbox(new_bbox, frame.shape):
                current_bbox = tuple(int(v) for v in new_bbox)
                normalized = self._normalize_bbox(current_bbox)
                positions[target_frame] = normalized
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    direction_str = "forward" if direction == 1 else "backward"
                    logger.info(f"Lost object at frame {target_frame} ({direction_str})")
                    break
            
            if progress_callback and i % 10 == 0:
                progress = progress_offset + 50 * (i / max(total_frames, 1))
                direction_str = "→" if direction == 1 else "←"
                progress_callback(progress, f"Tracking {direction_str} frame {target_frame}")
        
        return positions
    
    def _is_valid_bbox(self, bbox: Tuple, frame_shape: Tuple) -> bool:
        """Check if tracked bbox is valid"""
        x, y, w, h = bbox
        
        # Check bounds
        if x < 0 or y < 0 or w <= 10 or h <= 10:
            return False
        if x + w > frame_shape[1] or y + h > frame_shape[0]:
            return False
        
        return True
    
    def cancel(self):
        self._cancelled = True


def track_motion_in_video(
    video_path: str,
    bbox: Tuple[float, float, float, float],
    start_frame: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> MotionTrackResult:
    """Convenience function for motion tracking"""
    tracker = MotionTracker(video_path)
    return tracker.track_motion(bbox, start_frame, progress_callback)
