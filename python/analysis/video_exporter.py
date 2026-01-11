"""
ScreenSafe AI Analysis Pipeline

Video Exporter - Renders redacted video using OpenCV with blur overlays.
For production, can be replaced with FFmpeg for better performance.
"""

import cv2
import numpy as np
import logging
import time
import re
from pathlib import Path
from typing import Callable, Optional, List, Dict

from .models import Detection, VideoInfo, BoundingBox

logger = logging.getLogger(__name__)

# Blur parameters
BLUR_KERNEL = (51, 51)
BLUR_SIGMA = 30

# Regex patterns for automatic PII detection
REGEX_PATTERNS = {
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'credit_card': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    'phone': re.compile(r'\b(?:\+?1[-.\\s]?)?(?:\(?\d{3}\)?[-.\\s]?)?\d{3}[-.\\s]?\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'ipv4': re.compile(r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
}

def _normalize_for_matching(text: str) -> str:
    """
    Normalize text for fuzzy matching to handle OCR errors.
    Removes common OCR artifacts like spaces, dots, underscores.
    """
    # Lowercase and strip
    text = text.lower().strip()
    # Remove common OCR artifacts that cause matching failures
    # Keep @ and + as they're important for email matching
    text = text.replace(' ', '').replace('.', '').replace('_', '').replace('-', '')
    return text



class VideoExporter:
    """
    Export video with blur overlays applied to detected regions.
    
    Uses OpenCV for frame-by-frame processing with Gaussian blur.
    Detections are tracked across frames for smooth blur application.
    """
    
    def __init__(self, video_path: str, video_info: VideoInfo):
        """
        Initialize exporter.
        
        Args:
            video_path: Path to source video
            video_info: Video metadata
        """
        self.video_path = video_path
        self.video_info = video_info
        self._cancelled = False
    
    def export(self,
               output_path: str,
               detections: List[Detection],
               anchors: List[Dict] = None,
               watch_list: List[str] = None,
               scan_interval: int = 90,
               motion_threshold: float = 30.0,
               ocr_scale: float = 1.0,
               scan_zones: List[Dict] = None,
               progress_callback: Optional[Callable[[float, str], None]] = None,
               blur_strength: int = 51) -> bool:
        """
        Export video with blurs applied.
        
        Args:
            output_path: Path for output video
            detections: List of detections to blur
            anchors: List of anchor definitions for dynamic blur positioning
            watch_list: List of text strings to detect and blur
            scan_interval: Frames between OCR scans (default 90)
            motion_threshold: Motion threshold for scroll detection (default 30)
            ocr_scale: Scale factor for OCR (default 1.0)
            scan_zones: List of {start, end} time ranges to process (None = all)
            progress_callback: Called with (progress_percent, message)
            blur_strength: Kernel size for Gaussian blur (must be odd)
            
        Returns:
            True if successful
        """
        self._cancelled = False
        self._anchors = anchors or []
        self._watch_list = [w.lower() for w in (watch_list or [])]
        self._scan_interval = scan_interval
        self._motion_threshold = motion_threshold
        self._ocr_scale = ocr_scale
        self._scan_zones = scan_zones or []
        self._ocr_reader = None  # Lazy init OCR for anchor detection
        self._prev_frame = None  # For motion detection
        
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Build frame -> detections lookup for fast access
        frame_detections = self._build_frame_map(detections)
        
        # Open source video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return False
        
        # Prepare output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.video_info.fps, 
            (self.video_info.width, self.video_info.height)
        )
        
        if not out.isOpened():
            logger.error(f"Cannot create output video: {output_path}")
            cap.release()
            return False
        
        start_time = time.time()
        frame_number = 0
        total_frames = self.video_info.total_frames
        
        logger.info(f"Exporting video: {total_frames} frames (scan interval: {self._scan_interval})")
        
        try:
            while True:
                if self._cancelled:
                    logger.info("Export cancelled")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = frame_number / self.video_info.fps
                
                # Check if we're in a scan zone (if zones defined)
                in_scan_zone = True
                if self._scan_zones:
                    in_scan_zone = any(
                        z.get('start', 0) <= current_time <= z.get('end', float('inf'))
                        for z in self._scan_zones
                    )
                
                # Motion detection
                is_scrolling = False
                if self._prev_frame is not None:
                    is_scrolling = self._detect_motion(self._prev_frame, frame)
                    if is_scrolling:
                        logger.debug(f"Motion detected at frame {frame_number}, skipping OCR")
                self._prev_frame = frame.copy()
                
                # Get detections active in this frame
                active_dets = frame_detections.get(frame_number, [])
                
                # Also check detections that span this frame
                for det in detections:
                    if det.is_redacted and det.start_time <= current_time <= det.end_time:
                        if det not in active_dets:
                            active_dets.append(det)
                
                # Apply blurs (with frame number for motion tracking)
                if active_dets:
                    frame = self._apply_blurs(
                        frame, active_dets, blur_strength, frame_number
                    )
                
                # Only run OCR-based detection if in scan zone and not scrolling
                if in_scan_zone and not is_scrolling:
                    # Find and apply anchor-based blur boxes at configurable interval
                    if (self._anchors or self._watch_list) and frame_number % self._scan_interval == 0:
                        logger.info(f"Running OCR at frame {frame_number}...")
                        
                        # Anchor detection
                        if self._anchors:
                            anchor_boxes = self._find_anchor_blur_boxes(frame)
                            self._last_anchor_boxes = anchor_boxes
                        
                        # Watch list + regex pattern detection
                        if self._watch_list:
                            watch_boxes = self._find_watch_list_matches(frame)
                            self._last_watch_boxes = watch_boxes
                        
                        total_found = len(getattr(self, '_last_anchor_boxes', [])) + len(getattr(self, '_last_watch_boxes', []))
                        logger.info(f"Found {total_found} blur boxes")
                
                # When scrolling, apply full-frame blur overlay to prevent PII leakage
                # (cached positions become stale when content moves)
                if is_scrolling and (hasattr(self, '_last_anchor_boxes') or hasattr(self, '_last_watch_boxes')):
                    # Apply stronger blur overlay in regions where we had detections
                    # This is safer than using stale position data
                    h, w = frame.shape[:2]
                    frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
                else:
                    # Apply cached anchor blur boxes (when not scrolling)
                    if hasattr(self, '_last_anchor_boxes') and self._last_anchor_boxes:
                        frame = self._apply_anchor_blurs(frame, self._last_anchor_boxes, blur_strength)
                    
                    # Apply cached watch list blur boxes (when not scrolling)
                    if hasattr(self, '_last_watch_boxes') and self._last_watch_boxes:
                        frame = self._apply_anchor_blurs(frame, self._last_watch_boxes, blur_strength)
                
                out.write(frame)
                frame_number += 1
                
                # Progress update every 30 frames
                if frame_number % 30 == 0 and progress_callback:
                    progress = (frame_number / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_rate = frame_number / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_number) / fps_rate if fps_rate > 0 else 0
                    zone_status = " [in zone]" if in_scan_zone else " [skip zone]"
                    scroll_status = " [scroll]" if is_scrolling else ""
                    progress_callback(
                        progress, 
                        f"Frame {frame_number}/{total_frames} ({fps_rate:.1f} fps, ETA: {eta:.0f}s){zone_status}{scroll_status}"
                    )
        
        finally:
            cap.release()
            out.release()
        
        if self._cancelled:
            # Clean up partial output
            try:
                Path(output_path).unlink()
            except:
                pass
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"Export complete in {elapsed:.1f}s: {output_path}")
        
        if progress_callback:
            progress_callback(100, "Export complete!")
        
        return True
    
    def _build_frame_map(self, detections: List[Detection]) -> Dict[int, List[Detection]]:
        """Build a map of frame_number -> detections for fast lookup"""
        frame_map: Dict[int, List[Detection]] = {}
        
        for det in detections:
            if not det.is_redacted:
                continue
            
            for frame in range(det.start_frame, det.end_frame + 1):
                if frame not in frame_map:
                    frame_map[frame] = []
                frame_map[frame].append(det)
        
        return frame_map
    
    def _apply_blurs(self, 
                     frame: np.ndarray, 
                     detections: List[Detection],
                     blur_strength: int,
                     frame_number: int = 0) -> np.ndarray:
        """Apply Gaussian blur to detected regions, with motion tracking support"""
        height, width = frame.shape[:2]
        blur_kernel = (blur_strength, blur_strength)
        
        for det in detections:
            # Check for motion-tracked positions
            bbox = self._get_bbox_at_frame(det, frame_number)
            if bbox is None:
                bbox = det.bbox
            
            # Convert normalized bbox to pixels
            x1 = int(bbox.x * width)
            y1 = int(bbox.y * height)
            x2 = int((bbox.x + bbox.width) * width)
            y2 = int((bbox.y + bbox.height) * height)
            
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Skip invalid regions
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract ROI and blur
            try:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, blur_kernel, BLUR_SIGMA)
                    frame[y1:y2, x1:x2] = blurred
            except Exception as e:
                logger.warning(f"Failed to blur region ({x1},{y1},{x2},{y2}): {e}")
        
        return frame
    
    def _apply_anchor_blurs(self, 
                            frame: np.ndarray, 
                            blur_boxes: List[BoundingBox],
                            blur_strength: int) -> np.ndarray:
        """Apply Gaussian blur to anchor-detected regions"""
        height, width = frame.shape[:2]
        blur_kernel = (blur_strength, blur_strength)
        
        for bbox in blur_boxes:
            # Convert normalized to pixel coords
            x1 = int(bbox.x * width)
            y1 = int(bbox.y * height)
            x2 = int((bbox.x + bbox.width) * width)
            y2 = int((bbox.y + bbox.height) * height)
            
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            try:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, blur_kernel, BLUR_SIGMA)
                    frame[y1:y2, x1:x2] = blurred
            except Exception as e:
                logger.warning(f"Failed to blur anchor region ({x1},{y1},{x2},{y2}): {e}")
        
        return frame
    
    def _get_bbox_at_frame(self, det: Detection, frame_number: int) -> Optional[BoundingBox]:
        """Get interpolated bounding box position for a specific frame"""
        if not hasattr(det, 'frame_positions') or not det.frame_positions:
            return None
        
        positions = det.frame_positions  # List of [frame, x, y, w, h]
        if not positions:
            return None
        
        # Find surrounding keyframes
        prev_pos = None
        next_pos = None
        
        for pos in sorted(positions, key=lambda p: p[0]):
            if pos[0] <= frame_number:
                prev_pos = pos
            if pos[0] >= frame_number and next_pos is None:
                next_pos = pos
        
        if prev_pos is None and next_pos is None:
            return None
        
        if prev_pos is None:
            # Before first keyframe - use first
            return BoundingBox(x=next_pos[1], y=next_pos[2], width=next_pos[3], height=next_pos[4])
        
        if next_pos is None:
            # After last keyframe - use last
            return BoundingBox(x=prev_pos[1], y=prev_pos[2], width=prev_pos[3], height=prev_pos[4])
        
        if prev_pos[0] == next_pos[0]:
            # Exact frame match
            return BoundingBox(x=prev_pos[1], y=prev_pos[2], width=prev_pos[3], height=prev_pos[4])
        
        # Linear interpolation between keyframes
        t = (frame_number - prev_pos[0]) / (next_pos[0] - prev_pos[0])
        return BoundingBox(
            x=prev_pos[1] + t * (next_pos[1] - prev_pos[1]),
            y=prev_pos[2] + t * (next_pos[2] - prev_pos[2]),
            width=prev_pos[3] + t * (next_pos[3] - prev_pos[3]),
            height=prev_pos[4] + t * (next_pos[4] - prev_pos[4])
        )
    
    def _find_anchor_blur_boxes(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Find blur box positions based on anchor text detection.
        
        For each anchor, uses OCR to find the anchor text in the frame,
        then calculates blur box position based on direction/gap/size.
        
        Returns:
            List of BoundingBox objects for detected anchor blur regions
        """
        if not self._anchors:
            return []
        
        # Lazy init OCR reader
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], gpu=True)
            except Exception as e:
                logger.warning(f"Failed to init OCR reader: {e}")
                return []
        
        height, width = frame.shape[:2]
        blur_boxes = []
        
        # Run OCR on full frame
        try:
            results = self._ocr_reader.readtext(frame)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []
        
        # Build text -> bbox LIST (multiple instances per text)
        text_bboxes = {}  # text -> list of bboxes
        for (bbox_corners, text, conf) in results:
            if conf > 0.3:
                # Convert corner format to x,y,w,h
                x_coords = [p[0] for p in bbox_corners]
                y_coords = [p[1] for p in bbox_corners]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                text_lower = text.lower().strip()
                
                bbox_info = {
                    'x': x1 / width,
                    'y': y1 / height,
                    'width': (x2 - x1) / width,
                    'height': (y2 - y1) / height,
                    'original_text': text
                }
                
                # Store as list to support multiple instances
                if text_lower not in text_bboxes:
                    text_bboxes[text_lower] = []
                text_bboxes[text_lower].append(bbox_info)
        
        # Log all detected text for debugging
        logger.info(f"OCR detected {len(text_bboxes)} unique text regions: {list(text_bboxes.keys())}")
        
        # Find each anchor and calculate blur boxes for ALL matching positions
        for anchor in self._anchors:
            label = anchor.get('label', '').lower().strip()
            direction = anchor.get('direction', 'BELOW')
            gap = anchor.get('gap', 0) / 1000  # Convert back to normalized
            blur_width = anchor.get('width', 100) / width
            blur_height = anchor.get('height', 30) / height
            
            # Collect ALL matching bboxes, filtering URLs
            matching_bboxes = []
            matched_text = None
            
            for text, bbox_list in text_bboxes.items():
                # Skip URLs and long strings (likely not labels)
                if '/' in text or '&' in text or len(text) > 50:
                    continue
                
                is_match = False
                priority = 99
                
                # Exact match - highest priority
                if text == label:
                    is_match = True
                    priority = 0
                # Label is contained in short text (e.g., "email" in "email address")
                elif label in text and len(text) < 30:
                    is_match = True
                    priority = 1
                # Text is contained in label (e.g., "pass" detected, label is "password")  
                elif text in label and len(text) >= 3:
                    is_match = True
                    priority = 2
                
                if is_match:
                    # Add ALL bboxes for this matching text
                    for bbox in bbox_list:
                        matching_bboxes.append((priority, text, bbox))
            
            # Sort by priority then by y position (top to bottom)
            matching_bboxes.sort(key=lambda x: (x[0], x[2]['y']))
            
            if not matching_bboxes:
                logger.warning(f"Anchor '{label}' NOT found in OCR results")
                continue
            
            logger.info(f"Anchor '{label}' found {len(matching_bboxes)} matches")
            
            # Apply blur box for EACH matching position
            for priority, matched_text, anchor_bbox in matching_bboxes:
                # Calculate blur box position based on direction
                if direction == 'BELOW':
                    blur_x = anchor_bbox['x']
                    blur_y = anchor_bbox['y'] + anchor_bbox['height'] + gap
                elif direction == 'ABOVE':
                    blur_x = anchor_bbox['x']
                    blur_y = anchor_bbox['y'] - blur_height - gap
                elif direction == 'RIGHT':
                    blur_x = anchor_bbox['x'] + anchor_bbox['width'] + gap
                    blur_y = anchor_bbox['y']
                elif direction == 'LEFT':
                    blur_x = anchor_bbox['x'] - blur_width - gap
                    blur_y = anchor_bbox['y']
                else:
                    continue
                
                blur_boxes.append(BoundingBox(
                    x=max(0, blur_x),
                    y=max(0, blur_y),
                    width=blur_width,
                    height=blur_height
                ))
                
                logger.info(f"  -> '{matched_text}' at ({anchor_bbox['x']:.3f}, {anchor_bbox['y']:.3f}) -> blur at ({blur_x:.3f}, {blur_y:.3f})")
        
        return blur_boxes
    
    def _detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """
        Detect significant motion between frames (e.g., scrolling).
        
        Args:
            prev_frame: Previous video frame
            curr_frame: Current video frame
            
        Returns:
            True if motion exceeds threshold (scrolling detected)
        """
        # Convert to grayscale for comparison
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        
        return mean_diff > self._motion_threshold
    
    def _find_watch_list_matches(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Find text matching watch list items or regex patterns.
        
        Returns:
            List of BoundingBox objects for detected text regions
        """
        if not self._watch_list:
            return []
        
        # Lazy init OCR reader
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], gpu=True)
            except Exception as e:
                logger.warning(f"Failed to init OCR reader: {e}")
                return []
        
        height, width = frame.shape[:2]
        blur_boxes = []
        
        # Run OCR on frame
        try:
            results = self._ocr_reader.readtext(frame)
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return []
        
        for (bbox_corners, text, conf) in results:
            if conf < 0.3:
                continue
                
            text_lower = text.lower().strip()
            should_blur = False
            match_reason = None
            
            # Check against watch list using normalized matching
            # This handles OCR errors like missing dots (gmail.com -> gmailcom)
            text_normalized = _normalize_for_matching(text_lower)
            for watch_item in self._watch_list:
                watch_normalized = _normalize_for_matching(watch_item)
                # Only match if:
                # 1. Normalized watch item is found in normalized OCR text
                # 2. Minimum length to avoid single-character false positives
                if len(watch_normalized) >= 4 and watch_normalized in text_normalized:
                    should_blur = True
                    match_reason = f"watch:{watch_item}"
                    break
            
            # Check against regex patterns
            if not should_blur:
                for pattern_name, pattern in REGEX_PATTERNS.items():
                    if pattern.search(text):
                        should_blur = True
                        match_reason = f"regex:{pattern_name}"
                        break
            
            if should_blur:
                # Convert corner format to normalized coords
                x_coords = [p[0] for p in bbox_corners]
                y_coords = [p[1] for p in bbox_corners]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)
                
                blur_boxes.append(BoundingBox(
                    x=x1 / width,
                    y=y1 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height
                ))
                
                logger.info(f"Watch list match '{text[:20]}...' ({match_reason})")
        
        return blur_boxes
    
    def cancel(self) -> None:
        """Cancel ongoing export"""
        self._cancelled = True


def export_video(video_path: str,
                 output_path: str,
                 detections: List[Detection],
                 anchors: List[Dict] = None,
                 watch_list: List[str] = None,
                 scan_interval: int = 90,
                 motion_threshold: float = 30.0,
                 ocr_scale: float = 1.0,
                 scan_zones: List[Dict] = None,
                 video_info: VideoInfo = None,
                 progress_callback: Callable[[float, str], None] = None,
                 blur_strength: int = 51) -> bool:
    """
    Convenience function to export a redacted video.
    
    Args:
        video_path: Source video path
        output_path: Output video path
        detections: List of detections to blur
        anchors: List of anchor definitions for dynamic blur positioning
        watch_list: List of text strings to detect and blur
        scan_interval: Frames between OCR scans
        motion_threshold: Motion threshold for scroll detection
        ocr_scale: Scale factor for OCR
        scan_zones: List of time ranges to process
        video_info: Video metadata (will be extracted if not provided)
        progress_callback: Progress updates
        blur_strength: Blur kernel size
        
    Returns:
        True if successful
    """
    # Get video info if not provided
    if video_info is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        video_info = VideoInfo(
            path=video_path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        )
        cap.release()
    
    exporter = VideoExporter(video_path, video_info)
    return exporter.export(
        output_path, 
        detections, 
        anchors,
        watch_list,
        scan_interval,
        motion_threshold,
        ocr_scale,
        scan_zones,
        progress_callback, 
        blur_strength
    )
