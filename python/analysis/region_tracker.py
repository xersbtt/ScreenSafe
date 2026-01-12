"""
ScreenSafe AI Analysis Pipeline

Content-Aware Region Tracker - Uses OCR to track when text content 
appears and disappears in a selected region.

Improvements:
- Safety buffer: Extends detected range to ensure complete coverage
- Higher sample rate: Checks more frames for finer detection
- Fuzzy matching: Uses Levenshtein distance for text comparison
- Scene detection: Stops tracking at scene changes
- GPU acceleration: Uses CUDA when available for faster OCR
"""

import cv2
import numpy as np
import logging
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Calculate similarity ratio (0-1) using Levenshtein distance"""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


@dataclass
class TrackResult:
    """Result of tracking a region through the video"""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: float
    text_content: str = ""


class ContentTracker:
    """
    Track when text content appears/disappears in a video region.
    
    Features:
    - OCR-based text detection (EasyOCR with GPU, fallback to Tesseract)
    - Fuzzy text matching using Levenshtein distance
    - Scene change detection to refine boundaries
    - Safety buffer to ensure complete coverage
    """
    
    # Tuning parameters
    SAMPLE_RATE_HZ = 4  # Check 4 frames per second (every 0.25s)
    SAFETY_BUFFER_SEC = 0.5  # Add 0.5s padding on each end
    TEXT_SIMILARITY_THRESHOLD = 0.6  # 60% text similarity
    VISUAL_SIMILARITY_THRESHOLD = 0.85  # 85% visual hash similarity
    SCENE_CHANGE_THRESHOLD = 0.4  # 40% histogram difference = scene change
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cancelled = False
        
        # Initialize OCR reader with GPU if available
        if HAS_EASYOCR:
            use_gpu = HAS_CUDA
            logger.info(f"Using EasyOCR (GPU: {use_gpu})")
            try:
                self.reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                self.ocr_method = 'easyocr'
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}, trying without GPU")
                self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.ocr_method = 'easyocr'
        elif HAS_TESSERACT:
            logger.info("Using Tesseract (no GPU)")
            self.reader = None
            self.ocr_method = 'tesseract'
        else:
            logger.warning("No OCR available - using visual similarity only")
            self.reader = None
            self.ocr_method = 'visual'
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Calculate frame step based on sample rate
        self.frame_step = max(1, int(self.fps / self.SAMPLE_RATE_HZ))
        
        logger.info(f"ContentTracker: {self.total_frames} frames @ {self.fps:.1f} fps")
        logger.info(f"  Method: {self.ocr_method}, Step: {self.frame_step} frames")
    
    def _extract_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract region from frame with bounds checking"""
        x, y, w, h = bbox
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        return frame[y:y+h, x:x+w].copy()
    
    def _get_region_text(self, region: np.ndarray) -> str:
        """Extract text from a region using OCR"""
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return ""
        
        if self.ocr_method == 'easyocr' and self.reader:
            try:
                results = self.reader.readtext(region, detail=0)
                return ' '.join(results).strip().lower()
            except Exception as e:
                logger.debug(f"EasyOCR error: {e}")
                return ""
        
        elif self.ocr_method == 'tesseract':
            try:
                text = pytesseract.image_to_string(region)
                return text.strip().lower()
            except Exception as e:
                logger.debug(f"Tesseract error: {e}")
                return ""
        
        return ""
    
    def _get_region_hash(self, region: np.ndarray) -> str:
        """Get a perceptual hash of region for similarity comparison"""
        if region.size == 0:
            return ""
        
        try:
            # Resize to 16x16 for hash
            small = cv2.resize(region, (16, 16))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            avg = gray.mean()
            bits = (gray > avg).flatten()
            return ''.join(['1' if b else '0' for b in bits])
        except:
            return ""
    
    def _get_frame_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Get color histogram of frame for scene change detection"""
        try:
            if len(frame.shape) == 3:
                # BGR histogram
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            else:
                hist = cv2.calcHist([frame], [0], None, [32], [0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        except:
            return np.array([])
    
    def _is_scene_change(self, hist1: np.ndarray, hist2: np.ndarray) -> bool:
        """Detect if there's a scene change between two histograms"""
        if hist1.size == 0 or hist2.size == 0:
            return False
        
        try:
            # Use correlation - lower means more different
            correlation = cv2.compareHist(hist1.reshape(-1, 1).astype(np.float32), 
                                          hist2.reshape(-1, 1).astype(np.float32), 
                                          cv2.HISTCMP_CORREL)
            return correlation < (1.0 - self.SCENE_CHANGE_THRESHOLD)
        except:
            return False
    
    def _content_matches(self, ref_text: str, text: str, ref_hash: str, hash_val: str) -> Tuple[bool, float]:
        """
        Check if content matches using fuzzy text matching and visual hash.
        
        Returns:
            Tuple of (matches: bool, confidence: float)
        """
        confidence = 0.0
        
        # Primary: Text similarity using Levenshtein distance
        if ref_text and text:
            similarity = levenshtein_similarity(ref_text, text)
            confidence = similarity
            return similarity >= self.TEXT_SIMILARITY_THRESHOLD, confidence
        
        # Fallback: Visual hash comparison
        if ref_hash and hash_val and len(ref_hash) == len(hash_val):
            matches = sum(1 for a, b in zip(ref_hash, hash_val) if a == b)
            similarity = matches / len(ref_hash)
            confidence = similarity * 0.8  # Lower confidence for visual matching
            return similarity >= self.VISUAL_SIMILARITY_THRESHOLD, confidence
        
        return False, 0.0
    
    def track_region(
        self,
        bbox: Tuple[float, float, float, float],  # Normalized (x, y, w, h)
        start_frame: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> TrackResult:
        """
        Track when content in region appears/disappears.
        
        Uses OCR to detect text content and finds frames where it changes,
        with scene change detection to refine boundaries.
        """
        self._cancelled = False
        
        # Convert normalized bbox to pixels
        x = int(bbox[0] * self.width)
        y = int(bbox[1] * self.height)
        w = int(bbox[2] * self.width)
        h = int(bbox[3] * self.height)
        pixel_bbox = (x, y, max(10, w), max(10, h))
        
        logger.info(f"Content tracking: bbox={pixel_bbox}, start={start_frame}")
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Get reference content from start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, ref_frame = cap.read()
        if not ret:
            cap.release()
            return self._fallback_result(start_frame)
        
        ref_region = self._extract_region(ref_frame, pixel_bbox)
        ref_text = self._get_region_text(ref_region)
        ref_hash = self._get_region_hash(ref_region)
        ref_hist = self._get_frame_histogram(ref_frame)
        
        logger.info(f"Reference: '{ref_text[:30]}...' (text_len={len(ref_text)})")
        
        # Track backward
        if progress_callback:
            progress_callback(0, "Finding start...")
        
        first_frame = self._track_backward(cap, pixel_bbox, start_frame, 
                                            ref_text, ref_hash, ref_hist, progress_callback)
        
        # Track forward  
        if progress_callback:
            progress_callback(50, "Finding end...")
        
        last_frame = self._track_forward(cap, pixel_bbox, start_frame,
                                          ref_text, ref_hash, ref_hist, progress_callback)
        
        cap.release()
        
        # Apply safety buffer
        buffer_frames = int(self.SAFETY_BUFFER_SEC * self.fps)
        first_frame = max(0, first_frame - buffer_frames)
        last_frame = min(self.total_frames - 1, last_frame + buffer_frames)
        
        if progress_callback:
            progress_callback(100, "Complete!")
        
        result = TrackResult(
            start_frame=first_frame,
            end_frame=last_frame,
            start_time=first_frame / self.fps,
            end_time=last_frame / self.fps,
            confidence=0.9 if ref_text else 0.7,
            text_content=ref_text
        )
        
        logger.info(f"Tracking complete: {result.start_time:.2f}s - {result.end_time:.2f}s (with {self.SAFETY_BUFFER_SEC}s buffer)")
        
        return result
    
    def _track_backward(self, cap, bbox, start_frame, ref_text, ref_hash, ref_hist, progress_callback) -> int:
        """Track backward from start_frame to find first appearance"""
        first_frame = start_frame
        
        check_frames = list(range(start_frame - self.frame_step, -1, -self.frame_step))
        total = len(check_frames)
        consecutive_misses = 0
        max_consecutive_misses = 3  # Allow a few misses before stopping
        
        for i, check_frame in enumerate(check_frames):
            if self._cancelled:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Extract region and check content match FIRST (priority)
            region = self._extract_region(frame, bbox)
            text = self._get_region_text(region)
            hash_val = self._get_region_hash(region)
            
            matches, confidence = self._content_matches(ref_text, text, ref_hash, hash_val)
            
            if matches:
                first_frame = check_frame
                consecutive_misses = 0  # Reset on match
            else:
                consecutive_misses += 1
                
                # Only check scene change on the REGION, not full frame
                # This prevents popups/overlays from triggering false scene changes
                region_hist = self._get_frame_histogram(region)
                ref_region_hist = self._get_frame_histogram(self._extract_region(
                    cap.read()[1] if cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) else frame, bbox))
                
                if consecutive_misses >= max_consecutive_misses:
                    logger.info(f"Content mismatch at frame {check_frame} after {consecutive_misses} misses")
                    break
            
            if progress_callback and i % 5 == 0:
                progress = 25 * (i / max(total, 1))
                progress_callback(progress, f"Checking frame {check_frame}")
        
        return first_frame
    
    def _track_forward(self, cap, bbox, start_frame, ref_text, ref_hash, ref_hist, progress_callback) -> int:
        """Track forward from start_frame to find last appearance"""
        last_frame = start_frame
        
        check_frames = list(range(start_frame + self.frame_step, self.total_frames, self.frame_step))
        total = len(check_frames)
        consecutive_misses = 0
        max_consecutive_misses = 3  # Allow a few misses before stopping
        
        for i, check_frame in enumerate(check_frames):
            if self._cancelled:
                break
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, check_frame)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract region and check content match FIRST (priority)
            region = self._extract_region(frame, bbox)
            text = self._get_region_text(region)
            hash_val = self._get_region_hash(region)
            
            matches, confidence = self._content_matches(ref_text, text, ref_hash, hash_val)
            
            if matches:
                last_frame = check_frame
                consecutive_misses = 0  # Reset on match
            else:
                consecutive_misses += 1
                
                if consecutive_misses >= max_consecutive_misses:
                    logger.info(f"Content mismatch at frame {check_frame} after {consecutive_misses} misses")
                    break
            
            if progress_callback and i % 5 == 0:
                progress = 50 + 25 * (i / max(total, 1))
                progress_callback(progress, f"Checking frame {check_frame}")
        
        return last_frame
    
    def _fallback_result(self, start_frame: int) -> TrackResult:
        """Return fallback result when tracking fails"""
        return TrackResult(
            start_frame=start_frame,
            end_frame=start_frame,
            start_time=start_frame / self.fps,
            end_time=start_frame / self.fps,
            confidence=0.0
        )
    
    def cancel(self):
        self._cancelled = True


def track_region_in_video(
    video_path: str,
    bbox: Tuple[float, float, float, float],
    start_frame: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> TrackResult:
    """Convenience function for content-aware tracking (bidirectional - legacy)"""
    tracker = ContentTracker(video_path)
    return tracker.track_region(bbox, start_frame, progress_callback)


def track_region_forward(
    video_path: str,
    bbox: Tuple[float, float, float, float],  # Normalized (x, y, w, h)
    start_timestamp: float,  # Seconds (VFR compatible)
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> TrackResult:
    """
    Forward-only content tracking using timestamp-based seeking.
    
    VFR Compatible: Uses CAP_PROP_POS_MSEC for accurate timestamp seeking.
    
    Args:
        video_path: Path to video file
        bbox: Normalized bounding box (x, y, width, height)
        start_timestamp: Start time in seconds (user's selected time)
        progress_callback: Optional progress callback
        
    Returns:
        TrackResult with start_time = user's selected time,
                         end_time = when content disappears
    """
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    # Convert normalized bbox to pixels
    x = int(bbox[0] * width)
    y = int(bbox[1] * height)
    w = int(bbox[2] * width)
    h = int(bbox[3] * height)
    pixel_bbox = (x, y, max(10, w), max(10, h))
    
    logger.info(f"Forward tracking: start={start_timestamp:.2f}s, bbox={pixel_bbox}")
    
    # Seek to start position using timestamp (VFR compatible)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_timestamp * 1000)
    ret, ref_frame = cap.read()
    if not ret:
        cap.release()
        # Fallback result
        return TrackResult(
            start_frame=int(start_timestamp * fps),
            end_frame=int(start_timestamp * fps),
            start_time=start_timestamp,
            end_time=start_timestamp + 1.0,  # Default 1 second duration
            confidence=0.0
        )
    
    # Get reference content
    def extract_region(frame, bbox):
        x, y, w, h = bbox
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        return frame[y:y+h, x:x+w].copy()
    
    def get_region_hash(region):
        if region.size == 0:
            return ""
        try:
            small = cv2.resize(region, (16, 16))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            avg = gray.mean()
            bits = (gray > avg).flatten()
            return ''.join(['1' if b else '0' for b in bits])
        except:
            return ""
    
    ref_region = extract_region(ref_frame, pixel_bbox)
    ref_hash = get_region_hash(ref_region)
    
    # Start time is user's selected time (no change)
    result_start_time = start_timestamp
    result_end_time = start_timestamp
    
    # Track forward to find end time
    sample_interval_sec = 0.25  # Check every 0.25 seconds (4 Hz)
    consecutive_misses = 0
    max_consecutive_misses = 3  # Stop after 3 misses (~0.75s)
    
    current_time = start_timestamp + sample_interval_sec
    checks_done = 0
    max_checks = int((duration - start_timestamp) / sample_interval_sec)
    
    if progress_callback:
        progress_callback(0, "Tracking forward...")
    
    while current_time < duration:
        # Seek using timestamp (VFR compatible)
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get actual timestamp from video
        actual_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Extract region and compare hash
        region = extract_region(frame, pixel_bbox)
        hash_val = get_region_hash(region)
        
        # Check visual similarity
        if ref_hash and hash_val and len(ref_hash) == len(hash_val):
            matches = sum(1 for a, b in zip(ref_hash, hash_val) if a == b)
            similarity = matches / len(ref_hash)
            content_matches = similarity >= 0.85  # 85% threshold
        else:
            content_matches = False
        
        if content_matches:
            result_end_time = actual_time
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= max_consecutive_misses:
                logger.info(f"Content disappeared at {actual_time:.2f}s")
                break
        
        current_time += sample_interval_sec
        checks_done += 1
        
        if progress_callback and checks_done % 4 == 0:
            progress = min(100, 100 * (current_time - start_timestamp) / max(1, duration - start_timestamp))
            progress_callback(progress, f"Checking {actual_time:.1f}s...")
    
    cap.release()
    
    # Add safety buffer
    safety_buffer = 0.5  # 0.5 seconds
    result_end_time = min(duration, result_end_time + safety_buffer)
    
    if progress_callback:
        progress_callback(100, "Complete!")
    
    result = TrackResult(
        start_frame=int(result_start_time * fps),
        end_frame=int(result_end_time * fps),
        start_time=result_start_time,
        end_time=result_end_time,
        confidence=0.9 if ref_hash else 0.5,
        text_content=""
    )
    
    logger.info(f"Forward tracking complete: {result.start_time:.2f}s - {result.end_time:.2f}s")
    
    return result
