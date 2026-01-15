"""
ScreenSafe Optimized Video Exporter

Performance-optimized version with:
- Parallel OCR processing using thread pool
- OCR result caching
- Smart frame skipping (only OCR on keyframes)
- Optional FFmpeg output for GPU acceleration
"""

import cv2
import numpy as np
import logging
import time
import re
import subprocess
import shutil
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .models import Detection, VideoInfo, BoundingBox

logger = logging.getLogger(__name__)

# Blur parameters
BLUR_KERNEL = (51, 51)
BLUR_SIGMA = 30

# Cache directory for OCR results - use user's cache directory for proper isolation
def _get_cache_dir() -> Path:
    """Get the appropriate user cache directory for OCR results."""
    try:
        # Try platformdirs for cross-platform user cache dir
        from platformdirs import user_cache_dir
        cache_base = Path(user_cache_dir("ScreenSafe", "ScreenSafe"))
    except ImportError:
        # Fallback: use system temp directory with app subfolder
        import tempfile
        cache_base = Path(tempfile.gettempdir()) / "ScreenSafe"
    
    cache_dir = cache_base / "ocr_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

CACHE_DIR = _get_cache_dir()

# Regex patterns for automatic PII detection
REGEX_PATTERNS = {
    'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    'credit_card': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
    'phone': re.compile(r'\b(?:\+?1[-.\\s]?)?(?:\(?\d{3}\)?[-.\\s]?)?\d{3}[-.\\s]?\d{4}\b'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'ipv4': re.compile(r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'),
}


def _normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy matching to handle OCR errors."""
    text = text.lower().strip()
    text = text.replace(' ', '').replace('.', '').replace('_', '').replace('-', '')
    return text


def _get_video_hash(video_path: str) -> str:
    """Get a hash of the video file for cache keying."""
    path = Path(video_path)
    stat = path.stat()
    # Use file path, size, and mtime for a quick hash
    key = f"{path.absolute()}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _check_ffmpeg_nvenc() -> bool:
    """Check if FFmpeg with NVENC is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        has_nvenc = 'h264_nvenc' in result.stdout
        logger.info(f"FFmpeg NVENC check: {'available' if has_nvenc else 'not available'}")
        return has_nvenc
    except Exception as e:
        logger.warning(f"FFmpeg NVENC check failed: {e}")
        return False


def _check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
        return True
    else:
        # Try common Windows FFmpeg locations
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\tools\ffmpeg\bin\ffmpeg.exe',
        ]
        for path in common_paths:
            if Path(path).exists():
                logger.info(f"FFmpeg found at: {path}")
                return True
        logger.warning("FFmpeg not found in PATH or common locations")
        return False


@dataclass
class OCRResult:
    """Cached OCR result for a frame."""
    frame_number: int
    text_regions: List[Dict]  # List of {text, bbox, confidence}


class OCRCache:
    """Cache for OCR results to avoid recomputation."""
    
    def __init__(self, video_path: str):
        self.video_hash = _get_video_hash(video_path)
        CACHE_DIR.mkdir(exist_ok=True)
        self._cache_file = CACHE_DIR / f"{self.video_hash}.json"
        self._cache: Dict[int, List[Dict]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk if exists."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r') as f:
                    data = json.load(f)
                self._cache = {int(k): v for k, v in data.items()}
                logger.info(f"Loaded OCR cache with {len(self._cache)} frames")
            except Exception as e:
                logger.warning(f"Failed to load OCR cache: {e}")
                self._cache = {}
    
    def save(self):
        """Save cache to disk."""
        try:
            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
            logger.info(f"Saved OCR cache with {len(self._cache)} frames")
        except Exception as e:
            logger.warning(f"Failed to save OCR cache: {e}")
    
    def get(self, frame_number: int) -> Optional[List[Dict]]:
        """Get cached OCR result for a frame."""
        return self._cache.get(frame_number)
    
    def set(self, frame_number: int, results: List[Dict]):
        """Cache OCR results for a frame."""
        self._cache[frame_number] = results
    
    def clear(self):
        """Clear the cache."""
        self._cache = {}
        if self._cache_file.exists():
            self._cache_file.unlink()


class ParallelOCRProcessor:
    """Process OCR in parallel using thread pool."""
    
    def __init__(self, max_workers: int = 4, use_gpu: bool = True):
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self._reader = None
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def _get_reader(self):
        """Lazy init OCR reader."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(['en'], gpu=self.use_gpu)
                logger.info(f"Initialized EasyOCR reader (GPU: {self.use_gpu})")
            except Exception as e:
                logger.error(f"Failed to init OCR reader: {e}")
                raise
        return self._reader
    
    def process_frame(self, frame: np.ndarray, frame_number: int, scale: float = 1.0) -> Tuple[int, List[Dict]]:
        """Process a single frame and return OCR results."""
        reader = self._get_reader()
        
        try:
            # EasyOCR expects RGB
            rgb_frame = frame[:, :, ::-1]  # BGR to RGB
            
            # Apply scaling
            if scale != 1.0 and scale > 0:
                h, w = rgb_frame.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            results = reader.readtext(rgb_frame)
            text_regions = []
            
            # Use dimensions of the PROCESSED frame for normalization
            proc_h, proc_w = rgb_frame.shape[:2]
            
            for (bbox_corners, text, conf) in results:
                if conf > 0.3:
                    x_coords = [p[0] for p in bbox_corners]
                    y_coords = [p[1] for p in bbox_corners]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    text_regions.append({
                        'text': text.lower().strip(),
                        'original_text': text,
                        'bbox': {
                            'x': x1 / proc_w,
                            'y': y1 / proc_h,
                            'width': (x2 - x1) / proc_w,
                            'height': (y2 - y1) / proc_h,
                        },
                        'confidence': conf
                    })
            
            return (frame_number, text_regions)
        
        except Exception as e:
            logger.warning(f"OCR failed for frame {frame_number}: {e}")
            return (frame_number, [])
    
    def process_batch(self, frames: List[Tuple[int, np.ndarray]]) -> Dict[int, List[Dict]]:
        """Process a batch of frames in parallel.
        
        Note: EasyOCR isn't fully thread-safe, so we process sequentially
        but could batch GPU operations for better throughput.
        """
        results = {}
        for frame_number, frame in frames:
            fn, text_regions = self.process_frame(frame, frame_number)
            results[fn] = text_regions
        return results
    
    def shutdown(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class OptimizedVideoExporter:
    """
    Performance-optimized video exporter with:
    - Parallel OCR processing
    - OCR result caching
    - Smart frame skipping
    - Optional FFmpeg GPU output
    """
    
    def __init__(self, video_path: str, video_info: VideoInfo):
        self.video_path = video_path
        self.video_info = video_info
        self._cancelled = False
        self._use_ffmpeg = _check_ffmpeg_available()
        self._use_nvenc = _check_ffmpeg_nvenc() if self._use_ffmpeg else False
        self._preview_mode = False  # Half-resolution preview mode
        
        if self._use_nvenc:
            logger.info("GPU acceleration available (NVENC)")
        elif self._use_ffmpeg:
            logger.info("FFmpeg available (CPU encoding)")
        else:
            logger.info("Using OpenCV fallback for encoding")
    
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
               blur_strength: int = 51,
               use_cache: bool = True,
               use_gpu: bool = True,
               codec: str = 'h264',
               quality: str = 'high') -> bool:
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
            use_cache: Whether to use OCR result caching
            use_gpu: Whether to use GPU for OCR (if available)
            
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
        self._prev_frame = None
        self._codec = codec
        self._quality = quality
        self._enable_regex_patterns = False  # Export uses pre-defined detections
        
        # Initialize OCR cache
        self._cache = OCRCache(self.video_path) if use_cache else None
        
        # Initialize parallel OCR processor
        self._ocr_processor = ParallelOCRProcessor(max_workers=2, use_gpu=use_gpu)
        
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Build frame -> detections lookup
        frame_detections = self._build_frame_map(detections)
        
        # Collection for OCR-detected blur regions (to return to frontend)
        ocr_detected_regions = []
        
        # Open source video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {self.video_path}")
            return False
        
        # Prepare output writer
        if self._use_ffmpeg:
            out = self._create_ffmpeg_writer(output_path, codec, quality)
        else:
            out = self._create_opencv_writer(output_path)
        
        if out is None:
            cap.release()
            return False
        
        start_time = time.time()
        frame_number = 0
        total_frames = self.video_info.total_frames
        ocr_frames_processed = 0
        cache_hits = 0
        
        logger.info(f"Exporting video: {total_frames} frames (scan interval: {self._scan_interval})")
        
        try:
            while True:
                if self._cancelled:
                    logger.info("Export cancelled")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get actual timestamp from video (accurate for VFR videos)
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert ms to seconds
                
                # Check if we're in a scan zone
                in_scan_zone = self._is_in_scan_zone(current_time)
                
                # Motion detection
                is_scrolling = self._detect_motion(frame)
                
                # Get detections active in this frame
                active_dets = self._get_active_detections(
                    frame_detections, detections, frame_number, current_time
                )
                
                # Apply static blurs
                if active_dets:
                    frame = self._apply_blurs(frame, active_dets, blur_strength, frame_number)
                
                # NOTE: OCR detection removed from export - all detection happens during Scan
                # Export now only applies blurs from Detection objects passed from frontend
                # This respects the isRedacted flag and makes export significantly faster
                
                # Write frame
                self._write_frame(out, frame)
                frame_number += 1
                
                # Progress update
                if frame_number % 30 == 0 and progress_callback:
                    progress = (frame_number / total_frames) * 100
                    elapsed = time.time() - start_time
                    fps_rate = frame_number / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_number) / fps_rate if fps_rate > 0 else 0
                    
                    cache_rate = (cache_hits / max(1, ocr_frames_processed + cache_hits)) * 100 if cache_hits > 0 else 0
                    
                    progress_callback(
                        progress, 
                        f"Frame {frame_number}/{total_frames} ({fps_rate:.1f} fps, ETA: {eta:.0f}s)"
                    )
        
        finally:
            cap.release()
            self._close_writer(out)
            self._ocr_processor.shutdown()
            
            # Save cache
            if self._cache:
                self._cache.save()
        
        if self._cancelled:
            try:
                Path(output_path).unlink()
            except:
                pass
            return {'success': False, 'detected_blurs': []}
        
        elapsed = time.time() - start_time
        logger.info(f"Export complete in {elapsed:.1f}s (OCR: {ocr_frames_processed} frames, cache hits: {cache_hits})")
        logger.info(f"Collected {len(ocr_detected_regions)} raw OCR-detected blur regions")
        
        # Consolidate consecutive detections at same position into time ranges
        consolidated = self._consolidate_detections(ocr_detected_regions)
        logger.info(f"Consolidated to {len(consolidated)} blur region(s)")
        
        if progress_callback:
            progress_callback(100, "Export complete!")
        
        return {'success': True, 'detected_blurs': consolidated}
    
    def _create_opencv_writer(self, output_path: str) -> Optional[cv2.VideoWriter]:
        """Create OpenCV video writer."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Calculate output dimensions (half for preview mode)
        out_width = self.video_info.width // 2 if self._preview_mode else self.video_info.width
        out_height = self.video_info.height // 2 if self._preview_mode else self.video_info.height
        out = cv2.VideoWriter(
            output_path, 
            fourcc, 
            self.video_info.fps, 
            (out_width, out_height)
        )
        
        if not out.isOpened():
            logger.error(f"Cannot create output video: {output_path}")
            return None
        
        return out
    
    def _create_ffmpeg_writer(self, output_path: str, codec: str = 'h264', quality: str = 'high') -> Optional[subprocess.Popen]:
        """Create FFmpeg writer process with optional GPU acceleration.
        
        Args:
            output_path: Output video path
            codec: Video codec - 'h264', 'h265', or 'vp9'
            quality: Quality preset - 'low', 'medium', or 'high'
        """
        # Map codec and GPU availability to encoder
        encoder_map = {
            'h264': ('h264_nvenc', 'libx264'),
            'h265': ('hevc_nvenc', 'libx265'),
            'vp9': ('libvpx-vp9', 'libvpx-vp9'),  # VP9 has no NVENC
        }
        
        # Quality to CRF/preset mapping (lower CRF = better quality)
        quality_map = {
            'low': ('ultrafast', '28'),
            'medium': ('fast', '23'),
            'high': ('slow', '18'),
        }
        
        hw_encoder, sw_encoder = encoder_map.get(codec, ('h264_nvenc', 'libx264'))
        preset, crf = quality_map.get(quality, ('fast', '23'))
        
        # Use GPU encoder if available for h264/h265
        if self._use_nvenc and codec in ('h264', 'h265'):
            encoder = hw_encoder
            extra_args = ['-preset', 'fast', '-rc', 'vbr', '-cq', crf]
        else:
            encoder = sw_encoder
            extra_args = ['-preset', preset, '-crf', crf]
        
        # Calculate output dimensions (half for preview mode)
        out_width = self.video_info.width // 2 if self._preview_mode else self.video_info.width
        out_height = self.video_info.height // 2 if self._preview_mode else self.video_info.height
        
        logger.info(f"FFmpeg output dimensions: {out_width}x{out_height} (preview={self._preview_mode})")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{out_width}x{out_height}',
            '-r', str(self.video_info.fps),
            '-i', '-',
            '-c:v', encoder,
            *extra_args,
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"Using FFmpeg encoder: {encoder}")
            return process
        except Exception as e:
            logger.warning(f"Failed to start FFmpeg, falling back to OpenCV: {e}")
            return self._create_opencv_writer(output_path)
    
    def _write_frame(self, writer, frame: np.ndarray):
        """Write frame to video writer."""
        # Resize for preview mode (half resolution)
        if self._preview_mode:
            new_width = self.video_info.width // 2
            new_height = self.video_info.height // 2
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Debug log on first frame
        if not hasattr(self, '_logged_frame_info'):
            self._logged_frame_info = True
            logger.info(f"Writing frame: shape={frame.shape}, expected=({frame.shape[0]}, {frame.shape[1]}, 3)")
        
        if isinstance(writer, subprocess.Popen):
            writer.stdin.write(frame.tobytes())
        else:
            writer.write(frame)
    
    def _close_writer(self, writer):
        """Close the video writer."""
        if isinstance(writer, subprocess.Popen):
            writer.stdin.close()
            writer.wait()
        elif writer:
            writer.release()
    
    def _is_in_scan_zone(self, current_time: float) -> bool:
        """Check if current time is within any scan zone."""
        if not self._scan_zones:
            return True
        return any(
            z.get('start', 0) <= current_time <= z.get('end', float('inf'))
            for z in self._scan_zones
        )
    
    def _detect_motion(self, frame: np.ndarray) -> bool:
        """Detect significant motion between frames."""
        if self._prev_frame is None:
            self._prev_frame = frame.copy()
            return False
        
        prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        
        self._prev_frame = frame.copy()
        return mean_diff > self._motion_threshold
    
    def _build_frame_map(self, detections: List[Detection]) -> Dict[int, List[Detection]]:
        """Deprecated: Frame-map approach causes sync issues with VFR videos."""
        return {}
    
    def _get_active_detections(self, frame_map, detections, frame_number, current_time) -> List[Detection]:
        """Get all detections active at the current timestamp (VFR safe)."""
        active_dets = []
        
        # Use simple epsilon for float comparison safety
        epsilon = 0.05  # 50ms tolerance
        
        for det in detections:
            if not det.is_redacted:
                continue
                
            # STRICT timestamp check - ignore frame numbers completely for VFR correctness
            # We use a small epsilon to avoid floating point misses at boundaries
            if (det.start_time - epsilon) <= current_time <= (det.end_time + epsilon):
                active_dets.append(det)
        
        return active_dets
    
    def _apply_blurs(self, frame: np.ndarray, detections: List[Detection],
                     blur_strength: int, frame_number: int = 0) -> np.ndarray:
        """Apply obfuscation (blur or blackout) to detected regions."""
        height, width = frame.shape[:2]
        blur_kernel = (blur_strength, blur_strength)
        
        for det in detections:
            # Use interpolated bbox if frame_positions is available (motion tracking)
            interpolated_bbox = self._get_bbox_at_frame(det, frame_number)
            if interpolated_bbox is not None:
                bbox = interpolated_bbox
            else:
                bbox = det.bbox
            
            x1 = max(0, int(bbox.x * width))
            y1 = max(0, int(bbox.y * height))
            x2 = min(width, int((bbox.x + bbox.width) * width))
            y2 = min(height, int((bbox.y + bbox.height) * height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            try:
                # Handle Blackout vs Blur
                # Check for "blackout" type (using string check to match frontend/JSON)
                if getattr(det, 'type', '') == 'blackout' or det.type == 'blackout':
                    # Draw solid black rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                else:
                    # Apply Gaussian Blur
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        blurred = cv2.GaussianBlur(roi, blur_kernel, BLUR_SIGMA)
                        frame[y1:y2, x1:x2] = blurred
            except Exception as e:
                logger.warning(f"Failed to apply obfuscation to region: {e}")
        
        return frame
    
    def _apply_anchor_blurs(self, frame: np.ndarray, blur_boxes: List[BoundingBox],
                            blur_strength: int) -> np.ndarray:
        """Apply Gaussian blur to anchor-detected regions."""
        height, width = frame.shape[:2]
        blur_kernel = (blur_strength, blur_strength)
        
        for bbox in blur_boxes:
            x1 = max(0, int(bbox.x * width))
            y1 = max(0, int(bbox.y * height))
            x2 = min(width, int((bbox.x + bbox.width) * width))
            y2 = min(height, int((bbox.y + bbox.height) * height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            try:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    blurred = cv2.GaussianBlur(roi, blur_kernel, BLUR_SIGMA)
                    frame[y1:y2, x1:x2] = blurred
            except Exception as e:
                logger.warning(f"Failed to blur anchor region: {e}")
        
        return frame
    
    def _get_bbox_at_frame(self, det: Detection, frame_number: int) -> Optional[BoundingBox]:
        """Get interpolated bounding box position for a specific frame."""
        if not hasattr(det, 'frame_positions') or not det.frame_positions:
            return None
        
        positions = det.frame_positions
        if not positions:
            return None
        
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
            return BoundingBox(x=next_pos[1], y=next_pos[2], width=next_pos[3], height=next_pos[4])
        
        if next_pos is None:
            return BoundingBox(x=prev_pos[1], y=prev_pos[2], width=prev_pos[3], height=prev_pos[4])
        
        if prev_pos[0] == next_pos[0]:
            return BoundingBox(x=prev_pos[1], y=prev_pos[2], width=prev_pos[3], height=prev_pos[4])
        
        t = (frame_number - prev_pos[0]) / (next_pos[0] - prev_pos[0])
        return BoundingBox(
            x=prev_pos[1] + t * (next_pos[1] - prev_pos[1]),
            y=prev_pos[2] + t * (next_pos[2] - prev_pos[2]),
            width=prev_pos[3] + t * (next_pos[3] - prev_pos[3]),
            height=prev_pos[4] + t * (next_pos[4] - prev_pos[4])
        )
    
    def _find_anchor_blur_boxes(self, ocr_results: List[Dict]) -> List[BoundingBox]:
        """Find blur box positions based on anchor text detection."""
        if not self._anchors or not ocr_results:
            return []
        
        blur_boxes = []
        
        # Build text -> bbox lookup
        text_bboxes = {}
        for region in ocr_results:
            text_lower = region['text']
            if text_lower not in text_bboxes:
                text_bboxes[text_lower] = []
            text_bboxes[text_lower].append(region['bbox'])
        
        for anchor in self._anchors:
            label = anchor.get('label', '').lower().strip()
            direction = anchor.get('direction', 'BELOW')
            gap = anchor.get('gap', 0) / 1000
            blur_width = anchor.get('width', 100) / self.video_info.width
            blur_height = anchor.get('height', 30) / self.video_info.height
            
            # Find matching text
            matching_bboxes = []
            for text, bbox_list in text_bboxes.items():
                if '/' in text or '&' in text or len(text) > 50:
                    continue
                
                if text == label or (label in text and len(text) < 30) or (text in label and len(text) >= 3):
                    matching_bboxes.extend(bbox_list)
            
            for anchor_bbox in matching_bboxes:
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
                
                # Clamp bbox to stay within normalized 0-1 range
                blur_x = max(0.0, min(1.0, blur_x))
                blur_y = max(0.0, min(1.0, blur_y))
                # Ensure width/height don't extend beyond frame boundary
                blur_width = min(blur_width, 1.0 - blur_x)
                blur_height = min(blur_height, 1.0 - blur_y)
                # Skip if resulting box is too small
                if blur_width <= 0 or blur_height <= 0:
                    continue
                
                blur_boxes.append((
                    BoundingBox(
                        x=blur_x,
                        y=blur_y,
                        width=blur_width,
                        height=blur_height
                    ),
                    anchor.get('label', 'anchor')  # Include trigger keyword
                ))
        
        return blur_boxes  # List of (BoundingBox, anchor_label) tuples
    
    def _find_watch_list_matches(self, ocr_results: List[Dict]) -> List[BoundingBox]:
        """Find text matching watch list items or regex patterns."""
        if not self._watch_list or not ocr_results:
            return []
        
        blur_boxes = []  # List of (BoundingBox, trigger_keyword) tuples
        
        for region in ocr_results:
            text = region['original_text']
            text_lower = region['text']
            bbox = region['bbox']
            matched_keyword = None
            
            # Check against watch list
            text_normalized = _normalize_for_matching(text_lower)
            for watch_item in self._watch_list:
                watch_normalized = _normalize_for_matching(watch_item)
                if len(watch_normalized) >= 4 and watch_normalized in text_normalized:
                    matched_keyword = watch_item  # Store the original watch item
                    break
            
            # Check regex patterns (only if enabled)
            if not matched_keyword and self._enable_regex_patterns:
                for pattern_name, pattern in REGEX_PATTERNS.items():
                    if pattern.search(text):
                        matched_keyword = f"regex:{pattern_name}"  # e.g., "regex:email"
                        break
            
            if matched_keyword:
                # Clamp bbox values to 0-1 range
                bx = max(0.0, min(1.0, bbox['x']))
                by = max(0.0, min(1.0, bbox['y']))
                bw = min(bbox['width'], 1.0 - bx)
                bh = min(bbox['height'], 1.0 - by)
                if bw > 0 and bh > 0:
                    blur_boxes.append((
                        BoundingBox(x=bx, y=by, width=bw, height=bh),
                        matched_keyword
                    ))
        
        return blur_boxes  # List of (BoundingBox, keyword) tuples
    
    def _consolidate_detections(self, detections: List[Dict]) -> List[Dict]:
        """Consolidate consecutive detections at same position into time ranges.
        
        Instead of returning 100+ individual frame detections, merge detections
        that have the same type, similar position, AND consecutive frames into 
        single detections with start_time and end_time.
        
        Key fix: If there's a gap in frame numbers (anchor disappeared), start
        a new detection group instead of continuing the old one.
        """
        if not detections:
            return []
        
        # Maximum gap between frames to consider them "consecutive"
        # Smart Sampling can skip up to 10 intervals (scan_interval * 10)
        # So we need a gap large enough to bridge those static periods.
        # Set to 12x scan_interval to be safe.
        max_frame_gap = self._scan_interval * 12
        
        # Group by type (anchor/watchlist)
        from collections import defaultdict
        by_type = defaultdict(list)
        for d in detections:
            by_type[d['type']].append(d)
        
        consolidated = []
        
        for det_type, items in by_type.items():
            # Sort by frame number
            items.sort(key=lambda x: x['frame'])
            
            # Active tracks: list of {'items': [], 'last_frame': int, 'last_bbox': dict}
            tracks = []
            
            for item in items:
                bbox = item['bbox']
                frame = item['frame']
                source = item['source']
                
                # Find best matching track
                best_track_idx = -1
                best_dist = 1.0  # Max distance to consider (normalized)
                
                for i, track in enumerate(tracks):
                    last_item = track['items'][-1]
                    
                    # 1. Source check (Must match exactly)
                    if source != last_item['source']:
                        continue
                        
                    # 2. Consecutive check (Must be within max gap AND different frame)
                    frame_diff = frame - track['last_frame']
                    if frame_diff <= 0 or frame_diff > max_frame_gap:
                        continue
                        
                    # 3. Distance check (Euclidean distance of centers)
                    # Use TIGHT threshold (0.05 = 5% screen) to avoid merging different anchor instances
                    cx1 = bbox['x'] + bbox['width']/2
                    cy1 = bbox['y'] + bbox['height']/2
                    cx2 = track['last_bbox']['x'] + track['last_bbox']['width']/2
                    cy2 = track['last_bbox']['y'] + track['last_bbox']['height']/2
                    
                    dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5
                    
                    if dist < 0.05 and dist < best_dist:
                        best_dist = dist
                        best_track_idx = i
                
                if best_track_idx >= 0:
                    # Append to existing track
                    tracks[best_track_idx]['items'].append(item)
                    tracks[best_track_idx]['last_frame'] = frame
                    tracks[best_track_idx]['last_bbox'] = bbox
                else:
                    # Start new track
                    tracks.append({
                        'items': [item],
                        'last_frame': frame,
                        'last_bbox': bbox
                    })
            
            # Convert tracks to consolidated detections
            for track in tracks:
                track_items = track['items']
                first = track_items[0]
                last = track_items[-1]
                
                # Create frame_positions list [frame, x, y, w, h]
                frame_positions = []
                for item in track_items:
                    b = item['bbox']
                    frame_positions.append([
                        item['frame'], b['x'], b['y'], b['width'], b['height']
                    ])
                
                consolidated.append({
                    'frame': first['frame'],
                    'time': first['time'],
                    'end_time': last['time'] + 1/30,
                    'bbox': first['bbox'],  # Start position
                    'type': det_type,
                    'source': first['source'],
                    'occurrence_count': len(track_items),
                    'frame_positions': frame_positions  # Enable frontend tracking
                })
                
        return consolidated
        
        return consolidated
    
    def cancel(self) -> None:
        """Cancel ongoing export."""
        self._cancelled = True


def export_video_optimized(video_path: str,
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
                           blur_strength: int = 51,
                           use_cache: bool = True,
                           use_gpu: bool = True,
                           codec: str = 'h264',
                           quality: str = 'high') -> bool:
    """
    Convenience function to export a redacted video using optimized exporter.
    """
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
    
    exporter = OptimizedVideoExporter(video_path, video_info)
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
        blur_strength,
        use_cache,
        use_gpu,
        codec,
        quality
    )


def preview_export_video(video_path: str,
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
    Generate a low-resolution preview video for quick review.
    Uses 1/4 resolution and ultrafast encoding for speed.
    """
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
    
    # Use optimized exporter in preview mode (half resolution, faster encoding)
    exporter = OptimizedVideoExporter(video_path, video_info)
    exporter._preview_mode = True  # Enable half-resolution output
    
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
        blur_strength,
        use_cache=True,
        use_gpu=True,
        codec='h264',  # H.264 for fastest encoding
        quality='low'  # Low quality = faster
    )


def scan_video(video_path: str,
               anchors: List[Dict] = None,
               watch_list: List[str] = None,
               scan_interval: int = 30,
               motion_threshold: float = 30.0,
               ocr_scale: float = 1.0,
               scan_zones: List[Dict] = None,
               video_info: VideoInfo = None,
               progress_callback: Callable[[float, str], None] = None,
               use_cache: bool = True,
               use_gpu: bool = True,
               enable_regex_patterns: bool = False) -> Dict:
    """
    Scan video for blur regions using OCR WITHOUT encoding any video.
    
    This is much faster than preview export since it only runs OCR analysis
    and returns the detected blur regions for overlay display on the original video.
    
    Args:
        video_path: Path to source video
        anchors: Anchor definitions for relative blur positioning
        watch_list: List of text strings to blur
        scan_interval: Process every Nth frame for OCR
        motion_threshold: Threshold for motion detection
        ocr_scale: Scale factor for OCR processing
        scan_zones: Time ranges to scan
        video_info: Pre-computed video metadata
        progress_callback: Progress update function
        use_cache: Whether to use OCR cache
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Dict with 'success' and 'detected_blurs' containing consolidated blur regions
    """
    logger.info(f"Starting video scan: {video_path}")
    
    # Get video info if not provided
    if video_info is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return {'success': False, 'detected_blurs': []}
        video_info = VideoInfo(
            path=video_path,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        )
        cap.release()
    
    # Create scanner (reuses exporter's OCR logic)
    scanner = OptimizedVideoExporter(video_path, video_info)
    scanner._anchors = anchors or []
    scanner._watch_list = [w.lower() for w in (watch_list or [])]
    scanner._scan_interval = scan_interval
    scanner._motion_threshold = motion_threshold
    scanner._ocr_scale = ocr_scale
    scanner._scan_zones = scan_zones or []
    scanner._prev_frame = None
    scanner._enable_regex_patterns = enable_regex_patterns
    
    # Debug: log anchor configurations and video dimensions
    logger.info(f"Video dimensions: {video_info.width}x{video_info.height}")
    for i, anchor in enumerate(scanner._anchors):
        logger.info(f"Anchor {i}: label='{anchor.get('label')}', direction={anchor.get('direction')}, "
                    f"width={anchor.get('width')}, height={anchor.get('height')}, gap={anchor.get('gap')}")
        # Calculated normalized values:
        w = anchor.get('width', 100) / video_info.width
        h = anchor.get('height', 30) / video_info.height
        logger.info(f"  -> Normalized: width={w:.4f}, height={h:.4f}")
    
    # Initialize OCR cache and processor
    scanner._cache = OCRCache(video_path) if use_cache else None
    scanner._ocr_processor = ParallelOCRProcessor(max_workers=2, use_gpu=use_gpu)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {'success': False, 'detected_blurs': []}
    
    start_time = time.time()
    frame_number = 0
    total_frames = video_info.total_frames
    ocr_frames_processed = 0
    cache_hits = 0
    ocr_detected_regions = []
    
    logger.info(f"Scanning video: {total_frames} frames (scan interval: {scan_interval})")
    
    # Initial progress update so UI shows something immediately
    if progress_callback:
        progress_callback(0, f"Starting scan of {total_frames} frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get actual timestamp from video (accurate for VFR videos)
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert ms to seconds
            
            # Check if we're in a scan zone
            is_in_zone = scanner._is_in_scan_zone(current_time)
            
            # Motion detection is now integrated into Adaptive Scanning
            
            # --- Adaptive Scanning Logic ---
            should_scan = False
            
            # Default check: is it time to scan?
            is_interval_frame = (frame_number % scan_interval == 0)
            
            if is_in_zone:
                if is_interval_frame:
                    should_scan = True
                
                # Apply Smart Sampling to skip truly static interval frames
                # But ALWAYS scan at least every 3rd interval for reliability
                is_forced_interval = (frame_number % (scan_interval * 3) == 0)
                
                if scanner._prev_frame is not None:
                    # Downscale for fast comparison
                    curr_small = cv2.resize(frame, (64, 64))
                    prev_small = cv2.resize(scanner._prev_frame, (64, 64))
                    
                    # Calculate visual difference
                    diff = cv2.absdiff(curr_small, prev_small)
                    mean_diff = np.mean(diff) / 255.0
                    
                    if mean_diff < 0.005 and not is_forced_interval:
                        # Truly static (< 0.5% change) - skip unless forced interval
                        # This threshold catches most typing but skips static pauses
                        if is_interval_frame:
                            should_scan = False
                            # logger.debug(f"Smart skip: frame {frame_number} static (diff={mean_diff:.4f})")
                    
                    elif mean_diff > 0.05:
                        # Motion Boost - add extra scans during movement
                        if frame_number % 2 == 0:
                            logger.info(f"Motion Boost triggered at frame {frame_number} (diff={mean_diff:.2f})")
                            should_scan = True
            
            scanner._prev_frame = frame.copy()
            
            if not should_scan:
                # logger.debug(f"Skipping frame {frame_number} (static)")
                pass
            
            if should_scan:
                # Check cache first
                ocr_results = None
                if scanner._cache:
                    ocr_results = scanner._cache.get(frame_number)
                    if ocr_results is not None:
                        cache_hits += 1
                
                if ocr_results is None:
                    # Run OCR (with scaling)
                    ocr_frames_processed += 1
                    _, ocr_results = scanner._ocr_processor.process_frame(
                        frame, 
                        frame_number,
                        scale=ocr_scale
                    )
                    
                    # Cache the results
                    if scanner._cache:
                        scanner._cache.set(frame_number, ocr_results)
                
                # Find blur boxes
                anchor_boxes = scanner._find_anchor_blur_boxes(ocr_results)
                watch_boxes = scanner._find_watch_list_matches(ocr_results)
                
                # Collect detected regions - now with trigger keywords
                for box, trigger in (anchor_boxes or []):
                    bbox_dict = {
                        'x': box.x,
                        'y': box.y,
                        'width': box.width,
                        'height': box.height
                    }
                    ocr_detected_regions.append({
                        'frame': frame_number,
                        'time': current_time,
                        'bbox': bbox_dict,
                        'type': 'anchor',
                        'source': trigger  # e.g., "Email", "CARD NUMBER"
                    })
                for box, trigger in (watch_boxes or []):
                    bbox_dict = {
                        'x': box.x,
                        'y': box.y,
                        'width': box.width,
                        'height': box.height
                    }
                    ocr_detected_regions.append({
                        'frame': frame_number,
                        'time': current_time,
                        'bbox': bbox_dict,
                        'type': 'watchlist',
                        'source': trigger  # e.g., the matched watchlist word
                    })
            
            frame_number += 1
            
            # Progress update - every 30 frames for responsive feedback
            if frame_number % 30 == 0 and progress_callback:
                progress = (frame_number / total_frames) * 100
                elapsed = time.time() - start_time
                fps_rate = frame_number / elapsed if elapsed > 0 else 0
                eta = (total_frames - frame_number) / fps_rate if fps_rate > 0 else 0
                progress_callback(progress, f"Scanning frame {frame_number}/{total_frames} ({fps_rate:.1f} fps, ETA: {eta:.0f}s)")
    
    finally:
        cap.release()
        scanner._ocr_processor.shutdown()
        
        # Save cache
        if scanner._cache:
            scanner._cache.save()
    
    elapsed = time.time() - start_time
    logger.info(f"Scan complete in {elapsed:.1f}s (OCR: {ocr_frames_processed} frames, cache hits: {cache_hits})")
    logger.info(f"Collected {len(ocr_detected_regions)} raw blur regions")
    
    # Debug: show all unique sources found in raw detections
    raw_sources = set(r['source'] for r in ocr_detected_regions)
    logger.info(f"Raw detection sources: {raw_sources}")
    
    # Consolidate consecutive detections at same position
    consolidated = scanner._consolidate_detections(ocr_detected_regions)
    logger.info(f"Consolidated to {len(consolidated)} blur region(s)")
    
    # Debug: show sources in consolidated output
    consolidated_sources = set(c['source'] for c in consolidated)
    logger.info(f"Consolidated sources: {consolidated_sources}")
    
    if progress_callback:
        progress_callback(100, "Scan complete!")
    
    return {'success': True, 'detected_blurs': consolidated}

