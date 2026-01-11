"""
ScreenSafe PII Wizard Module

Port of proven pii_wizard_v20.py logic for use in the GUI.
Uses watch list + anchor approach instead of complex tracking.
"""

import cv2
import numpy as np
import re
import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (from pii_wizard_v20)
# =============================================================================

KEYBOARD_Y_THRESHOLD = 0.50
KEYBOARD_BLOCKER_HEIGHT = 0.40
RECT_MATCH_DISTANCE = 0.10
DIGIT_CLUSTER_RANGE = 50
QWERTY_MIN_HITS = 3

SCROLL_BLUR_KERNEL = (99, 99)
REGION_BLUR_KERNEL = (51, 51)
BLUR_SIGMA = 30

QWERTY_ROWS = [
    set("qwertyuiop"),
    set("asdfghjkl"),
    set("zxcvbnm")
]

DEFAULT_KB_TRIGGERS = [
    "go", "return", "enter", "done", "search", "next", "space", "send",
    "?123", "123", "abc", "english", "eng", "us", "uk", "shift", "delete",
    "backspace", "caps", "symbols"
]

# Precompiled regex patterns for common PII
REGEX_PATTERNS = [
    re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),  # Email
    re.compile(r'\b(?:\d[ -]*?){13,16}\b'),  # Credit cards
    re.compile(r'\b(?:\+?1[-.\\s]?)?(?:\(?\d{3}\)?[-.\\s]?)?\d{3}[-.\\s]?\d{4}\b'),  # Phone
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Anchor:
    """Configuration for an anchor-based blur zone."""
    label: str       # Text to find (e.g., "Card Number:")
    direction: str   # BELOW, RIGHT, ABOVE, LEFT
    gap: int         # Pixels from anchor
    width: int       # Blur box width
    height: int      # Blur box height


@dataclass
class ScanZone:
    """Time range to scan."""
    start_time: float  # seconds
    end_time: float    # seconds


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    watch_list: List[str]
    anchors: List[Anchor]
    enable_keyboard: bool = False
    kb_triggers: List[str] = None
    scan_zones: List[ScanZone] = None
    scan_interval: int = 15
    motion_threshold: float = 3.0
    ocr_scale: float = 0.5
    
    def __post_init__(self):
        if self.kb_triggers is None:
            self.kb_triggers = DEFAULT_KB_TRIGGERS.copy()
        if self.scan_zones is None:
            self.scan_zones = []


# =============================================================================
# OCR READER - SHARED ENGINE (prevents double VRAM usage)
# =============================================================================

def get_ocr_reader():
    """
    Get the shared EasyOCR reader from the central engine.
    This prevents loading the AI model into VRAM twice.
    """
    from .ocr_engine import get_ocr_engine
    
    # Get the singleton engine (use GPU if CUDA available)
    engine = get_ocr_engine(use_gpu=HAS_CUDA)
    
    # Ensure it's initialized
    if not engine._initialized:
        engine.initialize()
    
    # Return the raw EasyOCR reader object that pii_wizard expects
    return engine._reader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip()


def clean_for_match(text: str) -> str:
    """Normalize text for matching (remove spaces and hyphens)."""
    return text.lower().replace(" ", "").replace("-", "")


def is_in_zone(current_time: float, zones: List[ScanZone]) -> bool:
    """Check if current time is within any scan zone."""
    if not zones:
        return True  # No zones = scan everything
    return any(z.start_time <= current_time <= z.end_time for z in zones)


def detect_qwerty_keyboard(text_norm: str, qwerty_hits: int) -> int:
    """Detect QWERTY keyboard patterns."""
    if len(text_norm) == 1:
        for row in QWERTY_ROWS:
            if text_norm in row:
                return qwerty_hits + 1
    elif 2 <= len(text_norm) <= 10:
        text_chars = set(text_norm)
        for row in QWERTY_ROWS:
            if text_chars.issubset(row) and len(text_chars) >= 2:
                return qwerty_hits + 3
    return qwerty_hits


# =============================================================================
# FRAME ANALYSIS
# =============================================================================

def analyze_frame(
    frame: np.ndarray,
    config: ProcessingConfig,
    reader=None
) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    """
    Analyze a frame for PII content using OCR.
    
    Args:
        frame: Input video frame (BGR)
        config: Processing configuration with watch list, anchors, etc.
        reader: EasyOCR reader (uses singleton if None)
        
    Returns:
        Tuple of (list of blur rectangles, keyboard_detected flag)
    """
    if reader is None:
        reader = get_ocr_reader()
    
    rects: List[Tuple[int, int, int, int]] = []
    keyboard_detected = False
    
    orig_h, orig_w = frame.shape[:2]
    scale = config.ocr_scale
    
    # Downscale for faster OCR
    small_w, small_h = int(orig_w * scale), int(orig_h * scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    
    results = reader.readtext(small_frame)
    inv_scale = 1.0 / scale
    
    # For keyboard detection
    digit_y_coords: List[int] = []
    qwerty_hits = 0
    
    # Build anchor dict for faster lookup
    anchor_dict = {a.label.lower(): a for a in config.anchors}
    
    for (bbox, text, prob) in results:
        text_norm = clean_text(text)
        text_nospace = clean_for_match(text)
        
        # Scale bbox back to original size
        (tl, tr, br, bl) = bbox
        bx = max(0, min(int(tl[0] * inv_scale), orig_w - 1))
        by = max(0, min(int(tl[1] * inv_scale), orig_h - 1))
        bw = max(1, min(int(br[0] * inv_scale) - bx, orig_w - bx))
        bh = max(1, min(int(br[1] * inv_scale) - by, orig_h - by))
        centerY = by + (bh // 2)
        
        # KEYBOARD DETECTION
        if config.enable_keyboard and centerY > (orig_h * KEYBOARD_Y_THRESHOLD):
            for trig in config.kb_triggers:
                if trig in text_norm and len(text_norm) < (len(trig) + 4):
                    keyboard_detected = True
                    break
            
            if re.match(r'^\d$', text_norm):
                digit_y_coords.append(centerY)
            
            qwerty_hits = detect_qwerty_keyboard(text_norm, qwerty_hits)
        
        # ANCHOR MATCHING
        match_found = False
        for anchor_text, anchor in anchor_dict.items():
            if anchor_text in text_norm:
                if anchor.direction == "BELOW":
                    rects.append((bx, by + bh + anchor.gap, anchor.width, anchor.height))
                elif anchor.direction == "RIGHT":
                    rects.append((bx + bw + anchor.gap, by, anchor.width, anchor.height))
                elif anchor.direction == "ABOVE":
                    rects.append((bx, by - anchor.gap - anchor.height, anchor.width, anchor.height))
                elif anchor.direction == "LEFT":
                    rects.append((bx - anchor.gap - anchor.width, by, anchor.width, anchor.height))
                match_found = True
        
        # WATCH LIST MATCHING
        if not match_found:
            for secret in config.watch_list:
                if secret in text_norm or secret in text_nospace:
                    rects.append((bx, by, bw, bh))
                    match_found = True
                    break
        
        # REGEX PATTERN MATCHING
        if not match_found:
            for pattern in REGEX_PATTERNS:
                if pattern.search(text):
                    rects.append((bx, by, bw, bh))
                    break
    
    # Keyboard geometric analysis
    if config.enable_keyboard:
        if len(digit_y_coords) >= 4:
            y_range = max(digit_y_coords) - min(digit_y_coords)
            if y_range < DIGIT_CLUSTER_RANGE:
                keyboard_detected = True
        
        if qwerty_hits >= QWERTY_MIN_HITS:
            keyboard_detected = True
    
    return rects, keyboard_detected


def get_text_in_region(
    frame: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    reader=None
) -> str:
    """
    Get OCR text from a specific region of the frame.
    
    Args:
        frame: Input video frame (BGR)
        x, y, width, height: Normalized region (0-1)
        reader: EasyOCR reader (uses singleton if None)
        
    Returns:
        The detected text in the region, or empty string if none found.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if reader is None:
        reader = get_ocr_reader()
    
    orig_h, orig_w = frame.shape[:2]
    logger.info(f"get_text_in_region: frame={orig_w}x{orig_h}, region=({x:.3f},{y:.3f},{width:.3f},{height:.3f})")
    
    # Convert normalized to pixel coords with minimal padding
    # Keep padding small to stay close to user's selection
    min_size = 40  # Minimum crop size in pixels
    pad = 5  # pixels of padding around the region
    
    px = max(0, int(x * orig_w) - pad)
    py = max(0, int(y * orig_h) - pad)
    pw = min(int(width * orig_w) + pad * 2, orig_w - px)
    ph = min(int(height * orig_h) + pad * 2, orig_h - py)
    
    # Ensure minimum size (expand equally on both sides)
    if pw < min_size:
        extra = (min_size - pw) // 2
        px = max(0, px - extra)
        pw = min(min_size, orig_w - px)
    if ph < min_size:
        extra = (min_size - ph) // 2
        py = max(0, py - extra)
        ph = min(min_size, orig_h - py)
    
    logger.info(f"Cropping: pixel coords ({px},{py}) size {pw}x{ph}")
    
    if pw < 10 or ph < 10:
        logger.warning(f"Region too small: {pw}x{ph}")
        return ""
    
    # Crop the region
    cropped = frame[py:py+ph, px:px+pw]
    logger.info(f"Cropped region shape: {cropped.shape}")
    
    # Upscale small regions for better OCR
    if cropped.shape[0] < 100 or cropped.shape[1] < 200:
        scale = max(2, min(4, 200 // max(cropped.shape[1], 1)))
        new_w = cropped.shape[1] * scale
        new_h = cropped.shape[0] * scale
        cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.info(f"Upscaled to {new_w}x{new_h}")
    
    # OCR the cropped region
    results = reader.readtext(cropped)
    logger.info(f"OCR found {len(results)} results")
    
    # Log all results for debugging
    for i, (bbox, text, prob) in enumerate(results):
        logger.info(f"  OCR #{i+1}: '{text}' (prob={prob:.2f})")
    
    # Combine all detected text (lowered threshold from 0.3 to 0.1)
    texts = [text.strip() for (_, text, prob) in results if prob > 0.1]
    result = " ".join(texts)
    logger.info(f"Final result: '{result}'")
    return result


def get_text_at_position(
    frame: np.ndarray,
    click_x: float,
    click_y: float,
    ocr_scale: float = 0.75,  # Increased for better quality
    reader=None
) -> List[Dict[str, Any]]:
    """
    Get all OCR text regions in a frame, with the closest one to the click position first.
    
    Args:
        frame: Input video frame (BGR)
        click_x: Normalized X position (0-1)
        click_y: Normalized Y position (0-1)
        ocr_scale: Scale for OCR processing
        reader: EasyOCR reader (uses singleton if None)
        
    Returns:
        List of text regions sorted by distance to click, each with:
        - text: The detected text
        - bbox: {x, y, width, height} normalized
        - distance: Distance from click
    """
    if reader is None:
        reader = get_ocr_reader()
    
    orig_h, orig_w = frame.shape[:2]
    click_px = click_x * orig_w
    click_py = click_y * orig_h
    
    # Downscale for faster OCR
    small_w, small_h = int(orig_w * ocr_scale), int(orig_h * ocr_scale)
    small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    
    results = reader.readtext(small_frame)
    inv_scale = 1.0 / ocr_scale
    
    text_regions = []
    
    for (bbox, text, prob) in results:
        # Scale bbox back to original size
        (tl, tr, br, bl) = bbox
        bx = max(0, min(int(tl[0] * inv_scale), orig_w - 1))
        by = max(0, min(int(tl[1] * inv_scale), orig_h - 1))
        bw = max(1, min(int(br[0] * inv_scale) - bx, orig_w - bx))
        bh = max(1, min(int(br[1] * inv_scale) - by, orig_h - by))
        
        # Calculate center and distance from click
        center_x = bx + bw / 2
        center_y = by + bh / 2
        distance = np.sqrt((center_x - click_px)**2 + (center_y - click_py)**2)
        
        text_regions.append({
            "text": text.strip(),
            "bbox": {
                "x": bx / orig_w,
                "y": by / orig_h,
                "width": bw / orig_w,
                "height": bh / orig_h
            },
            "distance": distance,
            "confidence": prob
        })
    
    # Sort by distance to click
    text_regions.sort(key=lambda r: r["distance"])
    
    return text_regions


# =============================================================================
# INTERPOLATION
# =============================================================================

def interpolate_rects(
    start_rects: List[Tuple[int, int, int, int]],
    end_rects: List[Tuple[int, int, int, int]],
    num_steps: int,
    width: int,
    height: int
) -> List[List[Tuple[int, int, int, int]]]:
    """
    Interpolate blur rectangles between keyframes for smooth transitions.
    
    Returns list of rect lists, one per interpolated frame.
    """
    frame_rects: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(num_steps)]
    matched_end_indices: set = set()
    
    for s_rect in start_rects:
        sx, sy, sw, sh = s_rect
        s_center = (sx + sw / 2, sy + sh / 2)
        best_match_idx = -1
        min_dist = float('inf')
        
        for i, e_rect in enumerate(end_rects):
            if i in matched_end_indices:
                continue
            ex, ey, ew, eh = e_rect
            e_center = (ex + ew / 2, ey + eh / 2)
            dist = np.sqrt((s_center[0] - e_center[0])**2 + (s_center[1] - e_center[1])**2)
            
            if dist < (width * RECT_MATCH_DISTANCE) and dist < min_dist:
                min_dist = dist
                best_match_idx = i
        
        if best_match_idx != -1:
            matched_end_indices.add(best_match_idx)
            ex, ey, ew, eh = end_rects[best_match_idx]
            
            for i in range(num_steps):
                alpha = i / max(1, num_steps - 1)
                curr_x = int(sx * (1 - alpha) + ex * alpha)
                curr_y = int(sy * (1 - alpha) + ey * alpha)
                curr_w = int(sw * (1 - alpha) + ew * alpha)
                curr_h = int(sh * (1 - alpha) + eh * alpha)
                frame_rects[i].append((curr_x, curr_y, curr_w, curr_h))
        else:
            for i in range(num_steps):
                frame_rects[i].append(s_rect)
    
    # Add unmatched end rectangles
    for i, e_rect in enumerate(end_rects):
        if i not in matched_end_indices:
            for f_idx in range(num_steps):
                frame_rects[f_idx].append(e_rect)
    
    return frame_rects


# =============================================================================
# MOTION DETECTION
# =============================================================================

def detect_motion(frame_buffer: List[np.ndarray], threshold: float) -> List[bool]:
    """Detect which frames have significant motion (scrolling)."""
    if len(frame_buffer) < 2:
        return [False] * len(frame_buffer)
    
    flags = [False] * len(frame_buffer)
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frame_buffer]
    
    for i in range(len(frame_buffer) - 1):
        diff = cv2.absdiff(gray_frames[i], gray_frames[i + 1])
        if cv2.mean(diff)[0] > threshold:
            flags[i] = True
            flags[i + 1] = True
    
    # Safety margin
    safety_flags = flags.copy()
    for i in range(1, len(flags)):
        if flags[i - 1]:
            safety_flags[i] = True
    
    return safety_flags


# =============================================================================
# BLUR APPLICATION
# =============================================================================

def apply_blurs(
    frame: np.ndarray,
    rects: List[Tuple[int, int, int, int]],
    is_keyboard_active: bool,
    is_scrolling: bool,
    enable_keyboard: bool
) -> np.ndarray:
    """Apply blur effects to a frame."""
    height, width = frame.shape[:2]
    
    if is_scrolling:
        # Full frame blur during scrolling
        np.copyto(frame, cv2.GaussianBlur(frame, SCROLL_BLUR_KERNEL, BLUR_SIGMA))
    else:
        # Individual region blurs
        for (bx, by, bw, bh) in rects:
            bx, by = int(max(0, bx)), int(max(0, by))
            bw = int(min(bw, width - bx))
            bh = int(min(bh, height - by))
            
            if bw > 0 and bh > 0:
                try:
                    roi = frame[by:by + bh, bx:bx + bw]
                    frame[by:by + bh, bx:bx + bw] = cv2.GaussianBlur(
                        roi, REGION_BLUR_KERNEL, BLUR_SIGMA
                    )
                except Exception as e:
                    logger.warning(f"Failed to blur region: {e}")
        
        # Keyboard blocker
        if enable_keyboard and is_keyboard_active:
            kb_h = int(height * KEYBOARD_BLOCKER_HEIGHT)
            cv2.rectangle(frame, (0, height - kb_h), (width, height), (0, 0, 0), -1)
    
    return frame
