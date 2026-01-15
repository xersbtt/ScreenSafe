"""
ScreenSafe AI Analysis Pipeline

OCR Engine using EasyOCR for text detection and recognition.
Optimized for screen recordings with high text density.
"""

import numpy as np
from typing import Optional, List
import logging

from .models import TextDetection, BoundingBox

logger = logging.getLogger(__name__)


class OCREngine:
    """
    OCR Engine using EasyOCR for text detection and recognition.
    
    Optimized for:
    - Screen recordings with UI text
    - High accuracy on typed/rendered text
    - Works on Windows without build tools
    """
    
    def __init__(self, use_gpu: bool = False, lang: List[str] = None):
        """
        Initialize OCR engine.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Languages for recognition (default: ['en'])
        """
        self.use_gpu = use_gpu
        self.lang = lang or ['en']
        self._reader = None
        self._initialized = False
        
    def initialize(self) -> None:
        """Lazy initialization of EasyOCR reader"""
        if self._initialized:
            return
            
        try:
            import easyocr
            
            logger.info(f"Initializing EasyOCR (GPU: {self.use_gpu}, Lang: {self.lang})")
            
            self._reader = easyocr.Reader(
                self.lang,
                gpu=self.use_gpu,
                verbose=False
            )
            
            self._initialized = True
            logger.info("EasyOCR initialized successfully")
            
        except ImportError:
            logger.error("EasyOCR not installed. Run: pip install easyocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def detect_text(self, 
                    frame: np.ndarray, 
                    frame_number: int,
                    min_confidence: float = 0.5,
                    prev_frame: np.ndarray = None,
                    scale: float = 1.0) -> List[TextDetection]:
        """
        Detect and recognize text in a video frame.
        Skips OCR if frame is very similar to previous (for speed).
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            frame_number: Frame index for tracking
            min_confidence: Minimum confidence threshold
            prev_frame: Previous frame for change detection
            scale: Scale factor for image (1.0 = native, 2.0 = 2x size)
            
        Returns:
            List of TextDetection objects
        """
        if not self._initialized:
            self.initialize()
        
        # Skip if frame is very similar to previous (change detection)
        if prev_frame is not None:
            diff = np.abs(frame.astype(float) - prev_frame.astype(float)).mean()
            if diff < 5.0:  # Less than 5 units mean difference = skip
                return []
        
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Run OCR - EasyOCR expects RGB
            rgb_frame = frame[:, :, ::-1]  # BGR to RGB
            
            # Apply scaling if requested
            if scale != 1.0 and scale > 0:
                h, w = rgb_frame.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                rgb_frame = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            results = self._reader.readtext(rgb_frame)
            
            if results is None or len(results) == 0:
                return detections
            
            # Process results
            # Each result is (bbox_points, text, confidence)
            for detection in results:
                if detection is None or len(detection) < 3:
                    continue
                
                bbox_points, text, confidence = detection
                
                if confidence < min_confidence:
                    continue
                
                # Convert polygon to bounding box
                # bbox_points is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x_coords = [p[0] for p in bbox_points]
                y_coords = [p[1] for p in bbox_points]
                
                x1 = min(x_coords)
                y1 = min(y_coords)
                x2 = max(x_coords)
                y2 = max(y_coords)
                
                # Create normalized bounding box
                bbox = BoundingBox.from_pixels(
                    int(x1), int(y1), int(x2), int(y2),
                    frame_width, frame_height
                )
                
                detections.append(TextDetection(
                    text=str(text).strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                    frame_number=frame_number
                ))
                    
        except Exception as e:
            logger.warning(f"OCR failed on frame {frame_number}: {e}")
        
        return detections
    
    def batch_detect(self, 
                     frames: List[np.ndarray],
                     start_frame: int = 0,
                     min_confidence: float = 0.5) -> List[List[TextDetection]]:
        """
        Detect text in multiple frames.
        
        Args:
            frames: List of BGR images
            start_frame: Starting frame number for indexing
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detection lists, one per frame
        """
        results = []
        for i, frame in enumerate(frames):
            detections = self.detect_text(frame, start_frame + i, min_confidence)
            results.append(detections)
        return results
    
    def cleanup(self) -> None:
        """Release resources"""
        self._reader = None
        self._initialized = False
        logger.info("OCR engine cleaned up")


# Singleton instance for reuse
_ocr_engine: Optional[OCREngine] = None


def get_ocr_engine(use_gpu: bool = False, lang: List[str] = None) -> OCREngine:
    """Get or create the singleton OCR engine instance"""
    global _ocr_engine
    
    if _ocr_engine is None:
        _ocr_engine = OCREngine(use_gpu=use_gpu, lang=lang)
    
    return _ocr_engine
