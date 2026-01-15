"""
ScreenSafe AI Analysis Pipeline

Video Analyzer - Main orchestrator for the analysis pipeline.
Coordinates frame extraction, OCR, PII detection, and tracking.

Integrated features from PII Wizard:
- Watch list (user-defined text to blur)
- Anchor-based detection (blur near labels like "Password:")
- Motion detection (blur during scrolling)
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Callable, Optional, Generator, List, Dict

from .models import (
    Detection, TextDetection, VideoInfo, AnalysisSettings,
    AnalysisProgress, AnalysisStage, AnalysisResult, PIIType
)
from .ocr_engine import get_ocr_engine
from .pii_classifier import get_pii_classifier, PIIClassifier, AnchorConfig
from .tracker import get_tracker, SimpleBBoxTracker

logger = logging.getLogger(__name__)

# Motion detection constants
MOTION_THRESHOLD = 3.0  # Mean pixel difference to consider as motion


class VideoAnalyzer:
    """
    Main video analysis orchestrator.
    
    Pipeline:
    1. Extract frames from video
    2. Run OCR on each frame
    3. Classify text as PII
    4. Track detections across frames
    5. Merge detections into time ranges
    
    Features from PII Wizard:
    - Watch list: User-defined text to always blur
    - Anchors: Blur area relative to labels (e.g., below "Password:")
    - Motion detection: Blur entire frame during scrolling
    """
    
    def __init__(self, 
                 settings: Optional[AnalysisSettings] = None,
                 watch_list: List[str] = None,
                 anchors: Dict[str, dict] = None):
        """
        Initialize analyzer.
        
        Args:
            settings: Analysis configuration
            watch_list: List of text to always blur
            anchors: Dict of label -> {"direction": "BELOW", "gap": 10, "width": 300, "height": 50}
        """
        self.settings = settings or AnalysisSettings()
        self.watch_list = watch_list or []
        self.anchors = anchors or {}
        self.ocr = None
        self.classifier = None
        self.tracker = None
        self._cancelled = False
        
    def initialize(self, use_gpu: bool = False) -> None:
        """Initialize all components"""
        logger.info("Initializing video analyzer components...")
        
        self.ocr = get_ocr_engine(use_gpu=use_gpu)
        
        # Create anchor configs
        anchor_configs = {}
        for label, cfg in self.anchors.items():
            anchor_configs[label] = AnchorConfig(
                direction=cfg.get("direction", "BELOW"),
                gap=cfg.get("gap", 10),
                width=cfg.get("width", 300),
                height=cfg.get("height", 50)
            )
        
        self.classifier = PIIClassifier(
            watch_list=self.watch_list,
            anchors=anchor_configs
        )
        self.tracker = get_tracker(use_norfair=True)
        
        logger.info("Video analyzer ready")
    
    def get_video_info(self, video_path: str) -> VideoInfo:
        """Extract video metadata"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Try to get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            return VideoInfo(
                path=video_path,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration=duration,
                codec=codec.strip()
            )
        finally:
            cap.release()
    
    def extract_frames(self, 
                      video_path: str,
                      frame_skip: int = 1) -> Generator[tuple[int, np.ndarray], None, None]:
        """
        Generator that yields frames from video.
        
        Args:
            video_path: Path to video file
            frame_skip: Only yield every Nth frame
            
        Yields:
            (frame_number, frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_number % frame_skip == 0:
                    yield frame_number, frame
                
                frame_number += 1
                
                if self._cancelled:
                    break
                    
        finally:
            cap.release()
    
    def analyze(self,
                video_path: str,
                progress_callback: Optional[Callable[[AnalysisProgress], None]] = None,
                detection_callback: Optional[Callable[[Detection], None]] = None
               ) -> AnalysisResult:
        """
        Analyze video for sensitive content.
        
        Args:
            video_path: Path to video file
            progress_callback: Called with progress updates
            detection_callback: Called when new detection found
            
        Returns:
            AnalysisResult with all detections
        """
        start_time = time.time()
        self._cancelled = False
        
        # Get video info
        video_info = self.get_video_info(video_path)
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"  Resolution: {video_info.width}x{video_info.height}")
        logger.info(f"  Duration: {video_info.duration:.1f}s, FPS: {video_info.fps:.1f}")
        logger.info(f"  Total frames: {video_info.total_frames}")
        
        # Determine frame skip based on settings
        frame_skip = self.settings.frame_skip
        if self.settings.mode == "fast":
            frame_skip = max(frame_skip, 3)  # Analyze every 3rd frame in fast mode
        
        frames_to_process = video_info.total_frames // frame_skip
        
        # Initialize components
        if self.ocr is None:
            self.initialize()
        
        # Storage for detections (classifier output only - no raw OCR storage)
        all_detections: List[Detection] = []
        
        # Progress tracking
        frames_processed = 0
        last_progress_time = time.time()
        
        # Adaptive Sampling State
        frames_skipped = 0
        prev_frame = None
        last_scan_frame = -999
        
        def update_progress(stage: AnalysisStage, message: str = ""):
            if progress_callback:
                elapsed = time.time() - start_time
                if frames_processed > 0:
                    time_per_frame = elapsed / frames_processed
                    remaining_frames = frames_to_process - frames_processed
                    eta = time_per_frame * remaining_frames
                else:
                    eta = 0
                
                progress_callback(AnalysisProgress(
                    stage=stage,
                    progress=min(100, (frames_processed / max(1, frames_to_process)) * 100),
                    frames_processed=frames_processed,
                    total_frames=frames_to_process,
                    detections_found=len(all_detections),
                    estimated_time_remaining=eta,
                    current_message=message
                ))
        
        # Stage 1: Extract and OCR frames
        update_progress(AnalysisStage.EXTRACTING, "Starting frame extraction...")
        
        for frame_number, frame in self.extract_frames(video_path, frame_skip):
            if self._cancelled:
                break
            
            # Stage 2: Adaptive Scanning (Smart Sampling)
            if frames_processed == 0:
                update_progress(AnalysisStage.DETECTING, "Running Adaptive Scan...")
            
            # --- Adaptive Scanning Logic ---
            should_scan = True
            is_motion = False
            
            if self.settings.smart_sampling and prev_frame is not None:
                # 1. Downscale for fast comparison
                curr_small = cv2.resize(frame, (64, 64))
                prev_small = cv2.resize(prev_frame, (64, 64))
                
                # 2. Calculate visual difference (0.0 to 1.0)
                diff = cv2.absdiff(curr_small, prev_small)
                mean_diff = np.mean(diff) / 255.0
                
                # 3. Decision Logic
                if mean_diff < 0.01:  # <1% difference (Static)
                    # Skip scan unless it's been too long (force every 2s)
                    time_since_last = (frame_number - last_scan_frame) / video_info.fps
                    if time_since_last < 2.0:
                        should_scan = False
                
                elif mean_diff > 0.10: # >10% difference (High Motion/Scroll)
                    # Motion Boost! Scan more frequently to catch flying text
                    is_motion = True
                    # If we scanned very recently (e.g. 5 frames ago), we might scan again
                    # But if we just scanned 1 frame ago, maybe wait? 
                    # For now, let's just allow the scan.
            
            # Update prev frame
            prev_frame = frame.copy()
            
            # Skip if not needed
            if not should_scan:
                frames_skipped += 1
                continue
                
            last_scan_frame = frame_number
            
            # Run OCR on frame (with scaling)
            text_detections = self.ocr.detect_text(
                frame, 
                frame_number,
                min_confidence=0.5,
                scale=self.settings.ocr_scale
            )
            
            # If motion boost is active, detections might be blurry/partial
            # We accept them anyway as they are better than nothing
            
            # Classify PII in real-time
            for text_det in text_detections:
                detection = self.classifier.classify_detection(
                    text_det,
                    context_detections=text_detections,
                    frame_width=video_info.width if self.settings.ocr_scale == 1.0 else int(video_info.width * self.settings.ocr_scale),
                    frame_height=video_info.height if self.settings.ocr_scale == 1.0 else int(video_info.height * self.settings.ocr_scale)
                )
                
                if detection:
                    # Set time based on frame
                    detection.start_time = frame_number / video_info.fps
                    # If motion, validity is short. If static, validity is long (until next scan)
                    # Ideally, we extend until the next scan frame
                    detection.end_time = (frame_number + frame_skip) / video_info.fps # Placeholder, merge logic handles extension
                    detection.start_frame = frame_number
                    detection.end_frame = frame_number + frame_skip
                    
                    all_detections.append(detection)
                    
                    if detection_callback:
                        detection_callback(detection)
            
            frames_processed += 1
            
            # Update progress periodically
            if time.time() - last_progress_time > 0.5:
                # Calculate effective FPS including skips
                effective_fps = frames_processed / (time.time() - start_time)
                msg = f"Adaptive Scan: {frames_skipped} skipped" if frames_skipped > 0 else "Scanning..."
                if is_motion: msg = "Motion Boost Active! 🚀"
                update_progress(AnalysisStage.DETECTING, msg)
                last_progress_time = time.time()
        
        if self._cancelled:
            logger.info("Analysis cancelled")
            return AnalysisResult(
                video_info=video_info,
                detections=[],
                analysis_time=time.time() - start_time,
                settings_used=self.settings
            )
        
        # Stage 4: Track and merge detections
        update_progress(AnalysisStage.TRACKING, "Merging detections...")
        merged_detections = self._merge_detections(all_detections, video_info)
        
        # Stage 5: Complete
        update_progress(AnalysisStage.COMPLETE, "Analysis complete!")
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis complete in {analysis_time:.1f}s")
        logger.info(f"Found {len(merged_detections)} unique detections")
        
        return AnalysisResult(
            video_info=video_info,
            detections=merged_detections,
            analysis_time=analysis_time,
            settings_used=self.settings
        )
    
    def _merge_detections(self, 
                          detections: List[Detection],
                          video_info: VideoInfo) -> List[Detection]:
        """
        Merge detections that track the same content across frames.
        
        Groups detections by:
        1. Similar content (text match)
        2. Similar location (IoU)
        3. Temporal proximity
        """
        if not detections:
            return []
        
        # Sort by start time
        sorted_dets = sorted(detections, key=lambda d: d.start_time)
        
        merged: List[Detection] = []
        content_groups: Dict[str, List[Detection]] = {}
        
        # Group by content
        for det in sorted_dets:
            key = f"{det.type}:{det.content}"
            if key not in content_groups:
                content_groups[key] = []
            content_groups[key].append(det)
        
        # Merge each group
        for key, group in content_groups.items():
            if not group:
                continue
            
            # Start with first detection
            current = Detection(
                id=group[0].id,
                type=group[0].type,
                content=group[0].content,
                confidence=group[0].confidence,
                start_frame=group[0].start_frame,
                end_frame=group[0].end_frame,
                start_time=group[0].start_time,
                end_time=group[0].end_time,
                bbox=group[0].bbox,
                is_redacted=True,
                track_id=group[0].track_id
            )
            
            # Extend with subsequent detections
            for det in group[1:]:
                # Check if temporally adjacent (within 2 seconds)
                if det.start_time - current.end_time < 2.0:
                    # Extend current detection
                    current.end_time = max(current.end_time, det.end_time)
                    current.end_frame = max(current.end_frame, det.end_frame)
                    current.confidence = max(current.confidence, det.confidence)
                else:
                    # Save current and start new
                    merged.append(current)
                    current = Detection(
                        id=det.id,
                        type=det.type,
                        content=det.content,
                        confidence=det.confidence,
                        start_frame=det.start_frame,
                        end_frame=det.end_frame,
                        start_time=det.start_time,
                        end_time=det.end_time,
                        bbox=det.bbox,
                        is_redacted=True,
                        track_id=det.track_id
                    )
            
            # Add last detection
            merged.append(current)
        
        # Sort by start time
        merged.sort(key=lambda d: d.start_time)
        
        return merged
    
    def cancel(self) -> None:
        """Cancel ongoing analysis"""
        self._cancelled = True
        logger.info("Analysis cancellation requested")
    
    def cleanup(self) -> None:
        """Release all resources"""
        if self.ocr:
            self.ocr.cleanup()
        if self.classifier:
            self.classifier.cleanup()
        if self.tracker:
            self.tracker.cleanup() if hasattr(self.tracker, 'cleanup') else self.tracker.reset()
        
        self.ocr = None
        self.classifier = None
        self.tracker = None
