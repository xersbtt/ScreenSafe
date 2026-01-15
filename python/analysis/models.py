"""
ScreenSafe AI Analysis Pipeline - Type Definitions

Pydantic models for data exchange between Python sidecar and Tauri frontend.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from enum import Enum
import uuid


class PIIType(str, Enum):
    """Types of personally identifiable information"""
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit-card"
    PASSWORD = "password"
    NAME = "name"
    ADDRESS = "address"
    SSN = "ssn"
    API_KEY = "api-key"
    MANUAL = "manual"
    OTHER = "other"  # For watch list matches
    ANCHOR = "anchor"  # OCR-detected anchor-relative blur
    WATCHLIST = "watchlist"  # OCR-detected watchlist match
    BLACKOUT = "blackout"  # Solid black overlay (keyboard blocker)



class BoundingBox(BaseModel):
    """Normalized bounding box (0-1 coordinates)"""
    x: float = Field(ge=0, le=1, description="Left edge")
    y: float = Field(ge=0, le=1, description="Top edge")
    width: float = Field(ge=0, le=1, description="Width")
    height: float = Field(ge=0, le=1, description="Height")
    
    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates (x1, y1, x2, y2)"""
        x1 = int(self.x * frame_width)
        y1 = int(self.y * frame_height)
        x2 = int((self.x + self.width) * frame_width)
        y2 = int((self.y + self.height) * frame_height)
        return x1, y1, x2, y2
    
    @classmethod
    def from_pixels(cls, x1: int, y1: int, x2: int, y2: int, 
                    frame_width: int, frame_height: int) -> "BoundingBox":
        """Create from pixel coordinates"""
        return cls(
            x=x1 / frame_width,
            y=y1 / frame_height,
            width=(x2 - x1) / frame_width,
            height=(y2 - y1) / frame_height
        )


class TextDetection(BaseModel):
    """Raw text detection from OCR"""
    text: str
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    frame_number: int


class Detection(BaseModel):
    """A detected sensitive region with classification"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: PIIType
    content: str
    confidence: float = Field(ge=0, le=1)
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    bbox: BoundingBox
    is_redacted: bool = True
    track_id: Optional[str] = None
    # Per-frame positions for motion tracking: [[frame, x, y, w, h], ...]
    frame_positions: Optional[List[List[float]]] = None


class AnalysisStage(str, Enum):
    """Stages of the analysis pipeline"""
    EXTRACTING = "extracting"
    DETECTING = "detecting"
    TRACKING = "tracking"
    CLASSIFYING = "classifying"
    COMPLETE = "complete"


class AnalysisProgress(BaseModel):
    """Progress update during analysis"""
    stage: AnalysisStage
    progress: float = Field(ge=0, le=100)
    frames_processed: int
    total_frames: int
    detections_found: int
    estimated_time_remaining: float  # seconds
    current_message: str = ""


class AnalysisSettings(BaseModel):
    """User-configurable analysis settings"""
    mode: Literal["fast", "quality"] = "quality"
    detect_emails: bool = True
    detect_phones: bool = True
    detect_credit_cards: bool = True
    detect_passwords: bool = True
    detect_names: bool = True
    detect_addresses: bool = True
    detect_api_keys: bool = True
    frame_skip: int = 1  # Analyze every Nth frame (1 = all frames)
    ocr_scale: float = 1.0  # Scale factor for OCR (1.0 = native, 2.0 = upscaled)
    smart_sampling: bool = True  # Adaptive sampling based on visual activity


class VideoInfo(BaseModel):
    """Metadata about the input video"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds
    codec: str = ""


class AnalysisRequest(BaseModel):
    """Request to start video analysis"""
    video_path: str
    settings: AnalysisSettings = Field(default_factory=AnalysisSettings)


class AnalysisResult(BaseModel):
    """Final result of video analysis"""
    video_info: VideoInfo
    detections: List[Detection]
    analysis_time: float  # seconds
    settings_used: AnalysisSettings


# WebSocket message types
class WSMessageType(str, Enum):
    """Types of WebSocket messages"""
    START_ANALYSIS = "start_analysis"
    CANCEL_ANALYSIS = "cancel_analysis"
    PROGRESS_UPDATE = "progress_update"
    DETECTION_FOUND = "detection_found"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_ERROR = "analysis_error"
    TRACK_OBJECT = "track_object"
    TRACK_RESULT = "track_result"
    # Export messages
    START_EXPORT = "start_export"
    EXPORT_PROGRESS = "export_progress"
    EXPORT_COMPLETE = "export_complete"
    EXPORT_ERROR = "export_error"
    # Scan messages (OCR-only, no encoding)
    SCAN_COMPLETE = "scan_complete"
    # PII Wizard config-based processing
    UPDATE_CONFIG = "update_config"  # Send new watch list, anchors, zones
    PREVIEW_FRAME = "preview_frame"  # Analyze single frame with current config
    GET_TEXT_AT_CLICK = "get_text_at_click"  # Get OCR text at clicked position
    GET_TEXT_IN_REGION = "get_text_in_region"  # Get OCR text from drawn region
    # System info messages
    SYSTEM_INFO = "system_info"  # Broadcast GPU/system capabilities on connect


class WSMessage(BaseModel):
    """WebSocket message wrapper"""
    type: WSMessageType
    payload: dict
