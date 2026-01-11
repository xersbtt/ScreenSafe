"""
ScreenSafe AI Analysis Pipeline

PII Classifier - Combines regex matching with user-defined watch lists
and anchor-based detection from PII Wizard.
"""

import re
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field

from .models import TextDetection, Detection, PIIType, BoundingBox

logger = logging.getLogger(__name__)


# =============================================================================
# ANCHOR CONFIGURATION
# =============================================================================

@dataclass
class AnchorConfig:
    """Configuration for an anchor-based blur zone."""
    direction: str  # BELOW, RIGHT, ABOVE, LEFT
    gap: int        # Pixels from anchor text
    width: int      # Blur box width
    height: int     # Blur box height


# Default anchors for common form labels - blur the input field area
DEFAULT_ANCHORS = {
    # Password fields
    "password": AnchorConfig("RIGHT", 10, 250, 35),
    "password:": AnchorConfig("RIGHT", 5, 250, 35),
    "passwort": AnchorConfig("RIGHT", 10, 250, 35),  # German
    
    # Email fields
    "email": AnchorConfig("RIGHT", 10, 300, 35),
    "email:": AnchorConfig("RIGHT", 5, 300, 35),
    "e-mail": AnchorConfig("RIGHT", 10, 300, 35),
    
    # Credit card fields
    "card number": AnchorConfig("RIGHT", 10, 200, 35),
    "card number:": AnchorConfig("RIGHT", 5, 200, 35),
    "credit card": AnchorConfig("RIGHT", 10, 200, 35),
    "debit card": AnchorConfig("RIGHT", 10, 200, 35),
    
    # CVV/CVC
    "cvv": AnchorConfig("RIGHT", 10, 60, 35),
    "cvc": AnchorConfig("RIGHT", 10, 60, 35),
    "cvv:": AnchorConfig("RIGHT", 5, 60, 35),
    "cvc:": AnchorConfig("RIGHT", 5, 60, 35),
    "security code": AnchorConfig("RIGHT", 10, 60, 35),
    
    # Expiry
    "expiry": AnchorConfig("RIGHT", 10, 80, 35),
    "exp date": AnchorConfig("RIGHT", 10, 80, 35),
    "mm/yy": AnchorConfig("RIGHT", 10, 80, 35),
    
    # Name fields
    "full name": AnchorConfig("RIGHT", 10, 250, 35),
    "name:": AnchorConfig("RIGHT", 5, 250, 35),
    "cardholder": AnchorConfig("RIGHT", 10, 250, 35),
    
    # Address
    "address": AnchorConfig("RIGHT", 10, 350, 35),
    "street": AnchorConfig("RIGHT", 10, 350, 35),
    "zip": AnchorConfig("RIGHT", 10, 100, 35),
    "postal code": AnchorConfig("RIGHT", 10, 100, 35),
    
    # Phone
    "phone": AnchorConfig("RIGHT", 10, 180, 35),
    "phone:": AnchorConfig("RIGHT", 5, 180, 35),
    "mobile": AnchorConfig("RIGHT", 10, 180, 35),
    
    # SSN
    "ssn": AnchorConfig("RIGHT", 10, 150, 35),
    "social security": AnchorConfig("RIGHT", 10, 150, 35),
    
    # Account numbers
    "account number": AnchorConfig("RIGHT", 10, 200, 35),
    "account #": AnchorConfig("RIGHT", 10, 200, 35),
    "routing number": AnchorConfig("RIGHT", 10, 150, 35),
}


# =============================================================================
# PII VALIDATION FUNCTIONS
# =============================================================================

def luhn_checksum(card_number: str) -> bool:
    """Validate credit card with Luhn algorithm"""
    digits = [int(d) for d in card_number if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(divmod(d * 2, 10))
    
    return checksum % 10 == 0


def is_valid_email(text: str) -> bool:
    """Email validation - slightly relaxed for OCR errors"""
    # Must contain @ and at least one dot after @
    if '@' not in text or '.' not in text.split('@')[-1]:
        return False
    
    # Basic pattern - allows some OCR errors
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, text.strip()):
        return False
    
    # Minimum lengths
    parts = text.split('@')
    if len(parts) != 2:
        return False
    local, domain = parts
    if len(local) < 2 or len(domain) < 4:
        return False
    
    return True


def is_valid_phone(text: str) -> bool:
    """Phone validation - must look like a phone number, not a date"""
    text = text.strip()
    
    # Exclude date patterns (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, etc.)
    date_patterns = [
        r'^\d{1,2}/\d{1,2}/\d{2,4}$',      # 5/12/2025 or 05/12/25
        r'^\d{1,2}-\d{1,2}-\d{2,4}$',      # 5-12-2025
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',  # 2025-05-12
        r'^\d{1,2}\.\d{1,2}\.\d{2,4}$',    # 5.12.2025
    ]
    for pattern in date_patterns:
        if re.match(pattern, text):
            return False
    
    digits = re.sub(r'\D', '', text)
    
    # Must have 10-15 digits for phone (strict: no short numbers)
    if len(digits) < 10 or len(digits) > 15:
        return False
    
    # Must be mostly digits (at least 70% of characters)
    non_space = text.replace(' ', '')
    if len(non_space) > 0:
        digit_ratio = len(digits) / len(non_space)
        if digit_ratio < 0.7:
            return False
    
    return True


def is_valid_credit_card(text: str) -> bool:
    """Strict credit card validation with Luhn"""
    digits = re.sub(r'\D', '', text)
    if len(digits) < 13 or len(digits) > 19:
        return False
    return luhn_checksum(digits)


def is_valid_ssn(text: str) -> bool:
    """Strict SSN validation - XXX-XX-XXXX format"""
    pattern = r'^(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}$'
    return bool(re.match(pattern, text.strip()))


def is_valid_api_key(text: str) -> bool:
    """Strict API key validation"""
    patterns = [
        r'^sk-[a-zA-Z0-9]{32,}$',
        r'^AIza[0-9A-Za-z\-_]{35}$',
        r'^ghp_[a-zA-Z0-9]{36}$',
        r'^xox[baprs]-[0-9A-Za-z\-]{24,}$'
    ]
    return any(re.match(p, text.strip()) for p in patterns)


# =============================================================================
# PII CLASSIFIER
# =============================================================================

class PIIClassifier:
    """
    PII Classifier with:
    - Strict regex validation (Luhn, email format, etc.)
    - User-defined watch list (custom text to always blur)
    - Anchor-based detection (blur relative to labels like "Password:")
    """
    
    def __init__(self, 
                 watch_list: List[str] = None,
                 anchors: Dict[str, AnchorConfig] = None,
                 use_default_anchors: bool = True,
                 use_presidio: bool = False):
        """
        Initialize PII classifier.
        
        Args:
            watch_list: List of text strings to always blur (case-insensitive)
            anchors: Dict of label text -> AnchorConfig for context-based blurring
            use_default_anchors: Whether to include built-in anchors for common labels
            use_presidio: Whether to use Presidio NLP (not implemented)
        """
        self.watch_list = [w.lower() for w in (watch_list or [])]
        
        # Start with default anchors, then overlay user anchors
        if use_default_anchors:
            self.anchors = dict(DEFAULT_ANCHORS)
        else:
            self.anchors = {}
        
        # Add/override with user-provided anchors
        if anchors:
            self.anchors.update(anchors)
        
        self.use_presidio = use_presidio
        self._initialized = True
        
    def initialize(self) -> None:
        pass
    
    def add_to_watch_list(self, text: str) -> None:
        """Add text to the watch list"""
        self.watch_list.append(text.lower())
    
    def add_anchor(self, label: str, direction: str = "BELOW", 
                   gap: int = 10, width: int = 300, height: int = 50) -> None:
        """Add an anchor for context-based blurring"""
        self.anchors[label.lower()] = AnchorConfig(
            direction=direction, gap=gap, width=width, height=height
        )
    
    def check_watch_list(self, text: str) -> bool:
        """Check if text matches any watch list item"""
        text_lower = text.lower().strip()
        text_nospace = text_lower.replace(" ", "").replace("-", "")
        
        for secret in self.watch_list:
            if secret in text_lower or secret in text_nospace:
                return True
        return False
    
    def check_anchor(self, text: str) -> Optional[Tuple[str, AnchorConfig]]:
        """Check if text matches any anchor label"""
        text_lower = text.lower().strip()
        
        for label, config in self.anchors.items():
            if label in text_lower:
                return (label, config)
        return None
    
    def classify_text(self, text: str) -> Optional[Tuple[PIIType, float]]:
        """
        Classify a single text string.
        Returns (pii_type, confidence) or None if not PII.
        """
        text = text.strip()
        
        # Skip very short or very long text
        if len(text) < 5 or len(text) > 100:
            return None
        
        # Check watch list first (user-defined always wins)
        if self.check_watch_list(text):
            return (PIIType.OTHER, 0.99)
        
        # Email
        if '@' in text and '.' in text:
            if is_valid_email(text):
                return (PIIType.EMAIL, 0.95)
        
        # Phone
        if re.search(r'\d', text):
            if is_valid_phone(text):
                return (PIIType.PHONE, 0.90)
        
        # Credit Card
        digits_only = re.sub(r'\D', '', text)
        if len(digits_only) >= 13 and len(digits_only) <= 19:
            if is_valid_credit_card(text):
                return (PIIType.CREDIT_CARD, 0.95)
        
        # SSN
        if is_valid_ssn(text):
            return (PIIType.SSN, 0.95)
        
        # API Key
        if is_valid_api_key(text):
            return (PIIType.API_KEY, 0.95)
        
        return None
    
    def classify_detection(self, 
                          detection: TextDetection,
                          context_detections: List[TextDetection] = None,
                          frame_width: int = 1920,
                          frame_height: int = 1080) -> Optional[Detection]:
        """
        Classify a text detection, including anchor-based detection.
        
        Args:
            detection: The text detection to classify
            context_detections: Other detections in the same frame (for context)
            frame_width: Video frame width for offset calculations
            frame_height: Video frame height for offset calculations
        """
        # Check for anchor match (context-sensitive)
        anchor_match = self.check_anchor(detection.text)
        if anchor_match:
            label, config = anchor_match
            
            # Calculate the OFFSET bbox based on anchor direction
            # The blur should appear relative to the label, not on the label itself
            anchor_bbox = detection.bbox
            
            # Convert anchor config dimensions to normalized coordinates
            gap_norm_x = config.gap / frame_width
            gap_norm_y = config.gap / frame_height
            width_norm = config.width / frame_width
            height_norm = config.height / frame_height
            
            # Calculate new bbox based on direction
            if config.direction == "BELOW":
                new_x = anchor_bbox.x
                new_y = anchor_bbox.y + anchor_bbox.height + gap_norm_y
                new_width = width_norm
                new_height = height_norm
            elif config.direction == "RIGHT":
                new_x = anchor_bbox.x + anchor_bbox.width + gap_norm_x
                new_y = anchor_bbox.y
                new_width = width_norm
                new_height = height_norm
            elif config.direction == "ABOVE":
                new_x = anchor_bbox.x
                new_y = anchor_bbox.y - gap_norm_y - height_norm
                new_width = width_norm
                new_height = height_norm
            elif config.direction == "LEFT":
                new_x = anchor_bbox.x - gap_norm_x - width_norm
                new_y = anchor_bbox.y
                new_width = width_norm
                new_height = height_norm
            else:
                # Default to BELOW
                new_x = anchor_bbox.x
                new_y = anchor_bbox.y + anchor_bbox.height + gap_norm_y
                new_width = width_norm
                new_height = height_norm
            
            # Clamp to valid range
            new_x = max(0, min(1 - new_width, new_x))
            new_y = max(0, min(1 - new_height, new_y))
            
            offset_bbox = BoundingBox(
                x=new_x,
                y=new_y,
                width=new_width,
                height=new_height
            )
            
            return Detection(
                type=PIIType.PASSWORD,  # Anchors typically protect sensitive input
                content=f"[Near: {label}]",
                confidence=0.90,
                start_frame=detection.frame_number,
                end_frame=detection.frame_number,
                start_time=0,
                end_time=0,
                bbox=offset_bbox,  # Use the calculated offset position!
                is_redacted=True,
                metadata={"anchor": label, "config": config.__dict__}
            )
        
        # Regular PII classification
        result = self.classify_text(detection.text)
        
        if result is None:
            return None
        
        pii_type, confidence = result
        
        return Detection(
            type=pii_type,
            content=detection.text,
            confidence=confidence,
            start_frame=detection.frame_number,
            end_frame=detection.frame_number,
            start_time=0,
            end_time=0,
            bbox=detection.bbox,
            is_redacted=True
        )
    
    def cleanup(self) -> None:
        pass


# Singleton
_classifier: Optional[PIIClassifier] = None


def get_pii_classifier(watch_list: List[str] = None,
                       anchors: Dict[str, AnchorConfig] = None) -> PIIClassifier:
    global _classifier
    if _classifier is None:
        _classifier = PIIClassifier(watch_list=watch_list, anchors=anchors)
    return _classifier
