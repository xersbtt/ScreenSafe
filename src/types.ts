// ============================================
// ScreenSafe Type Definitions
// ============================================

/**
 * Represents a detected region in a video frame
 */
export interface Detection {
    id: string;
    type: PIIType;
    content: string;
    confidence: number;
    startTime: number;  // seconds
    endTime: number;    // seconds
    bbox: BoundingBox;
    isRedacted: boolean;
    trackId?: string;   // For tracked objects across frames
    // Per-frame positions for motion tracking: [frame, x, y, w, h][]
    // If present, blur follows these positions instead of static bbox
    framePositions?: Array<[number, number, number, number, number]>;
}

/**
 * Types of PII that can be detected
 */
export type PIIType =
    | 'email'
    | 'phone'
    | 'credit-card'
    | 'password'
    | 'name'
    | 'address'
    | 'ssn'
    | 'api-key'
    | 'manual'
    | 'anchor'
    | 'watchlist'
    | 'blackout';

/**
 * Bounding box for a detection (normalized 0-1 coordinates)
 */
export interface BoundingBox {
    x: number;      // Left edge (0-1)
    y: number;      // Top edge (0-1)
    width: number;  // Width (0-1)
    height: number; // Height (0-1)
}

/**
 * Analysis settings
 */
export interface AnalysisSettings {
    mode: 'fast' | 'quality';
    detectEmails: boolean;
    detectPhones: boolean;
    detectCreditCards: boolean;
    detectPasswords: boolean;
    detectNames: boolean;
    detectAddresses: boolean;
    detectApiKeys: boolean;
}

/**
 * Project state (can be saved/loaded)
 */
export interface Project {
    id: string;
    name: string;
    videoPath: string;
    duration: number;
    width: number;
    height: number;
    fps: number;
    createdAt: string;
    updatedAt: string;
    detections: Detection[];
    settings: AnalysisSettings;
}

/**
 * Analysis progress state
 */
export interface AnalysisProgress {
    isAnalyzing: boolean;
    stage: 'extracting' | 'detecting' | 'tracking' | 'classifying' | 'complete';
    progress: number;       // 0-100
    framesProcessed: number;
    totalFrames: number;
    detectionsFound: number;
    estimatedTimeRemaining: number; // seconds
}

/**
 * Video player state
 */
export interface VideoState {
    isPlaying: boolean;
    currentTime: number;
    duration: number;
    volume: number;
    isMuted: boolean;
    playbackRate: number;
}

/**
 * Application state
 */
export type AppState =
    | { status: 'idle' }
    | { status: 'loading'; videoPath: string }
    | { status: 'analyzing'; videoPath: string; progress: AnalysisProgress }
    | { status: 'editing'; project: Project }
    | { status: 'exporting'; project: Project; exportProgress: number };

/**
 * PII category metadata for UI display
 */
export const PII_CATEGORIES: Record<PIIType, {
    label: string;
    icon: string;
    color: string;
    description: string;
}> = {
    'email': {
        label: 'Email',
        icon: '📧',
        color: 'var(--color-pii-email)',
        description: 'Email addresses'
    },
    'phone': {
        label: 'Phone',
        icon: '📞',
        color: 'var(--color-pii-phone)',
        description: 'Phone numbers'
    },
    'credit-card': {
        label: 'Credit Card',
        icon: '💳',
        color: 'var(--color-pii-credit-card)',
        description: 'Credit card numbers'
    },
    'password': {
        label: 'Password',
        icon: '🔒',
        color: 'var(--color-pii-password)',
        description: 'Passwords and PINs'
    },
    'name': {
        label: 'Name',
        icon: '👤',
        color: 'var(--color-pii-name)',
        description: 'Personal names'
    },
    'address': {
        label: 'Address',
        icon: '📍',
        color: 'var(--color-pii-address)',
        description: 'Physical addresses'
    },
    'ssn': {
        label: 'SSN',
        icon: '🆔',
        color: 'var(--color-pii-password)',
        description: 'Social Security Numbers'
    },
    'api-key': {
        label: 'API Key',
        icon: '🔑',
        color: 'var(--color-pii-credit-card)',
        description: 'API keys and tokens'
    },
    'manual': {
        label: 'Manual',
        icon: '✏️',
        color: 'var(--color-pii-manual)',
        description: 'Manually selected regions'
    },
    'anchor': {
        label: 'Anchor',
        icon: '🎯',
        color: 'var(--color-primary)',
        description: 'OCR-detected anchor-relative regions'
    },
    'watchlist': {
        label: 'Watchlist',
        icon: '👁️',
        color: 'var(--color-warning)',
        description: 'OCR-detected watchlist matches'
    },
    'blackout': {
        label: 'Blackout',
        icon: '⬛',
        color: '#000000',
        description: 'Solid black overlay (keyboard blocker)'
    }
};

/**
 * Helper to format time as MM:SS or HH:MM:SS
 */
export function formatTime(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);

    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Helper to generate unique IDs
 */
export function generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
