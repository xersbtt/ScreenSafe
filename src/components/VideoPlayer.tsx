import { useRef, useCallback, forwardRef, useImperativeHandle, useState, useEffect } from 'react';
import { Detection, VideoState } from '../types';
import { SelectionOverlay } from './SelectionOverlay';

interface VideoPlayerProps {
    videoUrl: string;
    detections: Detection[];
    currentTime: number;
    onTimeUpdate: (time: number) => void;
    onDurationChange: (duration: number) => void;
    onVideoStateChange: (state: Partial<VideoState>) => void;
    // Selection mode props
    isSelectionMode?: boolean;
    onSelectionComplete?: (bbox: { x: number; y: number; width: number; height: number }, frameTime: number) => void;
    // Preview blur toggle
    previewBlur?: boolean;
    // Selected detection for highlight
    selectedDetection?: Detection | null;
    // Callback for resizing detection bbox
    onBboxChange?: (detectionId: string, bbox: { x: number; y: number; width: number; height: number }) => void;
    onResizeStart?: (detectionId: string) => void;
    onResizeEnd?: (detectionId: string) => void;
    // Callback for clicking/selecting a detection from the video overlay
    onDetectionClick?: (detection: Detection) => void;
    // Callback for deselecting (clicking on empty space)
    onDeselect?: () => void;
}

export interface VideoPlayerHandle {
    play: () => void;
    pause: () => void;
    seek: (time: number) => void;
    getVideoElement: () => HTMLVideoElement | null;
}

type ResizeHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(({
    videoUrl,
    detections,
    currentTime,
    onTimeUpdate,
    onDurationChange,
    onVideoStateChange,
    isSelectionMode = false,
    onSelectionComplete,
    previewBlur = true,
    selectedDetection = null,
    onBboxChange,
    onResizeStart,
    onResizeEnd,
    onDetectionClick,
    onDeselect
}, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Resize drag state
    const [resizeDrag, setResizeDrag] = useState<{
        detectionId: string;
        handle: ResizeHandle;
        startMouseX: number;
        startMouseY: number;
        startBbox: { x: number; y: number; width: number; height: number };
    } | null>(null);

    // Handle resize drag
    useEffect(() => {
        if (!resizeDrag || !onBboxChange || !videoRef.current) {
            return;
        }

        const video = videoRef.current;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        const handleMouseMove = (e: MouseEvent) => {
            const displayWidth = video.clientWidth;
            const displayHeight = video.clientHeight;
            const scaleX = displayWidth / videoWidth;
            const scaleY = displayHeight / videoHeight;
            const scale = Math.min(scaleX, scaleY);

            // Calculate delta in normalized coordinates
            const deltaX = (e.clientX - resizeDrag.startMouseX) / (videoWidth * scale);
            const deltaY = (e.clientY - resizeDrag.startMouseY) / (videoHeight * scale);

            let { x, y, width, height } = resizeDrag.startBbox;

            // Apply delta based on handle
            switch (resizeDrag.handle) {
                case 'nw':
                    x += deltaX; y += deltaY;
                    width -= deltaX; height -= deltaY;
                    break;
                case 'n':
                    y += deltaY; height -= deltaY;
                    break;
                case 'ne':
                    y += deltaY; width += deltaX; height -= deltaY;
                    break;
                case 'e':
                    width += deltaX;
                    break;
                case 'se':
                    width += deltaX; height += deltaY;
                    break;
                case 's':
                    height += deltaY;
                    break;
                case 'sw':
                    x += deltaX; width -= deltaX; height += deltaY;
                    break;
                case 'w':
                    x += deltaX; width -= deltaX;
                    break;
            }

            // Clamp to valid ranges
            x = Math.max(0, Math.min(1 - 0.01, x));
            y = Math.max(0, Math.min(1 - 0.01, y));
            width = Math.max(0.01, Math.min(1 - x, width));
            height = Math.max(0.01, Math.min(1 - y, height));

            onBboxChange(resizeDrag.detectionId, { x, y, width, height });
        };

        const handleMouseUp = () => {
            if (resizeDrag && onResizeEnd) {
                onResizeEnd(resizeDrag.detectionId);
            }
            setResizeDrag(null);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [resizeDrag, onBboxChange, onResizeEnd]);

    const handleResizeStart = useCallback((e: React.MouseEvent, detection: Detection, handle: ResizeHandle) => {
        e.stopPropagation();
        e.preventDefault();
        if (onResizeStart) {
            onResizeStart(detection.id);
        }
        setResizeDrag({
            detectionId: detection.id,
            handle,
            startMouseX: e.clientX,
            startMouseY: e.clientY,
            startBbox: { ...detection.bbox }
        });
    }, []);

    // Expose methods to parent
    useImperativeHandle(ref, () => ({
        play: () => videoRef.current?.play(),
        pause: () => videoRef.current?.pause(),
        seek: (time: number) => {
            if (videoRef.current) {
                videoRef.current.currentTime = time;
            }
        },
        getVideoElement: () => videoRef.current
    }));

    // Handle video events
    const handleTimeUpdate = useCallback(() => {
        if (videoRef.current) {
            onTimeUpdate(videoRef.current.currentTime);
        }
    }, [onTimeUpdate]);

    const handleLoadedMetadata = useCallback(() => {
        if (videoRef.current) {
            onDurationChange(videoRef.current.duration);
            onVideoStateChange({ duration: videoRef.current.duration });
        }
    }, [onDurationChange, onVideoStateChange]);

    const handlePlay = useCallback(() => {
        onVideoStateChange({ isPlaying: true });
    }, [onVideoStateChange]);

    const handlePause = useCallback(() => {
        onVideoStateChange({ isPlaying: false });
    }, [onVideoStateChange]);

    // Get active detections for current time
    const activeDetections = detections.filter(
        d => d.isRedacted && currentTime >= d.startTime && currentTime <= d.endTime
    );

    // Debug: log active detection IDs
    if (activeDetections.length > 0) {
        console.log('[VideoPlayer] Active detections:', activeDetections.map(d => ({ id: d.id, type: d.type, bbox: d.bbox })));
    }

    // Get interpolated bbox for current time based on framePositions (for motion tracking)
    const getInterpolatedBbox = (detection: Detection): { x: number; y: number; width: number; height: number } => {
        // If no frame positions, use static bbox
        if (!detection.framePositions?.length) {
            return detection.bbox;
        }

        // Estimate current frame (use 30fps as default, could be improved with actual video fps)
        const fps = 30;
        const currentFrame = Math.floor(currentTime * fps);
        const positions = detection.framePositions;

        // Find surrounding positions for interpolation
        let prev = positions[0];
        let next = positions[positions.length - 1];

        for (let i = 0; i < positions.length; i++) {
            const pos = positions[i];
            if (pos[0] <= currentFrame) {
                prev = pos;
            }
            if (pos[0] >= currentFrame) {
                next = pos;
                break;
            }
        }

        // If same position or no interpolation needed
        if (prev[0] === next[0]) {
            return { x: prev[1], y: prev[2], width: prev[3], height: prev[4] };
        }

        // Linear interpolation between positions
        const t = Math.max(0, Math.min(1, (currentFrame - prev[0]) / (next[0] - prev[0])));
        return {
            x: prev[1] + t * (next[1] - prev[1]),
            y: prev[2] + t * (next[2] - prev[2]),
            width: prev[3] + t * (next[3] - prev[3]),
            height: prev[4] + t * (next[4] - prev[4])
        };
    };

    // Calculate pixel position from a bbox
    const getPixelPositionFromBbox = (bbox: { x: number; y: number; width: number; height: number }) => {
        if (!containerRef.current || !videoRef.current) return null;

        const video = videoRef.current;
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const displayWidth = video.clientWidth;
        const displayHeight = video.clientHeight;

        // Calculate scale and offset for letterboxing/pillarboxing
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = videoWidth * scale;
        const scaledHeight = videoHeight * scale;
        const offsetX = (displayWidth - scaledWidth) / 2;
        const offsetY = (displayHeight - scaledHeight) / 2;

        return {
            left: offsetX + bbox.x * scaledWidth,
            top: offsetY + bbox.y * scaledHeight,
            width: bbox.width * scaledWidth,
            height: bbox.height * scaledHeight
        };
    };

    // Calculate pixel position from normalized detection bbox (with interpolation support)
    const getPixelPosition = (detection: Detection) => {
        const bbox = getInterpolatedBbox(detection);
        return getPixelPositionFromBbox(bbox);
    };

    return (
        <div className="video-player">
            <div className="video-container" ref={containerRef} onClick={(e) => {
                // Only deselect if clicking directly on the container (not on child elements)
                if (e.target === e.currentTarget || (e.target as HTMLElement).tagName === 'VIDEO') {
                    onDeselect?.();
                }
            }}>
                <video
                    ref={videoRef}
                    src={videoUrl}
                    onTimeUpdate={handleTimeUpdate}
                    onLoadedMetadata={handleLoadedMetadata}
                    onPlay={handlePlay}
                    onPause={handlePause}
                />

                {/* Redaction overlays */}
                <div className="video-overlay">
                    {activeDetections.map(detection => {
                        const pos = getPixelPosition(detection);
                        if (!pos) return null;
                        const isSelected = selectedDetection?.id === detection.id;
                        const isBlackout = detection.type === 'blackout';

                        return (
                            <div
                                key={detection.id}
                                className={`redaction-overlay${isBlackout ? ' blackout' : ''}${previewBlur ? '' : ' no-blur'}${isSelected ? ' selected-highlight' : ''}${onDetectionClick ? ' clickable' : ''}`}
                                style={{
                                    left: `${pos.left}px`,
                                    top: `${pos.top}px`,
                                    width: `${pos.width}px`,
                                    height: `${pos.height}px`,
                                    ...(isBlackout && { backgroundColor: '#000', backdropFilter: 'none' })
                                }}
                                onClick={(e) => {
                                    if (onDetectionClick) {
                                        e.stopPropagation();
                                        onDetectionClick(detection);
                                    }
                                }}
                            />
                        );
                    })}
                </div>

                {/* Selection tool overlay */}
                {onSelectionComplete && (
                    <SelectionOverlay
                        videoRef={videoRef}
                        isSelectionMode={isSelectionMode}
                        onSelectionComplete={onSelectionComplete}
                    />
                )}

                {/* Highlight overlay for selected detection - rendered LAST so resize handles are on top */}
                {/* Highlight overlay for selected detection (Visual Layer) */}
                {selectedDetection && (() => {
                    const pos = getPixelPosition(selectedDetection);
                    if (!pos) return null;
                    return (
                        <>
                            {/* 1. Visual Highlight Ring (Animated, pointer-events: none to pass through) */}
                            <div
                                className="selection-highlight-overlay"
                                style={{
                                    left: `${pos.left}px`,
                                    top: `${pos.top}px`,
                                    width: `${pos.width}px`,
                                    height: `${pos.height}px`,
                                    pointerEvents: 'none' // Ensure the visual layer doesn't block interactions
                                }}
                            />

                            {/* 2. Interactive Resize Controls (Stable, High Z-Index, No Animation) */}
                            {onBboxChange && (
                                <div
                                    className="resize-controls-group"
                                    style={{
                                        position: 'absolute',
                                        left: `${pos.left}px`,
                                        top: `${pos.top}px`,
                                        width: `${pos.width}px`,
                                        height: `${pos.height}px`,
                                        zIndex: 10000, // Extremely high z-index
                                        pointerEvents: 'none' // Container passes events, children capture them
                                    }}
                                >
                                    <div className="resize-handle nw" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'nw')} />
                                    <div className="resize-handle n" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'n')} />
                                    <div className="resize-handle ne" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'ne')} />
                                    <div className="resize-handle e" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'e')} />
                                    <div className="resize-handle se" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'se')} />
                                    <div className="resize-handle s" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 's')} />
                                    <div className="resize-handle sw" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'sw')} />
                                    <div className="resize-handle w" style={{ pointerEvents: 'auto' }} onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'w')} />
                                </div>
                            )}
                        </>
                    );
                })()}
            </div>
        </div>
    );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
