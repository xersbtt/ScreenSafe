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
    // Callback for clicking/selecting a detection from the video overlay
    onDetectionClick?: (detection: Detection) => void;
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
    onDetectionClick
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
        if (!resizeDrag || !onBboxChange || !videoRef.current) return;

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
            setResizeDrag(null);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [resizeDrag, onBboxChange]);

    const handleResizeStart = useCallback((e: React.MouseEvent, detection: Detection, handle: ResizeHandle) => {
        e.stopPropagation();
        e.preventDefault();
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

    // Calculate pixel position from normalized bbox
    const getPixelPosition = (detection: Detection) => {
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
            left: offsetX + detection.bbox.x * scaledWidth,
            top: offsetY + detection.bbox.y * scaledHeight,
            width: detection.bbox.width * scaledWidth,
            height: detection.bbox.height * scaledHeight
        };
    };

    return (
        <div className="video-player">
            <div className="video-container" ref={containerRef}>
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
                    {/* Highlight overlay for selected detection (even if not in current time range) */}
                    {selectedDetection && (() => {
                        const pos = getPixelPosition(selectedDetection);
                        if (!pos) return null;
                        return (
                            <div
                                className="selection-highlight-overlay"
                                style={{
                                    left: `${pos.left}px`,
                                    top: `${pos.top}px`,
                                    width: `${pos.width}px`,
                                    height: `${pos.height}px`
                                }}
                            >
                                {/* Resize handles - only show when onBboxChange is available */}
                                {onBboxChange && (
                                    <>
                                        <div className="resize-handle nw" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'nw')} />
                                        <div className="resize-handle n" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'n')} />
                                        <div className="resize-handle ne" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'ne')} />
                                        <div className="resize-handle e" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'e')} />
                                        <div className="resize-handle se" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'se')} />
                                        <div className="resize-handle s" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 's')} />
                                        <div className="resize-handle sw" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'sw')} />
                                        <div className="resize-handle w" onMouseDown={(e) => handleResizeStart(e, selectedDetection, 'w')} />
                                    </>
                                )}
                            </div>
                        );
                    })()}
                </div>

                {/* Selection tool overlay */}
                {onSelectionComplete && (
                    <SelectionOverlay
                        videoRef={videoRef}
                        isSelectionMode={isSelectionMode}
                        onSelectionComplete={onSelectionComplete}
                    />
                )}
            </div>
        </div>
    );
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
