import { useState, useRef, useCallback, useEffect } from 'react';

interface SelectionOverlayProps {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    isSelectionMode: boolean;
    onSelectionComplete: (bbox: { x: number; y: number; width: number; height: number }, frameTime: number) => void;
}

interface SelectionBox {
    startX: number;
    startY: number;
    endX: number;
    endY: number;
}

export function SelectionOverlay({ videoRef, isSelectionMode, onSelectionComplete }: SelectionOverlayProps) {
    const overlayRef = useRef<HTMLDivElement>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [selection, setSelection] = useState<SelectionBox | null>(null);
    const [confirmedSelection, setConfirmedSelection] = useState<SelectionBox | null>(null);

    // Get mouse position relative to video
    const getRelativePosition = useCallback((e: React.MouseEvent) => {
        if (!overlayRef.current || !videoRef.current) return { x: 0, y: 0, displayWidth: 0, displayHeight: 0 };

        const rect = overlayRef.current.getBoundingClientRect();
        const video = videoRef.current;

        // Get actual video display area (accounting for letterboxing)
        const videoAspect = video.videoWidth / video.videoHeight;
        const containerAspect = rect.width / rect.height;

        let displayWidth: number, displayHeight: number, offsetX: number, offsetY: number;

        if (videoAspect > containerAspect) {
            displayWidth = rect.width;
            displayHeight = rect.width / videoAspect;
            offsetX = 0;
            offsetY = (rect.height - displayHeight) / 2;
        } else {
            displayHeight = rect.height;
            displayWidth = rect.height * videoAspect;
            offsetX = (rect.width - displayWidth) / 2;
            offsetY = 0;
        }

        const mouseX = e.clientX - rect.left - offsetX;
        const mouseY = e.clientY - rect.top - offsetY;

        const x = Math.max(0, Math.min(displayWidth, mouseX));
        const y = Math.max(0, Math.min(displayHeight, mouseY));

        return { x, y, displayWidth, displayHeight };
    }, [videoRef]);

    // Convert pixel selection to normalized bbox (0-1)
    const selectionToNormalizedBbox = useCallback((sel: SelectionBox) => {
        if (!videoRef.current || !overlayRef.current) return null;

        const video = videoRef.current;
        const rect = overlayRef.current.getBoundingClientRect();

        const videoAspect = video.videoWidth / video.videoHeight;
        const containerAspect = rect.width / rect.height;

        let displayWidth: number, displayHeight: number;

        if (videoAspect > containerAspect) {
            displayWidth = rect.width;
            displayHeight = rect.width / videoAspect;
        } else {
            displayHeight = rect.height;
            displayWidth = rect.height * videoAspect;
        }

        const x1 = Math.min(sel.startX, sel.endX) / displayWidth;
        const y1 = Math.min(sel.startY, sel.endY) / displayHeight;
        const x2 = Math.max(sel.startX, sel.endX) / displayWidth;
        const y2 = Math.max(sel.startY, sel.endY) / displayHeight;

        return {
            x: Math.max(0, Math.min(1, x1)),
            y: Math.max(0, Math.min(1, y1)),
            width: Math.max(0, Math.min(1 - x1, x2 - x1)),
            height: Math.max(0, Math.min(1 - y1, y2 - y1))
        };
    }, [videoRef]);

    const handleMouseDown = useCallback((e: React.MouseEvent) => {
        if (!isSelectionMode) return;

        e.preventDefault();
        const pos = getRelativePosition(e);

        setIsDrawing(true);
        setSelection({
            startX: pos.x,
            startY: pos.y,
            endX: pos.x,
            endY: pos.y
        });
        setConfirmedSelection(null);
    }, [isSelectionMode, getRelativePosition]);

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (!isDrawing || !selection) return;

        const pos = getRelativePosition(e);
        setSelection(prev => prev ? { ...prev, endX: pos.x, endY: pos.y } : null);
    }, [isDrawing, selection, getRelativePosition]);

    const handleMouseUp = useCallback(() => {
        if (!isDrawing || !selection) return;

        setIsDrawing(false);

        const width = Math.abs(selection.endX - selection.startX);
        const height = Math.abs(selection.endY - selection.startY);

        if (width > 10 && height > 10) {
            // Auto-confirm: directly call onSelectionComplete instead of showing buttons
            const bbox = selectionToNormalizedBbox(selection);
            if (bbox && videoRef.current) {
                console.log('[SelectionOverlay] Auto-confirming selection:', bbox);
                onSelectionComplete(bbox, videoRef.current.currentTime);
            }
            setSelection(null);
        } else {
            setSelection(null);
        }
    }, [isDrawing, selection, selectionToNormalizedBbox, videoRef, onSelectionComplete]);

    const handleConfirmSelection = useCallback(() => {
        if (!confirmedSelection || !videoRef.current) return;

        const bbox = selectionToNormalizedBbox(confirmedSelection);
        if (bbox) {
            onSelectionComplete(bbox, videoRef.current.currentTime);
        }

        setConfirmedSelection(null);
        setSelection(null);
    }, [confirmedSelection, videoRef, selectionToNormalizedBbox, onSelectionComplete]);

    const handleCancelSelection = useCallback(() => {
        setConfirmedSelection(null);
        setSelection(null);
    }, []);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') handleCancelSelection();
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [handleCancelSelection]);

    const getSelectionStyle = (sel: SelectionBox): React.CSSProperties => {
        const left = Math.min(sel.startX, sel.endX);
        const top = Math.min(sel.startY, sel.endY);
        const width = Math.abs(sel.endX - sel.startX);
        const height = Math.abs(sel.endY - sel.startY);

        if (!overlayRef.current || !videoRef.current) return {};

        const rect = overlayRef.current.getBoundingClientRect();
        const video = videoRef.current;
        const videoAspect = video.videoWidth / video.videoHeight;
        const containerAspect = rect.width / rect.height;

        let offsetX: number, offsetY: number;

        if (videoAspect > containerAspect) {
            offsetX = 0;
            offsetY = (rect.height - rect.width / videoAspect) / 2;
        } else {
            offsetX = (rect.width - rect.height * videoAspect) / 2;
            offsetY = 0;
        }

        return {
            left: `${left + offsetX}px`,
            top: `${top + offsetY}px`,
            width: `${width}px`,
            height: `${height}px`
        };
    };

    const currentStyle = selection ? getSelectionStyle(selection) : {};
    const confirmedStyle = confirmedSelection ? getSelectionStyle(confirmedSelection) : {};

    return (
        <div
            ref={overlayRef}
            className={`selection-overlay ${isSelectionMode ? 'active' : ''}`}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
        >
            {isDrawing && selection && (
                <div className="selection-box drawing" style={currentStyle} />
            )}

            {confirmedSelection && !isDrawing && (
                <>
                    <div className="selection-box confirmed" style={confirmedStyle} />
                    <div
                        className="selection-buttons"
                        style={{
                            position: 'absolute',
                            left: confirmedStyle.left,
                            top: `calc(${confirmedStyle.top} + ${confirmedStyle.height} + 8px)`,
                            pointerEvents: 'auto'
                        }}
                        onMouseDown={(e) => e.stopPropagation()}
                    >
                        <button
                            className="btn btn-primary btn-sm"
                            onClick={(e) => {
                                e.stopPropagation();
                                console.log('[SelectionOverlay] Add Blur clicked');
                                console.log('[SelectionOverlay] confirmedSelection:', confirmedSelection);
                                console.log('[SelectionOverlay] videoRef.current:', videoRef.current);
                                handleConfirmSelection();
                            }}
                        >
                            ✓ Add Blur
                        </button>
                        <button
                            className="btn btn-ghost btn-sm"
                            onClick={(e) => {
                                e.stopPropagation();
                                handleCancelSelection();
                            }}
                        >
                            ✕ Cancel
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}

export default SelectionOverlay;
