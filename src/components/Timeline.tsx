import React, { useCallback } from 'react';
import { Detection, PIIType, PII_CATEGORIES, formatTime } from '../types';

export interface ScanZone {
    id: string;
    startTime: number;
    endTime: number;
}

interface TimelineProps {
    detections: Detection[];
    duration: number;
    currentTime: number;
    onSeek: (time: number) => void;
    onDetectionClick: (detection: Detection) => void;
    selectedDetectionId: string | null;
    // Callback to update detection start/end times (for draggable edges)
    onDetectionTimeChange?: (detectionId: string, startTime: number, endTime: number) => void;
    // Callback for toggle redaction (unblur/blur)
    onToggleRedaction?: (detectionId: string) => void;
    // Callback for deleting detection
    onDeleteDetection?: (detectionId: string) => void;
    // Scan zones
    scanZones?: ScanZone[];
    onZonesChange?: (zones: ScanZone[]) => void;
}

// Group detections by type
type DetectionGroup = {
    type: PIIType;
    detections: Detection[];
};

export const Timeline: React.FC<TimelineProps> = ({
    detections,
    duration,
    currentTime,
    onSeek,
    onDetectionClick,
    selectedDetectionId,
    onDetectionTimeChange,
    onToggleRedaction,
    onDeleteDetection,
    scanZones = [],
    onZonesChange
}) => {
    // Context menu state
    const [contextMenu, setContextMenu] = React.useState<{
        x: number;
        y: number;
        detection: Detection;
    } | null>(null);

    // Drag state for resizing segments
    const [dragState, setDragState] = React.useState<{
        detectionId: string;
        edge: 'start' | 'end';
        initialTime: number;
        trackRect: DOMRect | null;
    } | null>(null);

    // Handle mouse move during drag
    React.useEffect(() => {
        if (!dragState || !onDetectionTimeChange) return;

        const handleMouseMove = (e: MouseEvent) => {
            if (!dragState.trackRect) return;

            const percent = Math.max(0, Math.min(1,
                (e.clientX - dragState.trackRect.left) / dragState.trackRect.width
            ));
            const newTime = percent * duration;

            // Find the detection being dragged
            const detection = detections.find(d => d.id === dragState.detectionId);
            if (!detection) return;

            if (dragState.edge === 'start') {
                // Don't let start go past end
                const clampedStart = Math.min(newTime, detection.endTime - 0.1);
                onDetectionTimeChange(detection.id, Math.max(0, clampedStart), detection.endTime);
            } else {
                // Don't let end go before start
                const clampedEnd = Math.max(newTime, detection.startTime + 0.1);
                onDetectionTimeChange(detection.id, detection.startTime, Math.min(duration, clampedEnd));
            }
        };

        const handleMouseUp = () => {
            setDragState(null);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('mouseup', handleMouseUp);
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [dragState, duration, detections, onDetectionTimeChange]);

    // Start dragging an edge
    const handleEdgeDragStart = useCallback((
        e: React.MouseEvent,
        detection: Detection,
        edge: 'start' | 'end',
        trackElement: HTMLElement | null
    ) => {
        e.stopPropagation();
        e.preventDefault();
        if (!trackElement || !onDetectionTimeChange) return;

        setDragState({
            detectionId: detection.id,
            edge,
            initialTime: edge === 'start' ? detection.startTime : detection.endTime,
            trackRect: trackElement.getBoundingClientRect()
        });
    }, [onDetectionTimeChange]);

    // Group detections by type
    const groups: DetectionGroup[] = [];
    const typeMap = new Map<PIIType, Detection[]>();

    detections.forEach(d => {
        if (!typeMap.has(d.type)) {
            typeMap.set(d.type, []);
        }
        typeMap.get(d.type)!.push(d);
    });

    typeMap.forEach((dets, type) => {
        groups.push({ type, detections: dets });
    });

    // Sort groups by category order
    const typeOrder: PIIType[] = ['anchor', 'watchlist', 'email', 'phone', 'credit-card', 'password', 'name', 'address', 'manual'];
    groups.sort((a, b) => typeOrder.indexOf(a.type) - typeOrder.indexOf(b.type));

    // Calculate playhead position
    const playheadPosition = duration > 0 ? (currentTime / duration) * 100 : 0;

    const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        onSeek(percent * duration);
    };

    // Add zone at current time (10% of duration centered on current time)
    const handleAddZone = useCallback(() => {
        if (!onZonesChange) return;
        const zoneDuration = duration * 0.1;  // 10% of video
        const start = Math.max(0, currentTime - zoneDuration / 2);
        const end = Math.min(duration, currentTime + zoneDuration / 2);
        const newZone: ScanZone = {
            id: `zone-${Date.now()}`,
            startTime: start,
            endTime: end
        };
        onZonesChange([...scanZones, newZone]);
    }, [scanZones, currentTime, duration, onZonesChange]);

    // Delete a zone
    const handleDeleteZone = useCallback((zoneId: string) => {
        if (!onZonesChange) return;
        onZonesChange(scanZones.filter(z => z.id !== zoneId));
    }, [scanZones, onZonesChange]);

    return (
        <>
            <div className="timeline">
                <div className="timeline-header">
                    <span className="timeline-title">Redaction Timeline</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                        {onZonesChange && (
                            <button
                                className="btn btn-sm"
                                onClick={handleAddZone}
                                title="Add scan zone at current time"
                            >
                                + Zone
                            </button>
                        )}
                        <span className="text-muted text-mono" style={{ fontSize: 'var(--font-size-xs)' }}>
                            {detections.length} detection{detections.length !== 1 ? 's' : ''}
                        </span>
                    </div>
                </div>

                {/* Scan Zones Track */}
                {onZonesChange && (
                    <div className="timeline-zones-track">
                        <div className="timeline-track-label">
                            <span>🎯 Scan Zones ({scanZones.length || 'All'})</span>
                        </div>
                        <div className="timeline-track-bar" onClick={handleTrackClick}>
                            {/* Playhead */}
                            <div className="timeline-playhead" style={{ left: `${playheadPosition}%` }} />

                            {/* Zone segments */}
                            {scanZones.map(zone => {
                                const startPercent = (zone.startTime / duration) * 100;
                                const widthPercent = ((zone.endTime - zone.startTime) / duration) * 100;
                                return (
                                    <div
                                        key={zone.id}
                                        className="timeline-zone-segment"
                                        style={{
                                            left: `${startPercent}%`,
                                            width: `${Math.max(widthPercent, 1)}%`
                                        }}
                                        title={`${formatTime(zone.startTime)} - ${formatTime(zone.endTime)}`}
                                    >
                                        <button
                                            className="zone-delete-btn"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleDeleteZone(zone.id);
                                            }}
                                        >
                                            ×
                                        </button>
                                    </div>
                                );
                            })}

                            {/* Show empty zone hint */}
                            {scanZones.length === 0 && (
                                <div className="timeline-zone-hint">
                                    No zones = scan entire video
                                </div>
                            )}
                        </div>
                    </div>
                )}

                <div className="timeline-tracks">
                    {groups.length === 0 ? (
                        <div className="empty-state" style={{ padding: 'var(--space-lg)' }}>
                            <div className="empty-state-icon">📊</div>
                            <div className="empty-state-text">
                                No detections yet. Load a video to start analysis.
                            </div>
                        </div>
                    ) : (
                        groups.map(group => {
                            const category = PII_CATEGORIES[group.type];

                            return (
                                <div key={group.type} className="timeline-track">
                                    <div className="timeline-track-label">
                                        <span
                                            className={`timeline-track-label-icon ${group.type}`}
                                        />
                                        <span>{category.icon} {category.label} ({group.detections.length})</span>
                                    </div>

                                    {(() => {
                                        // Calculate rows for overlapping segments
                                        const segmentRows: Map<string, number> = new Map();
                                        const sortedDetections = [...group.detections].sort((a, b) => a.startTime - b.startTime);

                                        for (const det of sortedDetections) {
                                            // Find the first row where this segment doesn't overlap
                                            let row = 0;
                                            let placed = false;
                                            while (!placed) {
                                                const rowOccupied = sortedDetections.some(other => {
                                                    if (other.id === det.id) return false;
                                                    const otherRow = segmentRows.get(other.id);
                                                    if (otherRow !== row) return false;
                                                    // Check for time overlap
                                                    return !(det.endTime <= other.startTime || det.startTime >= other.endTime);
                                                });
                                                if (!rowOccupied) {
                                                    segmentRows.set(det.id, row);
                                                    placed = true;
                                                } else {
                                                    row++;
                                                }
                                            }
                                        }

                                        const rowHeight = 18; // px per row
                                        const maxRow = Math.max(0, ...Array.from(segmentRows.values()));
                                        const trackHeight = (maxRow + 1) * rowHeight;

                                        return (
                                            <div
                                                className="timeline-track-bar"
                                                onClick={handleTrackClick}
                                                style={{ minHeight: `${Math.max(18, trackHeight)}px` }}
                                            >
                                                {/* Playhead */}
                                                <div
                                                    className="timeline-playhead"
                                                    style={{ left: `${playheadPosition}%` }}
                                                />

                                                {/* Detection segments */}
                                                {group.detections.map(detection => {
                                                    const startPercent = (detection.startTime / duration) * 100;
                                                    const widthPercent = ((detection.endTime - detection.startTime) / duration) * 100;
                                                    const row = segmentRows.get(detection.id) || 0;

                                                    return (
                                                        <div
                                                            key={detection.id}
                                                            className={`timeline-track-segment ${group.type} ${detection.id === selectedDetectionId ? 'selected' : ''} ${dragState?.detectionId === detection.id ? 'dragging' : ''} ${!detection.isRedacted ? 'unblurred' : ''}`}
                                                            style={{
                                                                left: `${startPercent}%`,
                                                                width: `${Math.max(widthPercent, 0.5)}%`,
                                                                top: `${row * rowHeight}px`,
                                                                height: `${rowHeight - 2}px`
                                                            }}
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setContextMenu(null);
                                                                onSeek(detection.startTime);
                                                                onDetectionClick(detection);
                                                            }}
                                                            onContextMenu={(e) => {
                                                                e.preventDefault();
                                                                e.stopPropagation();
                                                                setContextMenu({ x: e.clientX, y: e.clientY, detection });
                                                            }}
                                                            title={`${detection.content}\n${formatTime(detection.startTime)} - ${formatTime(detection.endTime)}`}
                                                        >
                                                            {/* Left resize handle */}
                                                            {onDetectionTimeChange && (
                                                                <div
                                                                    className="segment-resize-handle left"
                                                                    onMouseDown={(e) => handleEdgeDragStart(e, detection, 'start', e.currentTarget.closest('.timeline-track-bar'))}
                                                                />
                                                            )}
                                                            {/* Right resize handle */}
                                                            {onDetectionTimeChange && (
                                                                <div
                                                                    className="segment-resize-handle right"
                                                                    onMouseDown={(e) => handleEdgeDragStart(e, detection, 'end', e.currentTarget.closest('.timeline-track-bar'))}
                                                                />
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        );
                                    })()}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Context Menu */}
            {contextMenu && (
                <>
                    <div
                        className="context-menu-backdrop"
                        onClick={() => setContextMenu(null)}
                    />
                    <div
                        className="context-menu"
                        style={{
                            position: 'fixed',
                            left: contextMenu.x,
                            top: contextMenu.y
                        }}
                    >
                        {onToggleRedaction && (
                            <button
                                className="context-menu-item"
                                onClick={() => {
                                    onToggleRedaction(contextMenu.detection.id);
                                    setContextMenu(null);
                                }}
                            >
                                {contextMenu.detection.isRedacted ? '👁️ Un-blur' : '🔒 Blur'}
                            </button>
                        )}
                        {onDeleteDetection && (
                            <button
                                className="context-menu-item danger"
                                onClick={() => {
                                    onDeleteDetection(contextMenu.detection.id);
                                    setContextMenu(null);
                                }}
                            >
                                🗑️ Delete
                            </button>
                        )}
                    </div>
                </>
            )}
        </>
    );
};

export default Timeline;
