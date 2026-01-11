import React, { useRef, useEffect } from 'react';
import { Detection, PII_CATEGORIES, formatTime } from '../types';

interface DetectionSidebarProps {
    detections: Detection[];
    selectedDetectionId: string | null;
    onDetectionSelect: (detection: Detection) => void;
    onToggleRedaction: (detectionId: string) => void;
    onDeleteDetection: (detectionId: string) => void;
}

export const DetectionSidebar: React.FC<DetectionSidebarProps> = ({
    detections,
    selectedDetectionId,
    onDetectionSelect,
    onToggleRedaction,
    onDeleteDetection
}) => {
    // Refs for detection card elements to enable scroll-to-view
    const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

    // Auto-scroll to selected detection when selection changes
    useEffect(() => {
        if (selectedDetectionId) {
            const cardElement = cardRefs.current.get(selectedDetectionId);
            if (cardElement) {
                cardElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'nearest'
                });
            }
        }
    }, [selectedDetectionId]);

    // Sort by start time
    const sortedDetections = [...detections].sort((a, b) => a.startTime - b.startTime);

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <span className="sidebar-title">Detections</span>
            </div>

            <div className="sidebar-content">
                {sortedDetections.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">🔍</div>
                        <div className="empty-state-text">
                            No sensitive data detected yet
                        </div>
                    </div>
                ) : (
                    sortedDetections.map(detection => {
                        const category = PII_CATEGORIES[detection.type];
                        const isSelected = detection.id === selectedDetectionId;

                        return (
                            <div
                                key={detection.id}
                                ref={(el) => {
                                    if (el) {
                                        cardRefs.current.set(detection.id, el);
                                    } else {
                                        cardRefs.current.delete(detection.id);
                                    }
                                }}
                                className={`detection-card ${isSelected ? 'selected' : ''}`}
                                onClick={() => onDetectionSelect(detection)}
                            >
                                <div className="detection-card-header">
                                    <div className="detection-card-type">
                                        <span className="detection-card-icon">{category.icon}</span>
                                        <span className="detection-card-label">{category.label}</span>
                                    </div>
                                    <span className="detection-card-confidence">
                                        {Math.round(detection.confidence * 100)}%
                                    </span>
                                </div>

                                <div className="detection-card-content">
                                    {detection.type === 'password'
                                        ? '••••••••'
                                        : detection.content.length > 30
                                            ? detection.content.slice(0, 30) + '...'
                                            : detection.content
                                    }
                                </div>

                                <div className="detection-card-time">
                                    {formatTime(detection.startTime)} - {formatTime(detection.endTime)}
                                </div>

                                <div className="detection-card-actions">
                                    <button
                                        className={`btn btn-sm ${detection.isRedacted ? '' : 'btn-primary'}`}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onToggleRedaction(detection.id);
                                        }}
                                    >
                                        {detection.isRedacted ? '👁️ Un-blur' : '🔒 Blur'}
                                    </button>
                                    <button
                                        className="btn btn-sm btn-danger"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onDeleteDetection(detection.id);
                                        }}
                                        title="Delete detection (irreversible)"
                                    >
                                        🗑️
                                    </button>
                                </div>
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
};

export default DetectionSidebar;
