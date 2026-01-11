import React from 'react';
import { AnalysisProgress } from '../types';

interface AnalysisOverlayProps {
    progress: AnalysisProgress;
    onCancel: () => void;
}

const STAGE_LABELS: Record<AnalysisProgress['stage'], string> = {
    'extracting': 'Extracting frames...',
    'detecting': 'Detecting text & UI elements...',
    'tracking': 'Tracking objects across frames...',
    'classifying': 'Classifying sensitive data...',
    'complete': 'Analysis complete!'
};

export const AnalysisOverlay: React.FC<AnalysisOverlayProps> = ({
    progress,
    onCancel
}) => {
    const formatTime = (seconds: number): string => {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${mins}m ${secs}s`;
    };

    return (
        <div className="analysis-overlay animate-fade-in">
            <div className="analysis-card animate-slide-up">
                <div className="analysis-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <path d="M12 6v6l4 2" />
                    </svg>
                </div>

                <h2 className="analysis-title">Analyzing Video</h2>
                <p className="analysis-subtitle">{STAGE_LABELS[progress.stage]}</p>

                <div className="analysis-progress">
                    <div className="progress-bar">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${progress.progress}%` }}
                        />
                    </div>
                    <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginTop: 'var(--space-sm)',
                        fontSize: 'var(--font-size-sm)',
                        color: 'var(--color-text-secondary)'
                    }}>
                        <span>{Math.round(progress.progress)}%</span>
                        <span>
                            {progress.estimatedTimeRemaining > 0
                                ? `~${formatTime(progress.estimatedTimeRemaining)} remaining`
                                : 'Finishing up...'
                            }
                        </span>
                    </div>
                </div>

                <div className="analysis-stats">
                    <div className="analysis-stat">
                        <div className="analysis-stat-value">{progress.framesProcessed}</div>
                        <div className="analysis-stat-label">Frames Processed</div>
                    </div>
                    <div className="analysis-stat">
                        <div className="analysis-stat-value">{progress.detectionsFound}</div>
                        <div className="analysis-stat-label">Detections Found</div>
                    </div>
                </div>

                <button
                    className="btn"
                    style={{ marginTop: 'var(--space-lg)' }}
                    onClick={onCancel}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
};

export default AnalysisOverlay;
