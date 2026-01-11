import React from 'react';
import { VideoState, formatTime } from '../types';

interface VideoControlsProps {
    videoState: VideoState;
    onPlay: () => void;
    onPause: () => void;
    onSeek: (time: number) => void;
    onMagicWandClick: () => void;
    isMagicWandActive: boolean;
    onBlackoutClick?: () => void;
    isBlackoutActive?: boolean;
}

export const VideoControls: React.FC<VideoControlsProps> = ({
    videoState,
    onPlay,
    onPause,
    onSeek,
    onMagicWandClick,
    isMagicWandActive,
    onBlackoutClick,
    isBlackoutActive = false
}) => {
    const { isPlaying, currentTime, duration } = videoState;
    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    const handleSeekBarClick = (e: React.MouseEvent<HTMLDivElement>) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        onSeek(percent * duration);
    };

    const handleSkipBackward = () => {
        onSeek(Math.max(0, currentTime - 5));
    };

    const handleSkipForward = () => {
        onSeek(Math.min(duration, currentTime + 5));
    };

    return (
        <div className="video-controls">
            {/* Left side - Magic Wand and Blackout tools */}
            <div className="video-controls-group">
                <button
                    className={`btn btn-icon ${isMagicWandActive ? 'btn-primary' : 'btn-ghost'}`}
                    onClick={onMagicWandClick}
                    title="Magic Wand - Blur with motion tracking"
                >
                    🪄
                </button>
                {onBlackoutClick && (
                    <button
                        className={`btn btn-icon ${isBlackoutActive ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={onBlackoutClick}
                        title="Blackout - Solid black overlay with motion tracking"
                        style={isBlackoutActive ? { background: '#000', borderColor: '#000' } : {}}
                    >
                        ⬛
                    </button>
                )}
            </div>

            {/* Center - Playback controls */}
            <div className="video-controls-center">
                <button
                    className="btn btn-ghost btn-icon"
                    onClick={handleSkipBackward}
                    title="Skip 5s backward"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12.5 3C17.15 3 21.08 6.03 22.47 10.22L20.1 11C19.05 7.81 16.04 5.5 12.5 5.5C10.54 5.5 8.77 6.22 7.38 7.38L10 10H3V3L5.6 5.6C7.45 4 9.85 3 12.5 3M10 12V22H8V14H6V12H10M18 14V20C18 21.11 17.11 22 16 22H14C12.9 22 12 21.1 12 20V14C12 12.9 12.9 12 14 12H16C17.11 12 18 12.9 18 14M14 14V20H16V14H14Z" />
                    </svg>
                </button>

                <button
                    className="btn btn-primary btn-icon"
                    onClick={isPlaying ? onPause : onPlay}
                    title={isPlaying ? 'Pause' : 'Play'}
                >
                    {isPlaying ? (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                        </svg>
                    ) : (
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5v14l11-7z" />
                        </svg>
                    )}
                </button>

                <button
                    className="btn btn-ghost btn-icon"
                    onClick={handleSkipForward}
                    title="Skip 5s forward"
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M11.5 3C6.85 3 2.92 6.03 1.53 10.22L3.9 11C4.95 7.81 7.96 5.5 11.5 5.5C13.46 5.5 15.23 6.22 16.62 7.38L14 10H21V3L18.4 5.6C16.55 4 14.15 3 11.5 3M10 12V22H8V14H6V12H10M18 14V20C18 21.11 17.11 22 16 22H14C12.9 22 12 21.1 12 20V14C12 12.9 12.9 12 14 12H16C17.11 12 18 12.9 18 14M14 14V20H16V14H14Z" />
                    </svg>
                </button>

                <div className="video-time">
                    {formatTime(currentTime)} / {formatTime(duration)}
                </div>

                <div className="video-seekbar" onClick={handleSeekBarClick}>
                    <div
                        className="video-seekbar-fill"
                        style={{ width: `${progress}%` }}
                    />
                    <div
                        className="video-seekbar-thumb"
                        style={{ left: `${progress}%` }}
                    />
                </div>
            </div>
        </div>
    );
};

export default VideoControls;
