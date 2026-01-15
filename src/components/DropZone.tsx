import React, { useCallback, useState, useEffect } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { getCurrentWebview } from '@tauri-apps/api/webview';

interface DropZoneProps {
    onFileSelect: (file: File) => void;
    onPathSelect?: (path: string) => void;
    onOpenProject?: () => void;
}

const FORMAT_LABELS = ['MP4', 'WebM', 'MOV', 'AVI'];

export const DropZone: React.FC<DropZoneProps> = ({ onFileSelect, onPathSelect, onOpenProject }) => {
    const [isDragOver, setIsDragOver] = useState(false);

    // Listen for Tauri's native drag-drop event to get real file paths
    useEffect(() => {
        let unlisten: (() => void) | undefined;

        const setupDragDropListener = async () => {
            try {
                const webview = getCurrentWebview();
                unlisten = await webview.onDragDropEvent((event) => {
                    if (event.payload.type === 'enter') {
                        setIsDragOver(true);
                    } else if (event.payload.type === 'drop') {
                        setIsDragOver(false);
                        const paths = event.payload.paths;
                        if (paths.length > 0) {
                            const filePath = paths[0];
                            // Check if it's a video file
                            if (filePath.match(/\.(mp4|webm|mov|avi)$/i)) {
                                console.log('[DropZone] Tauri drop event - file path:', filePath);
                                if (onPathSelect) {
                                    onPathSelect(filePath);
                                } else {
                                    // Fallback: create a synthetic file object
                                    const fileName = filePath.split(/[/\\]/).pop() || 'video.mp4';
                                    const syntheticFile = new File([], fileName, { type: 'video/mp4' });
                                    (syntheticFile as any).path = filePath;
                                    onFileSelect(syntheticFile);
                                }
                            } else {
                                alert('Please drop a video file (MP4, WebM, MOV, or AVI)');
                            }
                        }
                    }
                });
            } catch (err) {
                console.error('[DropZone] Failed to set up Tauri drag-drop listener:', err);
            }
        };

        setupDragDropListener();

        return () => {
            if (unlisten) {
                unlisten();
            }
        };
    }, [onFileSelect, onPathSelect]);

    // Browser drag-over/leave for visual feedback (actual drop handled by Tauri)
    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        // Visual state is now handled by Tauri event, but keep this for fallback
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        // Use browser event to detect when drag leaves the window
        // since Tauri 2.0 doesn't provide a 'leave' event
        setIsDragOver(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        // Note: Actual file handling is done by Tauri's onDragDropEvent
        // This is just to prevent browser default behavior
    }, []);

    const handleBrowseClick = useCallback(async () => {
        console.log('[DropZone] Browse clicked, attempting Tauri dialog...');

        try {
            // Use Tauri's dialog to get the real file path
            console.log('[DropZone] Calling open() dialog...');
            const selected = await open({
                multiple: false,
                filters: [{
                    name: 'Video',
                    extensions: ['mp4', 'webm', 'mov', 'avi']
                }]
            });

            console.log('[DropZone] Dialog returned:', selected, typeof selected);

            if (selected && typeof selected === 'string') {
                console.log('[DropZone] Selected file path:', selected);

                // Call the path callback if provided (for real AI analysis)
                if (onPathSelect) {
                    console.log('[DropZone] Calling onPathSelect with path');
                    onPathSelect(selected);
                } else {
                    console.log('[DropZone] No onPathSelect, using onFileSelect fallback');
                    // Fallback: create a synthetic file object
                    const fileName = selected.split(/[/\\]/).pop() || 'video.mp4';
                    const syntheticFile = new File([], fileName, { type: 'video/mp4' });
                    // Attach the path to the file object
                    (syntheticFile as any).path = selected;
                    onFileSelect(syntheticFile);
                }
            } else if (selected === null) {
                console.log('[DropZone] Dialog cancelled by user');
            }
        } catch (err) {
            console.error('[DropZone] Dialog error:', err);
            // Fall back to HTML file input
            console.log('[DropZone] Falling back to HTML file input...');
            const input = document.getElementById('file-input') as HTMLInputElement;
            input?.click();
        }
    }, [onFileSelect, onPathSelect]);

    const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            onFileSelect(files[0]);
        }
    }, [onFileSelect]);

    return (
        <div className="drop-zone-container">
            <div
                className={`drop-zone ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
            >
                <input
                    id="file-input"
                    type="file"
                    accept="video/*"
                    onChange={handleFileInput}
                    style={{ display: 'none' }}
                />

                <div className="drop-zone-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17,8 12,3 7,8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                </div>

                <div className="drop-zone-text">
                    <div className="drop-zone-title">
                        Drop your video here
                    </div>
                    <div className="drop-zone-subtitle">
                        or <strong onClick={handleBrowseClick}>browse</strong> to select a file
                    </div>
                </div>

                <div className="drop-zone-formats">
                    {FORMAT_LABELS.map(format => (
                        <span key={format} className="drop-zone-format">{format}</span>
                    ))}
                </div>

                {onOpenProject && (
                    <div className="drop-zone-divider">
                        <span>or</span>
                    </div>
                )}

                {onOpenProject && (
                    <button className="btn btn-ghost" onClick={onOpenProject}>
                        📁 Open Existing Project
                    </button>
                )}
            </div>
        </div>
    );
};

export default DropZone;
