import { useState, useEffect, useRef } from 'react';
import './SetupDialog.css';

interface SetupDialogProps {
    onSetupComplete: () => void;
}

type SetupStage = 'checking' | 'installing' | 'downloading_models' | 'complete' | 'error';

interface SetupProgress {
    stage: SetupStage;
    progress: number;
    message: string;
    detail?: string;
}

export function SetupDialog({ onSetupComplete }: SetupDialogProps) {
    const [setupProgress, setSetupProgress] = useState<SetupProgress>({
        stage: 'checking',
        progress: 0,
        message: 'Checking setup status...',
    });
    const setupStartedRef = useRef(false);

    useEffect(() => {
        // Prevent duplicate setup calls (React Strict Mode runs effects twice)
        if (setupStartedRef.current) return;
        setupStartedRef.current = true;
        startSetup();
    }, []);

    const startSetup = async () => {
        try {
            // Import Tauri commands dynamically
            const { invoke } = await import('@tauri-apps/api/core');

            // Check if setup is already complete
            const isComplete = await invoke<boolean>('check_setup_complete');
            if (isComplete) {
                onSetupComplete();
                return;
            }

            // Start the setup process
            setSetupProgress({
                stage: 'installing',
                progress: 5,
                message: 'Installing dependencies...',
                detail: 'This may take a few minutes on first run',
            });

            // Subscribe to progress updates
            const { listen } = await import('@tauri-apps/api/event');
            const unlisten = await listen<{ progress: number; message: string; detail?: string }>(
                'setup-progress',
                (event) => {
                    setSetupProgress((prev) => ({
                        ...prev,
                        progress: event.payload.progress,
                        message: event.payload.message,
                        detail: event.payload.detail,
                    }));
                }
            );

            // Run the setup
            await invoke('run_setup');

            unlisten();

            // Setup complete
            setSetupProgress({
                stage: 'complete',
                progress: 100,
                message: 'Setup complete!',
                detail: 'Starting ScreenSafe...',
            });

            // Brief delay to show completion
            setTimeout(() => {
                onSetupComplete();
            }, 1000);

        } catch (error) {
            console.error('Setup failed:', error);
            setSetupProgress({
                stage: 'error',
                progress: 0,
                message: 'Setup failed',
                detail: String(error),
            });
        }
    };

    const getStageIcon = () => {
        switch (setupProgress.stage) {
            case 'checking':
                return '🔍';
            case 'installing':
                return '📦';
            case 'downloading_models':
                return '🧠';
            case 'complete':
                return '✅';
            case 'error':
                return '❌';
            default:
                return '⚙️';
        }
    };

    return (
        <div className="setup-dialog-overlay">
            <div className="setup-dialog">
                <div className="setup-icon">{getStageIcon()}</div>
                <h1 className="setup-title">Setting up ScreenSafe</h1>
                <p className="setup-message">{setupProgress.message}</p>

                {setupProgress.stage !== 'error' && (
                    <div className="setup-progress-container">
                        <div className="setup-progress-bar">
                            <div
                                className="setup-progress-fill"
                                style={{ width: `${setupProgress.progress}%` }}
                            />
                        </div>
                        <span className="setup-progress-text">{Math.round(setupProgress.progress)}%</span>
                    </div>
                )}

                {setupProgress.detail && (
                    <p className="setup-detail">{setupProgress.detail}</p>
                )}

                {setupProgress.stage === 'error' && (
                    <button className="setup-retry-btn" onClick={startSetup}>
                        Retry Setup
                    </button>
                )}

                <p className="setup-note">
                    First-time setup installs AI components for text detection.
                    <br />
                    This only happens once.
                </p>
            </div>
        </div>
    );
}
