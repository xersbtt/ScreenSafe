import React from 'react';

export interface AppSettings {
    theme: 'dark' | 'light';
    blurStrength: number;
    autoSaveProjects: boolean;
    ocrLanguage: 'eng' | 'auto';
    exportQuality: 'low' | 'medium' | 'high';
    exportCodec: 'h264' | 'h265' | 'vp9';
    // Export processing settings
    scanInterval: number;
    motionThreshold: number;
    ocrScale: number;
    // Detection settings
    enableRegexPatterns: boolean;  // Auto-detect emails, phones, SSNs etc.
}

interface SettingsModalProps {
    isOpen: boolean;
    settings: AppSettings;
    onSettingsChange: (settings: Partial<AppSettings>) => void;
    onClose: () => void;
}


export const SettingsModal: React.FC<SettingsModalProps> = ({
    isOpen,
    settings,
    onSettingsChange,
    onClose
}) => {
    if (!isOpen) return null;

    // Default values for any missing settings
    const defaults: AppSettings = {
        theme: 'dark',
        blurStrength: 25,
        autoSaveProjects: false,
        ocrLanguage: 'eng',
        exportQuality: 'high',
        exportCodec: 'h264',
        scanInterval: 30,
        motionThreshold: 30,
        ocrScale: 0.75,
        enableRegexPatterns: false,  // OFF by default - anchors are preferred
    };

    // Merge settings with defaults for display
    const safeSettings = { ...defaults, ...settings } as AppSettings;

    const handleChange = (key: keyof AppSettings, value: AppSettings[keyof AppSettings]) => {
        // Pass only the changed field - App.tsx will merge it with current state
        onSettingsChange({ [key]: value });
    };

    return (
        <div className="settings-modal-overlay" onClick={onClose}>
            <div className="settings-modal" onClick={e => e.stopPropagation()}>
                <div className="settings-modal-header">
                    <h2>Settings</h2>
                    <button className="btn btn-ghost btn-icon" onClick={onClose}>
                        ✕
                    </button>
                </div>

                <div className="settings-modal-content">
                    {/* Theme */}
                    <div className="settings-group">
                        <label className="settings-label">Theme</label>
                        <div className="settings-toggle-group">
                            <button
                                className={`settings-toggle ${safeSettings.theme === 'dark' ? 'active' : ''}`}
                                onClick={() => handleChange('theme', 'dark')}
                            >
                                🌙 Dark
                            </button>
                            <button
                                className={`settings-toggle ${safeSettings.theme === 'light' ? 'active' : ''}`}
                                onClick={() => handleChange('theme', 'light')}
                            >
                                ☀️ Light
                            </button>
                        </div>
                    </div>

                    {/* Blur Strength */}
                    <div className="settings-group">
                        <label className="settings-label">
                            Default Blur Strength: {safeSettings.blurStrength}px
                        </label>
                        <input
                            type="range"
                            min="10"
                            max="50"
                            value={safeSettings.blurStrength}
                            onChange={e => handleChange('blurStrength', parseInt(e.target.value))}
                            className="settings-slider"
                        />
                    </div>

                    {/* Auto-save */}
                    <div className="settings-group">
                        <label className="settings-label">Auto-save Projects</label>
                        <button
                            className={`settings-toggle-single ${safeSettings.autoSaveProjects ? 'active' : ''}`}
                            onClick={() => handleChange('autoSaveProjects', !safeSettings.autoSaveProjects)}
                        >
                            {safeSettings.autoSaveProjects ? '✓ Enabled' : 'Disabled'}
                        </button>
                    </div>

                    {/* OCR Language */}
                    <div className="settings-group">
                        <label className="settings-label">OCR Language</label>
                        <select
                            value={safeSettings.ocrLanguage}
                            onChange={e => handleChange('ocrLanguage', e.target.value as 'eng' | 'auto')}
                            className="settings-select"
                        >
                            <option value="eng">English</option>
                            <option value="auto">Auto-detect</option>
                        </select>
                    </div>

                    {/* Export Quality */}
                    <div className="settings-group">
                        <label className="settings-label">Export Quality</label>
                        <select
                            value={safeSettings.exportQuality}
                            onChange={e => handleChange('exportQuality', e.target.value as 'low' | 'medium' | 'high')}
                            className="settings-select"
                        >
                            <option value="low">Low (faster export)</option>
                            <option value="medium">Medium</option>
                            <option value="high">High (best quality)</option>
                        </select>
                    </div>

                    {/* Export Codec */}
                    <div className="settings-group">
                        <label className="settings-label">Export Codec</label>
                        <select
                            value={safeSettings.exportCodec}
                            onChange={e => handleChange('exportCodec', e.target.value as 'h264' | 'h265' | 'vp9')}
                            className="settings-select"
                        >
                            <option value="h264">H.264 (most compatible)</option>
                            <option value="h265">H.265 (smaller files)</option>
                            <option value="vp9">VP9 (web-friendly)</option>
                        </select>
                    </div>

                    {/* Section divider */}
                    <div className="settings-divider">
                        <span>Detection Settings</span>
                    </div>

                    {/* Regex Pattern Detection */}
                    <div className="settings-group">
                        <label className="settings-label">Auto-detect PII Patterns</label>
                        <button
                            className={`settings-toggle-single ${safeSettings.enableRegexPatterns ? 'active' : ''}`}
                            onClick={() => handleChange('enableRegexPatterns', !safeSettings.enableRegexPatterns)}
                        >
                            {safeSettings.enableRegexPatterns ? '✓ Enabled' : 'Disabled'}
                        </button>
                        <small className="settings-hint">
                            Automatically detect emails, phones, SSNs, credit cards.
                            Disable if using anchors to avoid duplicates.
                        </small>
                    </div>

                    {/* Section divider */}
                    <div className="settings-divider">
                        <span>Advanced Scan Settings</span>
                    </div>

                    {/* Scan Presets */}
                    <div className="settings-group">
                        <label className="settings-label">Quick Presets</label>
                        <div className="settings-toggle-group">
                            <button
                                className="settings-toggle"
                                onClick={() => {
                                    onSettingsChange({
                                        scanInterval: 60,
                                        ocrScale: 0.5,
                                        motionThreshold: 40
                                    });
                                }}
                                title="Faster processing, good for quick previews"
                            >
                                ⚡ Fast
                            </button>
                            <button
                                className="settings-toggle"
                                onClick={() => {
                                    onSettingsChange({
                                        scanInterval: 30,
                                        ocrScale: 0.75,
                                        motionThreshold: 30
                                    });
                                }}
                                title="Balanced speed and accuracy"
                            >
                                ⚖️ Balanced
                            </button>
                            <button
                                className="settings-toggle"
                                onClick={() => {
                                    onSettingsChange({
                                        scanInterval: 15,
                                        ocrScale: 1.0,
                                        motionThreshold: 20
                                    });
                                }}
                                title="Maximum accuracy, slower processing"
                            >
                                🔍 Thorough
                            </button>
                        </div>
                    </div>

                    {/* Scan Interval */}
                    <div className="settings-group">
                        <label className="settings-label">
                            Scan Interval: {safeSettings.scanInterval} frames
                        </label>
                        <input
                            type="range"
                            min="15"
                            max="300"
                            step="5"
                            value={safeSettings.scanInterval}
                            onChange={e => handleChange('scanInterval', parseInt(e.target.value))}
                            className="settings-slider"
                        />
                        <small className="settings-hint">Frames between OCR scans (lower = more accurate, slower)</small>
                    </div>

                    {/* Motion Threshold */}
                    <div className="settings-group">
                        <label className="settings-label">
                            Motion Threshold: {safeSettings.motionThreshold}
                        </label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            step="5"
                            value={safeSettings.motionThreshold}
                            onChange={e => handleChange('motionThreshold', parseFloat(e.target.value))}
                            className="settings-slider"
                        />
                        <small className="settings-hint">Pixel difference for scroll detection (higher = less sensitive)</small>
                    </div>

                    {/* OCR Scale */}
                    <div className="settings-group">
                        <label className="settings-label">
                            OCR Scale: {safeSettings.ocrScale.toFixed(1)}x
                        </label>
                        <input
                            type="range"
                            min="0.5"
                            max="2.0"
                            step="0.1"
                            value={safeSettings.ocrScale}
                            onChange={e => handleChange('ocrScale', parseFloat(e.target.value))}
                            className="settings-slider"
                        />
                        <small className="settings-hint">Scale factor for OCR (higher = better quality, slower)</small>
                    </div>
                </div>

                <div className="settings-modal-footer">
                    <button className="btn btn-primary" onClick={onClose}>
                        Done
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SettingsModal;
