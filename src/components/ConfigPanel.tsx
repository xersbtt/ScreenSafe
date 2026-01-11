import React, { useState, useCallback } from 'react';

export interface WatchItem {
    id: string;
    text: string;
    enabled: boolean;
}

export interface Anchor {
    id: string;
    label: string;
    direction: 'BELOW' | 'RIGHT' | 'ABOVE' | 'LEFT';
    gap: number;
    width: number;
    height: number;
    enabled: boolean;
}

export interface ScanZone {
    id: string;
    start: number;  // seconds
    end: number;    // seconds
    enabled: boolean;
}

interface ConfigPanelProps {
    watchList: WatchItem[];
    anchors: Anchor[];
    scanZones: ScanZone[];
    videoDuration: number;
    onWatchListChange: (items: WatchItem[]) => void;
    onAnchorsChange: (anchors: Anchor[]) => void;
    onScanZonesChange: (zones: ScanZone[]) => void;
    onSavePreset: (name: string) => void;
    onLoadPreset: (name: string) => void;
    onPickFromVideo?: () => void;
    isPickingAnchor?: boolean;
    pickedAnchorText?: string | null;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({
    watchList,
    anchors,
    scanZones,
    videoDuration,
    onWatchListChange,
    onAnchorsChange,
    onScanZonesChange,
    onSavePreset,
    onLoadPreset,
    onPickFromVideo,
    isPickingAnchor,
    pickedAnchorText
}) => {
    const [newWatchText, setNewWatchText] = useState('');
    const [showAnchorForm, setShowAnchorForm] = useState(false);
    const [editingAnchorId, setEditingAnchorId] = useState<string | null>(null);
    const [anchorForm, setAnchorForm] = useState({
        label: '',
        direction: 'BELOW' as const,
        gap: 10,
        width: 500,
        height: 80
    });

    // Update existing anchor
    const updateAnchor = useCallback((id: string, updates: Partial<Anchor>) => {
        onAnchorsChange(anchors.map(a =>
            a.id === id ? { ...a, ...updates } : a
        ));
    }, [anchors, onAnchorsChange]);

    // Toggle anchor enabled state
    const toggleAnchor = useCallback((id: string) => {
        updateAnchor(id, { enabled: !anchors.find(a => a.id === id)?.enabled });
    }, [anchors, updateAnchor]);

    // Add new watch item
    const handleAddWatch = useCallback(() => {
        if (!newWatchText.trim()) return;

        const newItems = newWatchText
            .split(',')
            .map(t => t.trim().toLowerCase())
            .filter(t => t.length > 0)
            .map(text => ({
                id: `watch-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                text,
                enabled: true
            }));

        onWatchListChange([...watchList, ...newItems]);
        setNewWatchText('');
    }, [newWatchText, watchList, onWatchListChange]);

    // Toggle watch item
    const toggleWatchItem = useCallback((id: string) => {
        onWatchListChange(watchList.map(item =>
            item.id === id ? { ...item, enabled: !item.enabled } : item
        ));
    }, [watchList, onWatchListChange]);

    // Delete watch item
    const deleteWatchItem = useCallback((id: string) => {
        onWatchListChange(watchList.filter(item => item.id !== id));
    }, [watchList, onWatchListChange]);

    // Add new anchor
    const handleAddAnchor = useCallback(() => {
        if (!anchorForm.label.trim()) return;

        const newAnchor: Anchor = {
            id: `anchor-${Date.now()}`,
            ...anchorForm,
            enabled: true
        };

        onAnchorsChange([...anchors, newAnchor]);
        setAnchorForm({ label: '', direction: 'BELOW', gap: 10, width: 500, height: 80 });
        setShowAnchorForm(false);
    }, [anchorForm, anchors, onAnchorsChange]);

    // Delete anchor
    const deleteAnchor = useCallback((id: string) => {
        onAnchorsChange(anchors.filter(a => a.id !== id));
    }, [anchors, onAnchorsChange]);

    return (
        <div className="config-panel">
            {/* WATCH LIST SECTION */}
            <div className="config-section">
                <div className="config-section-header">
                    <span className="config-section-title">📝 Watch List</span>
                    <span className="config-section-count">{watchList.filter(w => w.enabled).length}</span>
                </div>

                <div className="config-section-desc">
                    Text to blur (names, passwords, card numbers)
                </div>

                <div className="watch-list-input">
                    <input
                        type="text"
                        placeholder="Enter text (comma separated)"
                        value={newWatchText}
                        onChange={e => setNewWatchText(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleAddWatch()}
                    />
                    <button className="btn btn-sm btn-primary" onClick={handleAddWatch}>
                        Add
                    </button>
                </div>

                <div className="watch-list-items">
                    {watchList.map(item => (
                        <div key={item.id} className={`watch-item ${!item.enabled ? 'disabled' : ''}`}>
                            <input
                                type="checkbox"
                                checked={item.enabled}
                                onChange={() => toggleWatchItem(item.id)}
                            />
                            <span className="watch-item-text">{item.text}</span>
                            <button
                                className="watch-item-delete"
                                onClick={() => deleteWatchItem(item.id)}
                            >
                                ×
                            </button>
                        </div>
                    ))}
                </div>
            </div>

            {/* ANCHORS SECTION */}
            <div className="config-section">
                <div className="config-section-header">
                    <span className="config-section-title">📍 Anchors</span>
                    <span className="config-section-count">{anchors.length}</span>
                </div>

                <div className="config-section-desc">
                    Blur relative to label text
                </div>

                {!showAnchorForm ? (
                    <button className="btn btn-sm" onClick={() => setShowAnchorForm(true)}>
                        + Add Anchor
                    </button>
                ) : (
                    <div className="anchor-form">
                        <div className="anchor-form-row">
                            <input
                                type="text"
                                placeholder="Label text (e.g., 'Card Number:')"
                                value={pickedAnchorText || anchorForm.label}
                                onChange={e => setAnchorForm({ ...anchorForm, label: e.target.value })}
                                style={{ flex: 1 }}
                            />
                            {onPickFromVideo && (
                                <button
                                    className={`btn btn-sm ${isPickingAnchor ? 'btn-primary' : ''}`}
                                    onClick={onPickFromVideo}
                                    title="Click on video to select label text"
                                >
                                    {isPickingAnchor ? '🎯 Click Video...' : '📍 Pick'}
                                </button>
                            )}
                        </div>
                        <div className="anchor-form-row">
                            <select
                                value={anchorForm.direction}
                                onChange={e => setAnchorForm({ ...anchorForm, direction: e.target.value as any })}
                            >
                                <option value="BELOW">Below</option>
                                <option value="RIGHT">Right</option>
                                <option value="ABOVE">Above</option>
                                <option value="LEFT">Left</option>
                            </select>
                            <input
                                type="number"
                                placeholder="Gap"
                                value={anchorForm.gap}
                                onChange={e => setAnchorForm({ ...anchorForm, gap: parseInt(e.target.value) || 0 })}
                                style={{ width: '60px' }}
                            />
                        </div>
                        <div className="anchor-form-row">
                            <input
                                type="number"
                                placeholder="Width"
                                value={anchorForm.width}
                                onChange={e => setAnchorForm({ ...anchorForm, width: parseInt(e.target.value) || 100 })}
                            />
                            <span>×</span>
                            <input
                                type="number"
                                placeholder="Height"
                                value={anchorForm.height}
                                onChange={e => setAnchorForm({ ...anchorForm, height: parseInt(e.target.value) || 50 })}
                            />
                        </div>
                        <div className="anchor-form-actions">
                            <button className="btn btn-sm btn-primary" onClick={handleAddAnchor}>
                                Add
                            </button>
                            <button className="btn btn-sm" onClick={() => setShowAnchorForm(false)}>
                                Cancel
                            </button>
                        </div>
                    </div>
                )}

                <div className="anchor-list">
                    {anchors.map(anchor => (
                        <div key={anchor.id} className={`anchor-item ${!anchor.enabled ? 'disabled' : ''} ${editingAnchorId === anchor.id ? 'editing' : ''}`}>
                            <div className="anchor-item-header" onClick={() => setEditingAnchorId(editingAnchorId === anchor.id ? null : anchor.id)}>
                                <input
                                    type="checkbox"
                                    checked={anchor.enabled}
                                    onChange={(e) => { e.stopPropagation(); toggleAnchor(anchor.id); }}
                                    onClick={(e) => e.stopPropagation()}
                                />
                                <div className="anchor-item-label">"{anchor.label}"</div>
                                <div className="anchor-item-details">
                                    {anchor.direction} +{anchor.gap}px, {anchor.width}×{anchor.height}
                                </div>
                                <button
                                    className="anchor-item-delete"
                                    onClick={(e) => { e.stopPropagation(); deleteAnchor(anchor.id); }}
                                >
                                    ×
                                </button>
                            </div>
                            {editingAnchorId === anchor.id && (
                                <div className="anchor-edit-form">
                                    <div className="anchor-form-row">
                                        <label>Label:</label>
                                        <input
                                            type="text"
                                            value={anchor.label}
                                            onChange={e => updateAnchor(anchor.id, { label: e.target.value })}
                                            style={{ flex: 1 }}
                                        />
                                    </div>
                                    <div className="anchor-form-row">
                                        <label>Direction:</label>
                                        <select
                                            value={anchor.direction}
                                            onChange={e => updateAnchor(anchor.id, { direction: e.target.value as any })}
                                        >
                                            <option value="BELOW">Below</option>
                                            <option value="RIGHT">Right</option>
                                            <option value="ABOVE">Above</option>
                                            <option value="LEFT">Left</option>
                                        </select>
                                        <label style={{ marginLeft: '8px' }}>Gap:</label>
                                        <input
                                            type="number"
                                            value={anchor.gap}
                                            onChange={e => updateAnchor(anchor.id, { gap: parseInt(e.target.value) || 0 })}
                                            style={{ width: '60px' }}
                                        />
                                        <span>px</span>
                                    </div>
                                    <div className="anchor-form-row">
                                        <label>Size:</label>
                                        <input
                                            type="number"
                                            value={anchor.width}
                                            onChange={e => updateAnchor(anchor.id, { width: parseInt(e.target.value) || 100 })}
                                            style={{ width: '70px' }}
                                        />
                                        <span>×</span>
                                        <input
                                            type="number"
                                            value={anchor.height}
                                            onChange={e => updateAnchor(anchor.id, { height: parseInt(e.target.value) || 50 })}
                                            style={{ width: '70px' }}
                                        />
                                        <span>px</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* SCAN ZONES */}
            <div className="config-section">
                <div className="config-section-header">
                    <span className="config-section-title">🎯 Scan Zones</span>
                    <span className="config-section-count">{scanZones.filter(z => z.enabled).length}</span>
                </div>

                <div className="config-section-desc">
                    {scanZones.length === 0
                        ? 'No zones = process entire video'
                        : 'Only process OCR in these time ranges'
                    }
                </div>

                <button
                    className="btn btn-sm"
                    onClick={() => {
                        const newZone: ScanZone = {
                            id: `zone-${Date.now()}`,
                            start: 0,
                            end: Math.min(60, videoDuration),
                            enabled: true
                        };
                        onScanZonesChange([...scanZones, newZone]);
                    }}
                >
                    + Add Zone
                </button>

                <div className="scan-zone-list">
                    {scanZones.map(zone => {
                        const formatTime = (seconds: number) => {
                            const mins = Math.floor(seconds / 60);
                            const secs = Math.floor(seconds % 60);
                            return `${mins}:${secs.toString().padStart(2, '0')}`;
                        };

                        return (
                            <div key={zone.id} className={`scan-zone-item ${!zone.enabled ? 'disabled' : ''}`}>
                                <input
                                    type="checkbox"
                                    checked={zone.enabled}
                                    onChange={() => onScanZonesChange(scanZones.map(z =>
                                        z.id === zone.id ? { ...z, enabled: !z.enabled } : z
                                    ))}
                                />
                                <div className="scan-zone-times">
                                    <input
                                        type="number"
                                        min="0"
                                        max={videoDuration}
                                        step="1"
                                        value={zone.start}
                                        onChange={e => onScanZonesChange(scanZones.map(z =>
                                            z.id === zone.id ? { ...z, start: parseFloat(e.target.value) || 0 } : z
                                        ))}
                                        title={`Start: ${formatTime(zone.start)}`}
                                    />
                                    <span>→</span>
                                    <input
                                        type="number"
                                        min="0"
                                        max={videoDuration}
                                        step="1"
                                        value={zone.end}
                                        onChange={e => onScanZonesChange(scanZones.map(z =>
                                            z.id === zone.id ? { ...z, end: parseFloat(e.target.value) || 0 } : z
                                        ))}
                                        title={`End: ${formatTime(zone.end)}`}
                                    />
                                    <span className="scan-zone-duration">
                                        ({formatTime(zone.end - zone.start)})
                                    </span>
                                </div>
                                <button
                                    className="scan-zone-delete"
                                    onClick={() => onScanZonesChange(scanZones.filter(z => z.id !== zone.id))}
                                >
                                    ×
                                </button>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* PRESETS */}
            <div className="config-section config-section-presets">
                <div className="config-section-header">
                    <span className="config-section-title">💾 Presets</span>
                </div>
                <div className="preset-buttons">
                    <button className="btn btn-sm" onClick={() => onLoadPreset('default')}>
                        📂 Load
                    </button>
                    <button className="btn btn-sm" onClick={() => onSavePreset('default')}>
                        💾 Save
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ConfigPanel;
