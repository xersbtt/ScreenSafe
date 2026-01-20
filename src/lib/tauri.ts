/**
 * ScreenSafe - Tauri IPC and WebSocket bindings
 * 
 * Handles communication with:
 * - Tauri backend (Rust) for file operations
 * - Python sidecar (WebSocket) for AI analysis
 */

import { Detection, AnalysisProgress, AnalysisSettings, PIIType } from '../types';

// WebSocket connection to Python sidecar
let ws: WebSocket | null = null;
let wsReconnectAttempts = 0;
const WS_URL = 'ws://127.0.0.1:9876';
const MAX_RECONNECT_ATTEMPTS = 5;

// Message types matching Python enum
export type WSMessageType =
    | 'start_analysis'
    | 'cancel_analysis'
    | 'progress_update'
    | 'detection_found'
    | 'analysis_complete'
    | 'analysis_error'
    | 'track_object'
    | 'track_result'
    | 'start_export'
    | 'export_progress'
    | 'export_complete'
    | 'export_error'
    | 'scan_complete'
    | 'update_config'
    | 'preview_frame'
    | 'get_text_at_click'
    | 'get_text_in_region'
    | 'start_scan'
    | 'system_info';

interface WSMessage {
    type: WSMessageType;
    payload: Record<string, unknown>;
}

// Event callbacks
type ProgressCallback = (progress: AnalysisProgress) => void;
type DetectionCallback = (detection: Detection) => void;
type CompleteCallback = (detections: Detection[]) => void;
type ErrorCallback = (error: string) => void;
type TrackResultCallback = (result: {
    detectionId: string;
    startTime: number;
    endTime: number;
    framePositions?: Array<[number, number, number, number, number]>; // [frame, x, y, w, h]
}) => void;
type SystemInfoCallback = (info: { gpuAvailable: boolean; gpuName: string | null; gpuType: string | null }) => void;

let onProgress: ProgressCallback | null = null;
let onDetection: DetectionCallback | null = null;
let onComplete: CompleteCallback | null = null;
let onError: ErrorCallback | null = null;
let onTrackResult: TrackResultCallback | null = null;
let onSystemInfo: SystemInfoCallback | null = null;

// GPU Status (updated on connection)
let gpuStatus: { available: boolean; name: string | null; type: string | null } = { available: false, name: null, type: null };

/**
 * Get current GPU status
 */
export function getGpuStatus(): { available: boolean; name: string | null; type: string | null } {
    return gpuStatus;
}

/**
 * Set callback for system info updates
 */
export function onSystemInfoUpdate(callback: SystemInfoCallback): void {
    onSystemInfo = callback;
}

/**
 * Connect to Python sidecar WebSocket
 */
export function connectSidecar(): Promise<void> {
    return new Promise((resolve, reject) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            resolve();
            return;
        }

        console.log('[Tauri] Connecting to Python sidecar...');
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[Tauri] Connected to Python sidecar');
            wsReconnectAttempts = 0;
            resolve();
        };

        ws.onmessage = (event) => {
            try {
                const message: WSMessage = JSON.parse(event.data);
                handleWSMessage(message);
            } catch (e) {
                console.error('[Tauri] Failed to parse message:', e);
            }
        };

        ws.onerror = (error) => {
            console.error('[Tauri] WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('[Tauri] WebSocket closed');
            ws = null;

            // Auto-reconnect
            if (wsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                wsReconnectAttempts++;
                console.log(`[Tauri] Reconnecting (attempt ${wsReconnectAttempts})...`);
                setTimeout(() => connectSidecar().catch(() => { }), 2000);
            }
        };

        // Timeout for initial connection
        setTimeout(() => {
            if (ws?.readyState !== WebSocket.OPEN) {
                reject(new Error('Connection timeout'));
            }
        }, 5000);
    });
}

/**
 * Disconnect from sidecar
 */
export function disconnectSidecar(): void {
    if (ws) {
        ws.close();
        ws = null;
    }
}

/**
 * Send message to Python sidecar
 */
function sendMessage(type: WSMessageType, payload: Record<string, unknown> = {}): void {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        console.error('[Tauri] WebSocket not connected, state:', ws?.readyState);
        onError?.('Not connected to AI engine');
        return;
    }

    const message: WSMessage = { type, payload };
    console.log('[Tauri] Sending message:', type, JSON.stringify(payload).slice(0, 200));
    ws.send(JSON.stringify(message));
}

/**
 * Handle incoming WebSocket message
 */
function handleWSMessage(message: WSMessage): void {
    switch (message.type) {
        case 'progress_update':
            if (onProgress) {
                const progress = mapProgress(message.payload);
                onProgress(progress);
            }
            // Handle text_regions callback for click-based anchor selection
            if ((message.payload as Record<string, unknown>).text_regions && onTextAtClick) {
                const regions = (message.payload as Record<string, unknown>).text_regions as TextRegion[];
                onTextAtClick(regions);
                onTextAtClick = null;  // Clear callback after use
            }
            // Handle region_text callback for box-based anchor selection
            if ((message.payload as Record<string, unknown>).region_text !== undefined && onTextInRegion) {
                const text = (message.payload as Record<string, unknown>).region_text as string;
                const region = (message.payload as Record<string, unknown>).region as { x: number; y: number; width: number; height: number };
                onTextInRegion(text, region);
                onTextInRegion = null;  // Clear callback after use
            }
            break;

        case 'scan_complete':
            if (onScanComplete) {
                const detectedBlurs = (message.payload as Record<string, unknown>).detected_blurs as Array<{
                    frame: number;
                    time: number;
                    end_time?: number;
                    bbox: { x: number; y: number; width: number; height: number };
                    type: string;
                    source: string;
                    frame_positions?: Array<[number, number, number, number, number]>;
                }>;
                const cancelled = (message.payload as Record<string, unknown>).cancelled as boolean || false;
                console.log('[Tauri] Scan complete, detected', detectedBlurs?.length || 0, 'blur regions, cancelled:', cancelled);
                onScanComplete(detectedBlurs || [], cancelled);
                onScanComplete = null;  // Clear callback after use
            }
            break;

        case 'detection_found':
            if (onDetection) {
                const detection = mapDetection(message.payload);
                onDetection(detection);
            }
            break;

        case 'analysis_complete':
            if (onComplete) {
                const result = message.payload as { detections: unknown[] };
                const detections = (result.detections || []).map(d => mapDetection(d as Record<string, unknown>));
                onComplete(detections);
            }
            break;

        case 'analysis_error':
            if (onError) {
                const error = message.payload.error as string || 'Unknown error';
                onError(error);
            }
            break;

        case 'export_progress':
            if (onExportProgress) {
                const progress = message.payload.progress as number || 0;
                const stage = message.payload.stage as string || '';
                onExportProgress(progress, stage);
            }
            break;

        case 'export_complete':
            if (onExportComplete) {
                const outputPath = message.payload.output_path as string || '';
                const success = message.payload.success as boolean || false;
                const detectedBlurs = message.payload.detected_blurs as Array<{
                    frame: number;
                    time: number;
                    bbox: { x: number; y: number; width: number; height: number };
                    type: string;
                    source: string;
                }> || [];
                onExportComplete(outputPath, success, detectedBlurs);
            }
            break;

        case 'export_error':
            if (onExportError) {
                const error = message.payload.error as string || 'Export failed';
                onExportError(error);
            }
            break;

        case 'track_result':
            if (onTrackResult) {
                const framePositions = message.payload.frame_positions as Array<[number, number, number, number, number]> | undefined;
                const result = {
                    detectionId: message.payload.detection_id as string,
                    startTime: message.payload.start_time as number,
                    endTime: message.payload.end_time as number,
                    framePositions: framePositions
                };
                console.log('[Tauri] Track result:', result.detectionId, framePositions?.length || 0, 'positions');
                onTrackResult(result);
            }
            break;

        case 'system_info':
            // Update GPU status from Python backend
            gpuStatus = {
                available: message.payload.gpu_available as boolean || false,
                name: message.payload.gpu_name as string || null,
                type: message.payload.gpu_type as string || null
            };
            console.log('[Tauri] System info:', gpuStatus);
            if (onSystemInfo) {
                onSystemInfo({
                    gpuAvailable: gpuStatus.available,
                    gpuName: gpuStatus.name,
                    gpuType: gpuStatus.type
                });
            }
            break;

        default:
            console.log('[Tauri] Unhandled message:', message.type);
    }
}

/**
 * Start video analysis
 */
export async function startAnalysis(
    videoPath: string,
    settings: Partial<AnalysisSettings> = {},
    callbacks: {
        onProgress?: ProgressCallback;
        onDetection?: DetectionCallback;
        onComplete?: CompleteCallback;
        onError?: ErrorCallback;
    } = {}
): Promise<void> {
    // Set callbacks
    onProgress = callbacks.onProgress || null;
    onDetection = callbacks.onDetection || null;
    onComplete = callbacks.onComplete || null;
    onError = callbacks.onError || null;

    // Ensure connected
    await connectSidecar();

    // Send analysis request
    sendMessage('start_analysis', {
        video_path: videoPath,
        settings: {
            mode: settings.mode || 'quality',
            detect_emails: settings.detectEmails ?? true,
            detect_phones: settings.detectPhones ?? true,
            detect_credit_cards: settings.detectCreditCards ?? true,
            detect_passwords: settings.detectPasswords ?? true,
            detect_names: settings.detectNames ?? true,
            detect_addresses: settings.detectAddresses ?? true,
            detect_api_keys: settings.detectApiKeys ?? true,
            frame_skip: 15,  // Process every 15th frame for speed (adjustable later)
        },
        use_gpu: true,  // RTX 3080 acceleration
    });
}

/**
 * Cancel ongoing analysis
 */
export function cancelAnalysis(): void {
    sendMessage('cancel_analysis');
}

/**
 * Request magic wand tracking
 */
export function trackObject(
    bbox: { x: number; y: number; width: number; height: number },
    frameNumber: number,
    direction: 'forward' | 'backward' | 'both'
): void {
    sendMessage('track_object', {
        bbox,
        frame_number: frameNumber,
        direction,
    });
}

// Export callbacks
let onExportProgress: ((progress: number, stage: string) => void) | null = null;
let onExportComplete: ((outputPath: string, success: boolean, detectedBlurs?: Array<{
    frame: number;
    time: number;
    bbox: { x: number; y: number; width: number; height: number };
    type: string;
    source: string;
}>) => void) | null = null;
let onExportError: ((error: string) => void) | null = null;

/**
 * Start video export with redactions
 */
export async function startExport(
    videoPath: string,
    detections: Array<{
        id: string;
        type: string;
        content: string;
        confidence: number;
        bbox: { x: number; y: number; width: number; height: number };
        startTime: number;
        endTime: number;
        frameStart: number;
        frameEnd: number;
        isRedacted: boolean;
        trackId?: string;
        framePositions?: Array<[number, number, number, number, number]>;  // [frame, x, y, width, height]
    }>,
    anchors: Array<{
        id: string;
        label: string;
        direction: 'BELOW' | 'RIGHT' | 'ABOVE' | 'LEFT';
        gap: number;
        width: number;
        height: number;
        enabled: boolean;
    }>,
    watchList?: string[],
    config?: {
        scanInterval?: number;
        motionThreshold?: number;
        ocrScale?: number;
        scanZones?: Array<{ start: number; end: number }>;
        codec?: 'h264' | 'h265' | 'vp9';
        quality?: 'low' | 'medium' | 'high';
        resolution?: 'original' | '1080p' | '720p' | '480p';
        includeAudio?: boolean;
        preview?: boolean;  // Low-res preview mode
    },
    outputPath?: string,
    callbacks?: {
        onProgress?: (progress: number, stage: string) => void;
        onComplete?: (outputPath: string, success: boolean, detectedBlurs?: Array<{
            frame: number;
            time: number;
            bbox: { x: number; y: number; width: number; height: number };
            type: string;
            source: string;
        }>) => void;
        onError?: (error: string) => void;
    }
): Promise<void> {
    // Set callbacks
    onExportProgress = callbacks?.onProgress || null;
    onExportComplete = callbacks?.onComplete || null;
    onExportError = callbacks?.onError || null;

    // Ensure connected
    await connectSidecar();

    // Send export request
    console.log('[Tauri] Starting export with', detections.length, 'detections,', anchors.length, 'anchors,', (watchList || []).length, 'watch items', config?.preview ? '(PREVIEW MODE)' : '');
    sendMessage('start_export', {
        video_path: videoPath,
        output_path: outputPath,
        detections: detections,
        anchors: anchors,
        watch_list: watchList || [],
        scan_interval: config?.scanInterval ?? 90,
        motion_threshold: config?.motionThreshold ?? 30.0,
        ocr_scale: config?.ocrScale ?? 1.0,
        scan_zones: config?.scanZones || [],
        codec: config?.codec ?? 'h264',
        quality: config?.quality ?? 'high',
        resolution: config?.resolution ?? 'original',
        include_audio: config?.includeAudio ?? true,
        preview: config?.preview ?? false,
    });
}

/**
 * Track a region forward through the video to find when content disappears
 * 
 * @param videoPath - Path to video file
 * @param detectionId - ID of the detection to update
 * @param bbox - Normalized bounding box {x, y, width, height}
 * @param timestamp - Timestamp in seconds where region was selected
 * @param callback - Called with tracking result
 */
export async function trackRegion(
    videoPath: string,
    detectionId: string,
    bbox: { x: number; y: number; width: number; height: number },
    timestamp: number,  // Seconds (supports VFR videos)
    callback: TrackResultCallback
): Promise<void> {
    // Set callback
    onTrackResult = callback;

    // Ensure connected
    await connectSidecar();

    // Send track request
    console.log('[Tauri] Starting region tracking:', { detectionId, bbox, timestamp });
    sendMessage('track_object', {
        video_path: videoPath,
        detection_id: detectionId,
        bbox: bbox,
        timestamp: timestamp,  // Send timestamp instead of frame_number
    });
}

// Scan callback
let onScanComplete: ((detectedBlurs: Array<{
    frame: number;
    time: number;
    end_time?: number;
    bbox: { x: number; y: number; width: number; height: number };
    type: string;
    source: string;
    frame_positions?: Array<[number, number, number, number, number]>;
}>, cancelled: boolean) => void) | null = null;

/**
 * Start video scan for blur regions WITHOUT encoding
 * 
 * This is much faster than preview export since it only runs OCR analysis
 * and returns detected blur regions for overlay display on the original video.
 */
export async function startScan(
    videoPath: string,
    anchors: Array<{
        id: string;
        label: string;
        direction: 'BELOW' | 'RIGHT' | 'ABOVE' | 'LEFT';
        gap: number;
        width: number;
        height: number;
        enabled: boolean;
    }>,
    watchList: string[],
    config?: {
        scanInterval?: number;
        motionThreshold?: number;
        ocrScale?: number;
        scanZones?: Array<{ start: number; end: number }>;
        enableRegexPatterns?: boolean;
    },
    callbacks?: {
        onProgress?: (progress: number, stage: string) => void;
        onComplete?: (detectedBlurs: Array<{
            frame: number;
            time: number;
            end_time?: number;
            bbox: { x: number; y: number; width: number; height: number };
            type: string;
            source: string;
            frame_positions?: Array<[number, number, number, number, number]>;
        }>, cancelled: boolean) => void;
        onError?: (error: string) => void;
    }
): Promise<void> {
    // Set callbacks
    onExportProgress = callbacks?.onProgress || null;
    onScanComplete = callbacks?.onComplete || null;
    onExportError = callbacks?.onError || null;

    // Ensure connected
    await connectSidecar();

    // Convert scan_zones format
    const zones = (config?.scanZones || []).map(z => ({
        start: z.start,
        end: z.end
    }));

    console.log('[Tauri] Starting video scan with', anchors.length, 'anchors,', watchList.length, 'watch items');
    sendMessage('start_scan', {
        video_path: videoPath,
        anchors: anchors,
        watch_list: watchList,
        scan_interval: config?.scanInterval ?? 30,
        motion_threshold: config?.motionThreshold ?? 30.0,
        ocr_scale: config?.ocrScale ?? 1.0,
        scan_zones: zones,
        enable_regex_patterns: config?.enableRegexPatterns ?? false,
    });
}

/**
 * Update PII wizard configuration (watch list, anchors, etc.)
 */
export interface PiiWizardConfig {
    watchList: string[];
    anchors: Record<string, [string, number, number, number]>;  // label -> [direction, gap, width, height]
    scanZones?: Array<{ startTime: number; endTime: number }>;
    enableKeyboard?: boolean;
    scanInterval?: number;
    motionThreshold?: number;
    ocrScale?: number;
}

export async function updateConfig(config: PiiWizardConfig): Promise<void> {
    await connectSidecar();

    console.log('[Tauri] Updating config:', config.watchList.length, 'watch items,', Object.keys(config.anchors).length, 'anchors');

    sendMessage('update_config', {
        watch_list: config.watchList,
        anchors: config.anchors,
        scan_zones: config.scanZones || [],
        enable_keyboard: config.enableKeyboard || false,
        scan_interval: config.scanInterval || 15,
        motion_threshold: config.motionThreshold || 3.0,
        ocr_scale: config.ocrScale || 0.5,
    });
}

/**
 * Get OCR text at a clicked position in the video
 */
export interface TextRegion {
    text: string;
    bbox: { x: number; y: number; width: number; height: number };
    distance: number;
    confidence: number;
}

let onTextAtClick: ((regions: TextRegion[]) => void) | null = null;

export async function getTextAtClick(
    videoPath: string,
    timestamp: number,  // Seconds (supports VFR videos)
    clickX: number,  // Normalized 0-1
    clickY: number,  // Normalized 0-1
    callback: (regions: TextRegion[]) => void
): Promise<void> {
    await connectSidecar();

    onTextAtClick = callback;

    console.log('[Tauri] Getting text at click:', { clickX, clickY, timestamp });
    sendMessage('get_text_at_click', {
        video_path: videoPath,
        timestamp: timestamp,
        click_x: clickX,
        click_y: clickY,
    });
}

/**
 * Get OCR text from a drawn region in the video (box-based selection)
 */
let onTextInRegion: ((text: string, region: { x: number; y: number; width: number; height: number }) => void) | null = null;

export async function getTextInRegion(
    videoPath: string,
    timestamp: number,  // Seconds (supports VFR videos)
    region: { x: number; y: number; width: number; height: number },
    callback: (text: string, region: { x: number; y: number; width: number; height: number }) => void
): Promise<void> {
    await connectSidecar();

    onTextInRegion = callback;

    console.log('[Tauri] Getting text in region:', region, 'at', timestamp, 's');
    sendMessage('get_text_in_region', {
        video_path: videoPath,
        timestamp: timestamp,
        region: region,
    });
}

// Export the callback setter for handleWSMessage
export function handleTextInRegionCallback(text: string, region: { x: number; y: number; width: number; height: number }) {
    if (onTextInRegion) {
        onTextInRegion(text, region);
        onTextInRegion = null;
    }
}

// ========== Mappers ==========

function mapProgress(data: Record<string, unknown>): AnalysisProgress {
    return {
        isAnalyzing: true,
        stage: mapStage(data.stage as string),
        progress: (data.progress as number) || 0,
        framesProcessed: (data.frames_processed as number) || 0,
        totalFrames: (data.total_frames as number) || 0,
        detectionsFound: (data.detections_found as number) || 0,
        estimatedTimeRemaining: (data.estimated_time_remaining as number) || 0,
    };
}

function mapStage(stage: string): AnalysisProgress['stage'] {
    const stageMap: Record<string, AnalysisProgress['stage']> = {
        'extracting': 'extracting',
        'detecting': 'detecting',
        'tracking': 'tracking',
        'classifying': 'classifying',
        'complete': 'complete',
    };
    return stageMap[stage] || 'extracting';
}

function mapDetection(data: Record<string, unknown>): Detection {
    const bbox = data.bbox as Record<string, number>;

    return {
        id: (data.id as string) || crypto.randomUUID(),
        type: mapPIIType(data.type as string),
        content: (data.content as string) || '',
        confidence: (data.confidence as number) || 0,
        startTime: (data.start_time as number) || 0,
        endTime: (data.end_time as number) || 0,
        bbox: {
            x: bbox?.x || 0,
            y: bbox?.y || 0,
            width: bbox?.width || 0,
            height: bbox?.height || 0,
        },
        isRedacted: (data.is_redacted as boolean) ?? true,
        trackId: data.track_id as string | undefined,
    };
}

function mapPIIType(type: string): PIIType {
    const typeMap: Record<string, PIIType> = {
        'email': 'email',
        'phone': 'phone',
        'credit-card': 'credit-card',
        'password': 'password',
        'name': 'name',
        'address': 'address',
        'ssn': 'ssn' as PIIType,
        'api-key': 'api-key' as PIIType,
        'manual': 'manual',
    };
    return typeMap[type] || 'manual';
}

// ========== File System Helpers ==========

/**
 * Get the file:// URL for a local video file
 * Works in both browser and Tauri contexts
 */
export function getVideoUrl(file: File): string {
    return URL.createObjectURL(file);
}

/**
 * Get the actual file path for Tauri
 * Note: This requires Tauri's file dialog or drop event
 */
export function getFilePath(file: File): string | null {
    // In Tauri, dropped files have a path property
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const tauriFile = file as any;
    return tauriFile.path || null;
}
