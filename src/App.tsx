import { useState, useRef, useCallback, useEffect } from 'react';
import { DropZone, VideoPlayer, VideoControls, Timeline, DetectionSidebar, AnalysisOverlay, ConfigPanel, SettingsModal, SetupDialog } from './components';
import type { WatchItem, Anchor } from './components';
import type { AppSettings } from './lib/config';
import { Detection, VideoState, AnalysisProgress } from './types';
import { cancelAnalysis, getFilePath, updateConfig, getTextInRegion, startScan, getGpuStatus, onSystemInfoUpdate, connectSidecar } from './lib/tauri';
import { convertFileSrc, invoke } from '@tauri-apps/api/core';
import { open, save } from '@tauri-apps/plugin-dialog';
import { readTextFile, writeTextFile } from '@tauri-apps/plugin-fs';
import type { VideoPlayerHandle } from './components/VideoPlayer';
import './App.css';


// Configuration: Set to true to use real AI pipeline (requires Python sidecar)
const USE_REAL_AI = true;  // Python sidecar must be running!

// Demo detections for UI preview
const DEMO_DETECTIONS: Detection[] = [
  {
    id: 'demo-1',
    type: 'email',
    content: 'john.doe@example.com',
    confidence: 0.98,
    startTime: 2,
    endTime: 8,
    bbox: { x: 0.2, y: 0.3, width: 0.3, height: 0.05 },
    isRedacted: true
  },
  {
    id: 'demo-2',
    type: 'phone',
    content: '+1 (555) 123-4567',
    confidence: 0.95,
    startTime: 12,
    endTime: 18,
    bbox: { x: 0.4, y: 0.5, width: 0.25, height: 0.04 },
    isRedacted: true
  },
  {
    id: 'demo-3',
    type: 'password',
    content: 'MySecureP@ss123!',
    confidence: 0.92,
    startTime: 25,
    endTime: 32,
    bbox: { x: 0.3, y: 0.6, width: 0.2, height: 0.04 },
    isRedacted: true
  },
  {
    id: 'demo-4',
    type: 'credit-card',
    content: '4532 •••• •••• 1234',
    confidence: 0.99,
    startTime: 40,
    endTime: 45,
    bbox: { x: 0.25, y: 0.4, width: 0.35, height: 0.05 },
    isRedacted: true
  }
];

type AppMode = 'idle' | 'loading' | 'analyzing' | 'editing';
type SidebarTab = 'detections' | 'config';

function App() {
  // Setup state - show setup dialog on first launch
  const [isSetupComplete, setIsSetupComplete] = useState<boolean | null>(null);

  // Handle setup completion - start sidecar and show main app
  const handleSetupComplete = useCallback(async () => {
    console.log('[App] Setup complete, starting sidecar...');
    try {
      await invoke('start_sidecar_command');
      console.log('[App] Sidecar started successfully');

      // Wait briefly for sidecar to initialize, then connect to get GPU status
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Connect to sidecar to trigger GPU status message
      const { connectSidecar, getGpuStatus } = await import('./lib/tauri');
      await connectSidecar();

      // Update GPU status after connection
      const status = getGpuStatus();
      console.log('[App] GPU status after setup:', status);
    } catch (err) {
      console.error('[App] Failed to start sidecar:', err);
    }
    setIsSetupComplete(true);
  }, []);

  // Check setup status on mount
  useEffect(() => {
    invoke<boolean>('check_setup_complete').then((complete) => {
      console.log('[App] Setup complete:', complete);
      if (complete) {
        // Already set up, start sidecar immediately
        handleSetupComplete();
      } else {
        setIsSetupComplete(false);
      }
    }).catch((err) => {
      console.log('[App] Setup check failed (dev mode?):', err);
      // In dev mode or if check fails, assume setup is complete
      setIsSetupComplete(true);
    });
  }, [handleSetupComplete]);

  // Application state
  const [mode, setMode] = useState<AppMode>('idle');
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [_filePath, setFilePath] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null);
  const [isSelectionMode, setIsSelectionMode] = useState(false);
  const [selectionType, setSelectionType] = useState<'blur' | 'blackout'>('blur');
  const [enableContentDetection, setEnableContentDetection] = useState(true);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('config');

  // PII Wizard Config state
  const [watchList, setWatchList] = useState<WatchItem[]>([]);
  const [anchors, setAnchors] = useState<Anchor[]>([]);
  const [exportConfig, setExportConfig] = useState({
    scanInterval: 90,
    motionThreshold: 30.0,
    ocrScale: 1.0,
  });
  const [scanZones, setScanZones] = useState<Array<{ id: string; start: number; end: number; enabled: boolean }>>([]);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    onConfirm: () => void;
    onSaveAndContinue?: () => void;
  } | null>(null);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showAboutModal, setShowAboutModal] = useState(false);
  const [appSettings, setAppSettings] = useState<AppSettings>({
    theme: 'dark',
    blurStrength: 25,
    autoSaveProjects: false,
    ocrLanguage: 'eng',
    exportQuality: 'high',
    exportCodec: 'h264',
    scanInterval: 30,
    motionThreshold: 30,
    ocrScale: 0.75,
    enableRegexPatterns: false
  });
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  // Load settings from config file on mount
  useEffect(() => {
    import('./lib/config').then(({ loadSettings }) => {
      loadSettings().then((loaded) => {
        setAppSettings(loaded);
        setSettingsLoaded(true);
        console.log('[App] Settings loaded from config file');
      });
    });
  }, []);

  // Apply theme to document when it changes
  useEffect(() => {
    document.body.setAttribute('data-theme', appSettings.theme);
  }, [appSettings.theme]);

  // Persist settings to config file when they change (after initial load)
  useEffect(() => {
    if (settingsLoaded) {
      import('./lib/config').then(({ saveSettings }) => {
        saveSettings(appSettings);
      });
    }
  }, [appSettings, settingsLoaded]);

  // Anchor pick flow: 'idle' -> 'pickText' -> 'drawBox' -> 'idle'
  const [anchorPickStep, setAnchorPickStep] = useState<'idle' | 'pickText' | 'drawBox'>('idle');
  const [pickedAnchorText, setPickedAnchorText] = useState<string | null>(null);
  const [pickedAnchorBbox, setPickedAnchorBbox] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [anchorDrawBox, setAnchorDrawBox] = useState<{ startX: number; startY: number; endX: number; endY: number } | null>(null);

  // Video state
  const [videoState, setVideoState] = useState<VideoState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    isMuted: false,
    playbackRate: 1
  });

  // Analysis progress state
  const [analysisProgress, setAnalysisProgress] = useState<AnalysisProgress>({
    isAnalyzing: false,
    stage: 'extracting',
    progress: 0,
    framesProcessed: 0,
    totalFrames: 0,
    detectionsFound: 0,
    estimatedTimeRemaining: 0
  });

  // Export progress state
  const [exportProgress, setExportProgress] = useState({
    isExporting: false,
    progress: 0,
    stage: '',
    eta: ''
  });

  // Blur preview toggle (on by default)
  const [previewBlur, setPreviewBlur] = useState(true);

  // GPU status (received from Python backend on connection)
  const [gpuStatus, setGpuStatus] = useState<{ available: boolean; name: string | null; type: string | null }>(() => getGpuStatus());

  // Subscribe to GPU status updates from Python sidecar
  useEffect(() => {
    // Register callback for GPU status updates
    onSystemInfoUpdate((info) => {
      setGpuStatus({ available: info.gpuAvailable, name: info.gpuName, type: info.gpuType });
    });

    // Connect to sidecar on startup to receive GPU status immediately
    connectSidecar().then(() => {
      // After connection, check the status
      const status = getGpuStatus();
      setGpuStatus(status);
    }).catch((err) => {
      console.log('[App] Sidecar connection pending, will retry on first action:', err);
    });

    // Also check after a delay in case message arrived before callback was set
    const timer = setTimeout(() => {
      const status = getGpuStatus();
      setGpuStatus(status);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const videoPlayerRef = useRef<VideoPlayerHandle>(null);

  // Refs to track latest values for window close handler (avoids stale closure)
  const hasUnsavedChangesRef = useRef(hasUnsavedChanges);
  const handleSaveProjectRef = useRef<(() => Promise<void>) | null>(null);

  // Handle file path selection from Tauri dialog (preferred - gives real path)
  const handlePathSelect = useCallback((path: string) => {
    console.log('[App] Path selected via Tauri dialog:', path);
    setFilePath(path);

    // Use Tauri's convertFileSrc to create a proper asset:// URL for the webview
    const assetUrl = convertFileSrc(path);
    console.log('[App] Asset URL for video:', assetUrl);
    setVideoUrl(assetUrl);

    // New workflow: Go directly to editing mode (no auto-analysis)
    setDetections([]);
    setMode('editing');

    console.log('[App] Instant load complete - ready for manual selection');
  }, []);

  // Handle file selection (legacy - for drag-drop which may not have real path)
  const handleFileSelect = useCallback((file: File) => {
    const path = getFilePath(file);
    setFilePath(path);
    const url = URL.createObjectURL(file);
    setVideoUrl(url);

    if (USE_REAL_AI && path) {
      // Use handlePathSelect for real AI
      handlePathSelect(path);
    } else {
      // Demo mode with mock analysis - will trigger startMockAnalysis via effect
      setMode('analyzing');
      // Note: mock analysis runs via startMockAnalysis defined below
      console.log('Demo mode: starting mock analysis');
    }
  }, [handlePathSelect]);


  // Mock analysis (will be replaced with real AI pipeline)
  const startMockAnalysis = useCallback(() => {
    setAnalysisProgress({
      isAnalyzing: true,
      stage: 'extracting',
      progress: 0,
      framesProcessed: 0,
      totalFrames: 300,
      detectionsFound: 0,
      estimatedTimeRemaining: 15
    });

    const stages: AnalysisProgress['stage'][] = ['extracting', 'detecting', 'tracking', 'classifying', 'complete'];
    let currentProgress = 0;
    let stageIndex = 0;
    let detectionsFound = 0;

    const interval = setInterval(() => {
      currentProgress += Math.random() * 8 + 2;

      if (currentProgress >= 25 && stageIndex === 0) stageIndex = 1;
      if (currentProgress >= 50 && stageIndex === 1) stageIndex = 2;
      if (currentProgress >= 75 && stageIndex === 2) stageIndex = 3;
      if (currentProgress >= 100) {
        stageIndex = 4;
        currentProgress = 100;
      }

      // Simulate finding detections
      if (stageIndex >= 1 && Math.random() > 0.7) {
        detectionsFound += 1;
      }

      setAnalysisProgress(prev => ({
        ...prev,
        stage: stages[stageIndex],
        progress: Math.min(currentProgress, 100),
        framesProcessed: Math.floor(currentProgress * 3),
        detectionsFound,
        estimatedTimeRemaining: Math.max(0, Math.round(15 - (currentProgress / 100) * 15))
      }));

      if (currentProgress >= 100) {
        clearInterval(interval);
        setTimeout(() => {
          setDetections(DEMO_DETECTIONS);
          setMode('editing');
          setAnalysisProgress(prev => ({ ...prev, isAnalyzing: false }));
        }, 500);
      }
    }, 200);
  }, []);

  // Effect: Start mock analysis when in demo mode and analyzing
  useEffect(() => {
    if (!USE_REAL_AI && mode === 'analyzing' && !analysisProgress.isAnalyzing) {
      startMockAnalysis();
    }
  }, [mode, analysisProgress.isAnalyzing, startMockAnalysis]);

  // Cancel analysis
  const handleCancelAnalysis = useCallback(() => {
    if (USE_REAL_AI) {
      cancelAnalysis();
    }
    setMode('idle');
    setVideoUrl('');
    setFilePath(null);
    setDetections([]);
    setAnalysisProgress(prev => ({ ...prev, isAnalyzing: false }));
  }, []);

  // New Project - full reset to idle state
  const handleNewProject = useCallback(() => {
    setMode('idle');
    setVideoUrl('');
    setFilePath(null);
    setDetections([]);
    setAnalysisProgress(prev => ({ ...prev, isAnalyzing: false }));
    setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
    // Optionally reset config too for a truly clean slate
    setWatchList([]);
    setAnchors([]);
    setScanZones([]);
    setHasUnsavedChanges(false);
    setToastMessage('🆕 New project started');
  }, []);

  // Wrapper to check for unsaved changes before new project
  const handleNewProjectWithConfirm = useCallback(() => {
    // Use Ref to avoid stale closure
    if (hasUnsavedChangesRef.current) {
      setConfirmDialog({
        isOpen: true,
        title: 'Unsaved Changes',
        message: 'You have unsaved changes. Do you want to save before starting a new project?',
        onConfirm: () => {
          setConfirmDialog(null);
          handleNewProject();
        },
        onSaveAndContinue: () => {
          setConfirmDialog(null);
          handleSaveProject();
          handleNewProject();
        }
      });
    } else {
      handleNewProject();
    }
  }, [handleNewProject]);

  // Load New Video - keep current settings, load a different video
  const handleLoadNewVideoCore = useCallback(async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const result = await open({
        multiple: false,
        filters: [
          { name: 'Video Files', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }
        ]
      });
      if (result) {
        const path = result as string;
        // Clear detections but keep settings
        setDetections([]);
        setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
        setHasUnsavedChanges(false);
        // Use handlePathSelect for consistent URL generation (fixes video reload bug)
        handlePathSelect(path);
        setToastMessage('📼 Video loaded - settings preserved');
      }
    } catch (err) {
      console.error('Failed to load video:', err);
      setToastMessage('❌ Failed to load video');
    }
  }, [handlePathSelect]);

  // Wrapper to check for unsaved changes before loading new video
  const handleLoadNewVideo = useCallback(() => {
    // Use Ref to avoid stale closure
    if (hasUnsavedChangesRef.current) {
      setConfirmDialog({
        isOpen: true,
        title: 'Unsaved Changes',
        message: 'You have unsaved changes. Do you want to save before loading a new video?',
        onConfirm: () => {
          setConfirmDialog(null);
          handleLoadNewVideoCore();
        },
        onSaveAndContinue: () => {
          setConfirmDialog(null);
          handleSaveProject();
          handleLoadNewVideoCore();
        }
      });
    } else {
      handleLoadNewVideoCore();
    }
  }, [handleLoadNewVideoCore]);

  // Video control handlers
  const handlePlay = useCallback(() => {
    videoPlayerRef.current?.play();
  }, []);

  const handlePause = useCallback(() => {
    videoPlayerRef.current?.pause();
  }, []);

  const handleSeek = useCallback((time: number) => {
    videoPlayerRef.current?.seek(time);
    setVideoState(prev => ({ ...prev, currentTime: time }));
  }, []);

  const handleTimeUpdate = useCallback((time: number) => {
    setVideoState(prev => ({ ...prev, currentTime: time }));
  }, []);

  const handleDurationChange = useCallback((duration: number) => {
    setVideoState(prev => ({ ...prev, duration }));
  }, []);

  const handleVideoStateChange = useCallback((state: Partial<VideoState>) => {
    setVideoState(prev => ({ ...prev, ...state }));
  }, []);

  // Detection handlers
  const handleDetectionSelect = useCallback((detection: Detection) => {
    // Toggle selection - click same item again to deselect
    if (selectedDetectionId === detection.id) {
      setSelectedDetectionId(null);
    } else {
      setSelectedDetectionId(detection.id);
      handleSeek(detection.startTime);
    }
  }, [handleSeek, selectedDetectionId]);

  const handleToggleRedaction = useCallback((detectionId: string) => {
    setDetections(prev => prev.map(d =>
      d.id === detectionId ? { ...d, isRedacted: !d.isRedacted } : d
    ));
  }, []);

  // Handle detection time changes from timeline drag
  const handleDetectionTimeChange = useCallback((detectionId: string, startTime: number, endTime: number) => {
    setDetections(prev => prev.map(d =>
      d.id === detectionId ? { ...d, startTime, endTime } : d
    ));
  }, []);

  // Handle detection deletion (irreversible - requires rescan to restore)
  const handleDeleteDetection = useCallback((detectionId: string) => {
    setDetections(prev => prev.filter(d => d.id !== detectionId));
    // Clear selection if deleted detection was selected
    if (selectedDetectionId === detectionId) {
      setSelectedDetectionId(null);
    }
  }, [selectedDetectionId]);

  const handleDetectionClick = useCallback((detection: Detection) => {
    // Toggle selection - click same item again to deselect
    if (selectedDetectionId === detection.id) {
      setSelectedDetectionId(null);
    } else {
      setSelectedDetectionId(detection.id);
      // Auto-switch to detections tab to show and scroll to the item
      if (sidebarTab !== 'detections') {
        setSidebarTab('detections');
      }
    }
  }, [selectedDetectionId, sidebarTab]);

  // Config handlers - sync to Python backend
  const handleWatchListChange = useCallback((items: WatchItem[]) => {
    setWatchList(items);
    // Sync to backend
    const config = {
      watchList: items.filter(i => i.enabled).map(i => i.text),
      anchors: anchors.reduce((acc, a) => {
        acc[a.label] = [a.direction, a.gap, a.width, a.height];
        return acc;
      }, {} as Record<string, [string, number, number, number]>)
    };
    updateConfig(config);
  }, [anchors]);

  const handleAnchorsChange = useCallback((newAnchors: Anchor[]) => {
    setAnchors(newAnchors);
    // Sync to backend
    const config = {
      watchList: watchList.filter(i => i.enabled).map(i => i.text),
      anchors: newAnchors.reduce((acc, a) => {
        acc[a.label] = [a.direction, a.gap, a.width, a.height];
        return acc;
      }, {} as Record<string, [string, number, number, number]>)
    };
    updateConfig(config);
  }, [watchList]);

  const handleSavePreset = useCallback(async (_name: string) => {
    console.log('[App] handleSavePreset called, opening save dialog...');
    try {
      const filePath = await save({
        title: 'Save Preset',
        filters: [{ name: 'JSON Preset', extensions: ['json'] }],
        defaultPath: 'screensafe-preset.json'
      });

      if (filePath) {
        const preset = {
          watchList: watchList,
          anchors: anchors,
          exportConfig: {
            scanInterval: appSettings.scanInterval,
            motionThreshold: appSettings.motionThreshold,
            ocrScale: appSettings.ocrScale,
          }
        };
        await writeTextFile(filePath, JSON.stringify(preset, null, 2));
        console.log('[App] Preset saved to file:', filePath);
        const fileName = filePath.split(/[/\\]/).pop() || 'preset';
        setToastMessage(`✅ Preset saved to "${fileName}"`);
      }
    } catch (err) {
      console.error('[App] Error saving preset:', err);
      setToastMessage(`❌ Failed to save preset: ${err}`);
    }
    setTimeout(() => setToastMessage(null), 3000);
  }, [watchList, anchors, appSettings]);

  const handleLoadPreset = useCallback(async (_name: string) => {
    console.log('[App] handleLoadPreset called, opening load dialog...');
    try {
      const filePath = await open({
        title: 'Load Preset',
        multiple: false,
        filters: [{ name: 'JSON Preset', extensions: ['json'] }]
      });

      if (filePath && typeof filePath === 'string') {
        const content = await readTextFile(filePath);
        const preset = JSON.parse(content);

        // Ensure watchList is an array
        const loadedWatchList = Array.isArray(preset.watchList) ? preset.watchList : [];
        setWatchList(loadedWatchList);

        // Ensure anchors is an array - convert from object format if needed
        let loadedAnchors = [];
        if (Array.isArray(preset.anchors)) {
          loadedAnchors = preset.anchors;
        } else if (preset.anchors && typeof preset.anchors === 'object') {
          // Convert object format to array format
          // Supports: {label: [direction, gap, width, height]} (Python config format)
          // And:      {label: {direction, gap, width, height}} (object format)
          loadedAnchors = Object.entries(preset.anchors).map(([label, config]: [string, any]) => {
            // Check if config is an array [direction, gap, width, height]
            if (Array.isArray(config) && config.length >= 4) {
              return {
                id: crypto.randomUUID(),
                label: label,
                direction: config[0] || 'BELOW',
                gap: config[1] || 10,
                width: config[2] || 200,
                height: config[3] || 30,
                enabled: true
              };
            }
            // Otherwise treat as object with named properties
            return {
              id: crypto.randomUUID(),
              label: label,
              direction: config.direction || 'BELOW',
              gap: config.gap || 10,
              width: config.width || 200,
              height: config.height || 30,
              enabled: true
            };
          });
          console.log('[App] Converted anchors from object to array format:', loadedAnchors);
        }
        setAnchors(loadedAnchors);

        // Load export config into appSettings if present in preset
        if (preset.exportConfig) {
          setAppSettings((prev: AppSettings) => ({
            ...prev,
            scanInterval: preset.exportConfig.scanInterval ?? prev.scanInterval,
            motionThreshold: preset.exportConfig.motionThreshold ?? prev.motionThreshold,
            ocrScale: preset.exportConfig.ocrScale ?? prev.ocrScale,
          }));
        }
        console.log('[App] Preset loaded from file:', filePath);
        const fileName = filePath.split(/[/\\]/).pop() || 'preset';
        setToastMessage(`✅ Loaded preset from "${fileName}"`);
      }
    } catch (err) {
      console.error('[App] Error loading preset:', err);
      setToastMessage(`❌ Failed to load preset: ${err}`);
    }
    setTimeout(() => setToastMessage(null), 3000);
  }, []);

  // Anchor pick from video handlers
  const handlePickFromVideo = useCallback(() => {
    if (!_filePath) {
      setToastMessage('Load a video first');
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }
    setAnchorPickStep('pickText');  // Step 1: Draw box around anchor text
    setPickedAnchorText(null);
    setPickedAnchorBbox(null);
    setAnchorDrawBox(null);
    setToastMessage('🎯 Step 1: Draw a box around the anchor text...');
  }, [_filePath]);

  // Handle anchor box drawn (Step 1) - OCR the region to get anchor text
  const handleAnchorBoxDrawn = useCallback((anchorBox: { x: number; y: number; width: number; height: number }) => {
    if (anchorPickStep !== 'pickText' || !_filePath) return;

    // Send timestamp directly - no fps assumption needed
    // This properly supports VFR (Variable Frame Rate) videos
    const timestamp = videoState.currentTime;

    console.log('[App] OCR-ing anchor region:', anchorBox);
    setToastMessage('🔍 Detecting text in region...');

    getTextInRegion(_filePath, timestamp, anchorBox, (text, region) => {
      console.log('[App] Got region text:', text);
      if (text && text.trim().length > 0) {
        setPickedAnchorText(text);
        setPickedAnchorBbox(region);
        setAnchorPickStep('drawBox');
        setToastMessage(`📍 "${text}" detected! Now draw the blur box...`);
      } else {
        setToastMessage('No text found in selected region');
        setAnchorPickStep('idle');
        setTimeout(() => setToastMessage(null), 3000);
      }
    });
  }, [anchorPickStep, _filePath, videoState.currentTime]);

  // Handle blur box drawn for anchor
  const handleBlurBoxDrawnForAnchor = useCallback((blurBox: { x: number; y: number; width: number; height: number }) => {
    if (anchorPickStep !== 'drawBox' || !pickedAnchorText || !pickedAnchorBbox) return;

    // Calculate edges for direction detection
    const anchorLeft = pickedAnchorBbox.x;
    const anchorRight = pickedAnchorBbox.x + pickedAnchorBbox.width;
    const anchorTop = pickedAnchorBbox.y;
    const anchorBottom = pickedAnchorBbox.y + pickedAnchorBbox.height;

    const blurLeft = blurBox.x;
    const blurRight = blurBox.x + blurBox.width;
    const blurTop = blurBox.y;
    const blurBottom = blurBox.y + blurBox.height;

    let direction: 'BELOW' | 'RIGHT' | 'ABOVE' | 'LEFT';
    let gap: number;

    // Check edge relationships to determine direction
    // Priority: vertical first (BELOW/ABOVE), then horizontal (RIGHT/LEFT)
    if (blurTop >= anchorBottom) {
      // Blur box is clearly below anchor
      direction = 'BELOW';
      gap = Math.round((blurTop - anchorBottom) * 1000);
    } else if (blurBottom <= anchorTop) {
      // Blur box is clearly above anchor
      direction = 'ABOVE';
      gap = Math.round((anchorTop - blurBottom) * 1000);
    } else if (blurLeft >= anchorRight) {
      // Blur box is to the right
      direction = 'RIGHT';
      gap = Math.round((blurLeft - anchorRight) * 1000);
    } else if (blurRight <= anchorLeft) {
      // Blur box is to the left
      direction = 'LEFT';
      gap = Math.round((anchorLeft - blurRight) * 1000);
    } else {
      // Overlapping - check how much the blur extends beyond anchor in each direction
      const extendBelow = Math.max(0, blurBottom - anchorBottom);
      const extendAbove = Math.max(0, anchorTop - blurTop);
      const extendRight = Math.max(0, blurRight - anchorRight);
      const extendLeft = Math.max(0, anchorLeft - blurLeft);

      // PRIORITIZE VERTICAL over horizontal (labels are typically above/below inputs)
      if (extendBelow > 0 || extendAbove > 0) {
        // There's vertical extension - use that
        if (extendBelow >= extendAbove) {
          direction = 'BELOW';
        } else {
          direction = 'ABOVE';
        }
        gap = 0;
      } else if (extendRight > 0 || extendLeft > 0) {
        // Only horizontal extension
        if (extendRight >= extendLeft) {
          direction = 'RIGHT';
        } else {
          direction = 'LEFT';
        }
        gap = 0;
      } else {
        // Complete overlap - default to BELOW
        direction = 'BELOW';
        gap = 0;
      }
    }
    // Convert normalized dimensions to pixel values using ACTUAL video dimensions
    const videoElement = videoPlayerRef.current?.getVideoElement();
    const videoWidth = videoElement?.videoWidth || 1920;
    const videoHeight = videoElement?.videoHeight || 1080;
    const width = Math.round(blurBox.width * videoWidth);
    const height = Math.round(blurBox.height * videoHeight);

    // Create the new anchor
    const newAnchor: Anchor = {
      id: `anchor-${Date.now()}`,
      label: pickedAnchorText,
      direction,
      gap: Math.max(0, gap),
      width,
      height,
      enabled: true
    };

    // Add anchor and reset state
    setAnchors(prev => [...prev, newAnchor]);
    setAnchorPickStep('idle');
    setPickedAnchorText(null);
    setPickedAnchorBbox(null);
    setToastMessage(`✅ Anchor "${pickedAnchorText}" added with ${direction} blur box!`);
    setTimeout(() => setToastMessage(null), 3000);
  }, [anchorPickStep, pickedAnchorText, pickedAnchorBbox]);


  // Export handler
  const handleExport = useCallback(async () => {
    if (!_filePath) {
      setToastMessage('❌ No video loaded');
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }

    const redactedDetections = detections.filter(d => d.isRedacted);
    const enabledAnchors = anchors.filter(a => a.enabled);

    if (redactedDetections.length === 0 && enabledAnchors.length === 0) {
      setToastMessage('❌ No redactions to export. Please add detections or anchors.');
      setTimeout(() => setToastMessage(null), 4000);
      return;
    }

    // Show save dialog for user to choose output location
    const ext = _filePath.split('.').pop() || 'mp4';
    const baseName = _filePath.split(/[/\\]/).pop()?.replace(/\.[^.]+$/, '') || 'video';
    const defaultName = `${baseName}_redacted.${ext}`;

    const outputPath = await save({
      title: 'Export Redacted Video',
      defaultPath: defaultName,
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'mov', 'mkv', 'avi', 'webm'] }
      ]
    });

    if (!outputPath) {
      // User cancelled the save dialog
      return;
    }

    // Import startExport dynamically to avoid circular deps
    const { startExport } = await import('./lib/tauri');

    // Prepare detections for export
    const exportDetections = redactedDetections.map(d => ({
      id: d.id,
      type: d.type,
      content: d.content,
      confidence: d.confidence,
      bbox: d.bbox,
      startTime: d.startTime,
      endTime: d.endTime,
      frameStart: Math.floor(d.startTime * 30),  // Approximate frame from time
      frameEnd: Math.floor(d.endTime * 30),
      isRedacted: d.isRedacted,
      trackId: d.trackId,
      framePositions: d.framePositions  // Per-frame positions for motion blur
    }));

    // Prepare anchors for export
    const exportAnchors = enabledAnchors.map(a => ({
      id: a.id,
      label: a.label,
      direction: a.direction,
      gap: a.gap,
      width: a.width,
      height: a.height,
      enabled: a.enabled
    }));

    // Extract enabled watch list items as strings
    const enabledWatchItems = watchList
      .filter(w => w.enabled)
      .map(w => w.text);

    try {
      await startExport(
        _filePath,
        exportDetections,
        exportAnchors,
        enabledWatchItems,  // Pass watch list
        {
          scanInterval: appSettings.scanInterval,
          motionThreshold: appSettings.motionThreshold,
          ocrScale: appSettings.ocrScale,
          scanZones: scanZones.filter(z => z.enabled).map(z => ({ start: z.start, end: z.end })),
          codec: appSettings.exportCodec,
          quality: appSettings.exportQuality,
        },
        outputPath,  // User-selected output path
        {
          onProgress: (progress, stage) => {
            console.log(`[Export] ${progress.toFixed(1)}% - ${stage}`);
            // Extract ETA from stage message if present (format: "... ETA: 123s")
            const etaMatch = stage.match(/ETA:\s*(\d+)s/);
            let eta = '';
            if (etaMatch && progress < 99) {
              const seconds = parseInt(etaMatch[1]);
              if (seconds > 0) {
                eta = seconds >= 60 ? `${Math.floor(seconds / 60)}m ${seconds % 60}s` : `${seconds}s`;
              }
            }
            setExportProgress({
              isExporting: true,
              progress,
              stage: stage.replace(/\s*\(.*\)/, '').replace(/, ETA:.*/, ''),  // Clean up stage text
              eta
            });
          },
          onComplete: (completedPath, success) => {
            setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
            if (success) {
              setToastMessage(`✅ Export complete! Saved to: ${completedPath}`);
              setTimeout(() => setToastMessage(null), 5000);
            } else {
              setToastMessage('❌ Export failed. Check console for details.');
              setTimeout(() => setToastMessage(null), 4000);
            }
          },
          onError: (error) => {
            setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
            console.error('[Export] Error:', error);
            setToastMessage(`❌ Export failed: ${error}`);
            setTimeout(() => setToastMessage(null), 4000);
          }
        }
      );
    } catch (err) {
      setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
      console.error('[App] Export error:', err);
      alert('Failed to start export. Is the Python sidecar running?');
    }
  }, [_filePath, detections, anchors, watchList, appSettings, scanZones]);

  // Scan handler - runs OCR scan WITHOUT encoding video
  const handleScan = useCallback(async () => {
    if (!_filePath) {
      setToastMessage('❌ No video loaded');
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }

    const enabledAnchors = anchors.filter(a => a.enabled);

    if (enabledAnchors.length === 0 && watchList.length === 0) {
      setToastMessage('❌ No anchors or watch list items. Add at least one to scan.');
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }

    setToastMessage('🔍 Scanning video for blur regions...');

    try {
      await startScan(
        _filePath,
        enabledAnchors,
        watchList.map(w => w.text),  // Convert WatchItem[] to string[]
        {
          scanInterval: appSettings.scanInterval,
          motionThreshold: appSettings.motionThreshold,
          scanZones: scanZones.map(z => ({ start: z.start, end: z.end })),
        },
        {
          onProgress: (progress, stage) => {
            setExportProgress({ isExporting: true, progress, stage, eta: '' });
          },
          onComplete: (detectedBlurs) => {
            setExportProgress({ isExporting: false, progress: 100, stage: 'Scan complete!', eta: '' });

            if (detectedBlurs && detectedBlurs.length > 0) {
              // Convert to Detection objects
              const ocrDetections: Detection[] = detectedBlurs.map((blur, index) => ({
                id: `scan-${blur.type}-${blur.frame}-${index}`,
                type: blur.type as 'anchor' | 'watchlist',
                content: blur.source,
                confidence: 1.0,
                bbox: blur.bbox,
                startTime: blur.time,
                endTime: blur.end_time || blur.time + 1 / 30,
                isRedacted: true,
                trackId: undefined,
                framePositions: blur.frame_positions  // Motion tracking positions from scan
              }));

              // Replace old OCR/scan detections with new scan results
              // This prevents old detections with incorrect coordinates from showing
              setDetections(prev => {
                // Keep only manual detections (not from OCR/scan)
                const manualDetections = prev.filter(d =>
                  !d.id.startsWith('ocr-') && !d.id.startsWith('scan-')
                );
                console.log(`[App] Replacing ${prev.length - manualDetections.length} old scan detections with ${ocrDetections.length} new ones`);
                return [...manualDetections, ...ocrDetections];
              });

              setToastMessage(`✅ Scan complete! Found ${detectedBlurs.length} blur regions. Toggle any false positives, then export.`);
            } else {
              setToastMessage('✅ Scan complete! No blur regions detected. Add manual blur boxes or check your anchors.');
            }

            // Auto-switch to detections tab to show results
            setSidebarTab('detections');

            setTimeout(() => setToastMessage(null), 5000);
          },
          onError: (error) => {
            setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
            console.error('[Scan] Error:', error);
            setToastMessage(`❌ Scan failed: ${error}`);
            setTimeout(() => setToastMessage(null), 4000);
          }
        }
      );
    } catch (err) {
      setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' });
      console.error('[App] Scan error:', err);
      setToastMessage('❌ Failed to start scan. Is the Python sidecar running?');
      setTimeout(() => setToastMessage(null), 4000);
    }
  }, [_filePath, anchors, watchList, appSettings, scanZones]);

  // Save project handler - saves current state to .screensafe file
  const handleSaveProject = useCallback(async () => {
    if (!_filePath) {
      setToastMessage('No video loaded to save');
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }

    try {
      const savePath = await save({
        title: 'Save Project',
        filters: [{ name: 'ScreenSafe Project', extensions: ['screensafe'] }],
        defaultPath: 'project.screensafe'
      });

      if (savePath) {
        const project = {
          version: '1.0',
          videoPath: _filePath,
          watchList,
          anchors,
          scanZones,
          exportConfig,
          detections: detections.map(d => ({
            ...d,
            // Exclude framePositions from save to reduce file size
            framePositions: undefined
          }))
        };
        await writeTextFile(savePath, JSON.stringify(project, null, 2));
        setHasUnsavedChanges(false);
        const fileName = savePath.split(/[\/\\]/).pop() || 'project';
        setToastMessage(`✅ Project saved to "${fileName}"`);
      }
    } catch (err) {
      console.error('[App] Error saving project:', err);
      setToastMessage(`❌ Failed to save project: ${err}`);
    }
    setTimeout(() => setToastMessage(null), 3000);
  }, [_filePath, watchList, anchors, scanZones, exportConfig, detections]);

  // Open project handler - loads state from .screensafe file
  const handleOpenProject = useCallback(async () => {
    try {
      const projectPath = await open({
        title: 'Open Project',
        multiple: false,
        filters: [{ name: 'ScreenSafe Project', extensions: ['screensafe'] }]
      });

      if (projectPath && typeof projectPath === 'string') {
        const content = await readTextFile(projectPath);
        const project = JSON.parse(content);

        // Restore state
        if (project.videoPath) {
          handlePathSelect(project.videoPath);
        }
        if (project.watchList) setWatchList(project.watchList);
        if (project.anchors) setAnchors(project.anchors);
        if (project.scanZones) setScanZones(project.scanZones);
        if (project.exportConfig) setExportConfig(project.exportConfig);
        if (project.detections) setDetections(project.detections);

        const fileName = projectPath.split(/[\/\\]/).pop() || 'project';
        setToastMessage(`✅ Loaded project "${fileName}"`);
      }
    } catch (err) {
      console.error('[App] Error loading project:', err);
      setToastMessage(`❌ Failed to load project: ${err}`);
    }
    setTimeout(() => setToastMessage(null), 3000);
  }, [handlePathSelect]);

  // Keyboard shortcuts for video playback and common actions
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture when typing in input fields
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      // Space - Play/Pause
      if (e.key === ' ' && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        if (videoPlayerRef.current) {
          if (videoState.isPlaying) {
            videoPlayerRef.current.pause();
          } else {
            videoPlayerRef.current.play();
          }
        }
      }

      // Arrow keys handled by frame-by-frame navigation useEffect below

      // Ctrl+S or Cmd+S - Save preset
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault();
        handleSavePreset('preset');
      }

      // Ctrl+E or Cmd+E - Export video
      if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        if (_filePath && mode === 'editing') {
          handleExport();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [videoState.isPlaying, videoState.currentTime, videoState.duration, _filePath, mode, handleSavePreset, handleExport]);

  // Selection mode toggle
  const handleSelectionModeToggle = useCallback(() => {
    setIsSelectionMode(prev => !prev);
  }, []);

  // Handle resize start (cache original detection for delta calculation)
  const dragStartDetection = useRef<Detection | null>(null);

  const handleResizeStart = useCallback((detectionId: string) => {
    const d = detections.find(d => d.id === detectionId);
    if (d) {
      dragStartDetection.current = JSON.parse(JSON.stringify(d));
    }
  }, [detections]);

  const handleResizeEnd = useCallback((detectionId: string) => {
    const original = dragStartDetection.current;
    const current = detections.find(d => d.id === detectionId);

    if (original && current && original.id === detectionId && original.framePositions?.length) {
      const dx = current.bbox.x - original.bbox.x;
      const dy = current.bbox.y - original.bbox.y;
      const dw = current.bbox.width - original.bbox.width;
      const dh = current.bbox.height - original.bbox.height;

      const newFramePositions = original.framePositions.map(([f, x, y, w, h]) => [
        f,
        x + dx,
        y + dy,
        w + dw,
        h + dh
      ] as [number, number, number, number, number]);

      setDetections(prev => prev.map(d =>
        d.id === detectionId ? { ...d, framePositions: newFramePositions } : d
      ));
    }
    dragStartDetection.current = null;
  }, [detections]);

  // Handle bbox change from spatial resize
  const handleBboxChange = useCallback((detectionId: string, bbox: { x: number; y: number; width: number; height: number }) => {
    setDetections(prev => prev.map(d =>
      d.id === detectionId ? { ...d, bbox, framePositions: undefined } : d
    ));
    setHasUnsavedChanges(true);
  }, []);

  // Add blur from manual selection
  const handleAddBlur = useCallback(async (bbox: { x: number; y: number; width: number; height: number }, frameTime: number) => {
    const duration = videoState.duration > 0 ? videoState.duration : 999999;

    const detectionId = `${selectionType}-${Date.now()}`;
    const isBlackout = selectionType === 'blackout';
    const newDetection: Detection = {
      id: detectionId,
      type: isBlackout ? 'blackout' : 'manual',
      content: enableContentDetection
        ? (isBlackout ? 'Blackout region' : 'Analyzing content...')
        : (isBlackout ? 'Blackout (static)' : 'Static blur'),
      confidence: 1.0,
      bbox,
      startTime: Math.max(0, frameTime - 0.1),  // Start 100ms before selection to ensure PII is fully covered
      endTime: duration,     // Until end (will be refined by tracking if enabled)
      isRedacted: true,
      trackId: undefined
    };
    console.log('[App] Adding manual blur:', { id: detectionId, bbox, startTime: frameTime, contentDetection: enableContentDetection });
    setDetections(prev => [...prev, newDetection]);
    setHasUnsavedChanges(true);
    setIsSelectionMode(false);

    // Start OCR-based content tracking to find exact time range (only if enabled)
    if (_filePath && enableContentDetection) {
      try {
        const { trackRegion } = await import('./lib/tauri');
        // Send timestamp directly - no fps assumption (VFR support)
        const timestamp = frameTime;

        console.log('[App] Starting content detection for', detectionId, 'at', timestamp, 's');

        await trackRegion(_filePath, detectionId, bbox, timestamp, (result) => {
          console.log('[App] Content detection result:', result.framePositions?.length || 0, 'positions');

          // Update detection with tracked time range and motion positions
          setDetections(prev => prev.map(d =>
            d.id === result.detectionId
              ? {
                ...d,
                startTime: result.startTime,
                endTime: result.endTime,
                content: result.framePositions?.length
                  ? `Tracked (${result.framePositions.length} frames)`
                  : 'Manual selection',
                framePositions: result.framePositions
              }
              : d
          ));
        });
      } catch (err) {
        console.error('[App] Content detection error:', err);
        // Update content label even if tracking fails
        setDetections(prev => prev.map(d =>
          d.id === detectionId ? { ...d, content: 'Manual selection' } : d
        ));
      }
    }
  }, [videoState.duration, _filePath, selectionType, enableContentDetection]);

  // Keyboard shortcuts for frame-by-frame navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const FPS = 30; // Approximate FPS
      const frameDuration = 1 / FPS;
      const duration = videoState.duration || 0;
      const currentTime = videoState.currentTime || 0;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          // Step back 1 frame (or 10 frames with Shift)
          const stepBack = e.shiftKey ? frameDuration * 10 : frameDuration;
          handleSeek(Math.max(0, currentTime - stepBack));
          break;

        case 'ArrowRight':
          e.preventDefault();
          // Step forward 1 frame (or 10 frames with Shift)
          const stepForward = e.shiftKey ? frameDuration * 10 : frameDuration;
          handleSeek(Math.min(duration, currentTime + stepForward));
          break;
        // Space bar handled by main keyboard shortcuts useEffect above
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [videoState.currentTime, videoState.duration, videoState.isPlaying, handleSeek, handlePlay, handlePause]);

  // Cleanup URL on unmount
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  // Keep refs in sync with current values for window close handler
  useEffect(() => {
    hasUnsavedChangesRef.current = hasUnsavedChanges;
  }, [hasUnsavedChanges]);

  useEffect(() => {
    handleSaveProjectRef.current = handleSaveProject;
  }, [handleSaveProject]);

  // Listen for window close request to prompt for unsaved changes (set up once)
  useEffect(() => {
    let unlisten: (() => void) | null = null;

    const setupCloseListener = async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window');
        const currentWindow = getCurrentWindow();

        unlisten = await currentWindow.onCloseRequested(async (event) => {
          // Use ref to get current value (avoids stale closure)
          if (hasUnsavedChangesRef.current) {
            // Prevent the window from closing
            event.preventDefault();

            // Show confirmation dialog
            setConfirmDialog({
              isOpen: true,
              title: 'Unsaved Changes',
              message: 'You have unsaved changes. Do you want to save before closing?',
              onConfirm: async () => {
                // Clear unsaved changes flag first
                setHasUnsavedChanges(false);
                hasUnsavedChangesRef.current = false;
                setConfirmDialog(null);
                // Use destroy() to close window immediately without triggering onCloseRequested again
                const { getCurrentWindow } = await import('@tauri-apps/api/window');
                await getCurrentWindow().destroy();
              },
              onSaveAndContinue: async () => {
                setConfirmDialog(null);
                // Save then close
                if (handleSaveProjectRef.current) {
                  await handleSaveProjectRef.current();
                }
                // hasUnsavedChanges should be false after save
                hasUnsavedChangesRef.current = false;
                // Use destroy() to close window immediately without triggering onCloseRequested again
                const { getCurrentWindow } = await import('@tauri-apps/api/window');
                await getCurrentWindow().destroy();
              }
            });
          }
        });
      } catch (err) {
        console.error('Failed to setup close listener:', err);
      }
    };

    setupCloseListener();

    return () => {
      if (unlisten) unlisten();
    };
  }, []); // Empty deps - set up listener once, use refs for current values

  // Show setup dialog if setup is needed
  if (isSetupComplete === false) {
    return <SetupDialog onSetupComplete={handleSetupComplete} />;
  }

  // Show loading while checking setup status
  if (isSetupComplete === null) {
    return (
      <div className="app" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '100vh' }}>
        <div style={{ textAlign: 'center', color: 'var(--color-text-secondary)' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔍</div>
          <p>Checking setup status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      {/* Export Progress Overlay */}
      {exportProgress.isExporting && (
        <div className="export-progress-overlay">
          <div className="export-progress-modal">
            <h3>📹 Exporting Video...</h3>
            <div className="export-progress-bar-container">
              <div
                className="export-progress-bar"
                style={{ width: `${exportProgress.progress}%` }}
              />
            </div>
            <div className="export-progress-info">
              <span className="export-progress-percent">{exportProgress.progress.toFixed(1)}%</span>
              {exportProgress.eta && <span className="export-progress-eta">ETA: {exportProgress.eta}</span>}
            </div>
            <div className="export-progress-stage">{exportProgress.stage}</div>
          </div>
        </div>
      )}

      {/* Confirmation Dialog */}
      {confirmDialog && (
        <div className="export-progress-overlay" onClick={() => setConfirmDialog(null)}>
          <div className="export-progress-modal" onClick={e => e.stopPropagation()} style={{ maxWidth: '400px' }}>
            <h3 style={{ marginBottom: 'var(--space-md)', color: 'var(--color-warning)' }}>
              ⚠️ {confirmDialog.title}
            </h3>
            <p style={{ marginBottom: 'var(--space-lg)', color: 'var(--color-text-secondary)' }}>
              {confirmDialog.message}
            </p>
            <div style={{ display: 'flex', gap: 'var(--space-md)', justifyContent: 'flex-end' }}>
              <button className="btn btn-ghost" onClick={() => setConfirmDialog(null)}>
                Cancel
              </button>
              <button className="btn btn-ghost" onClick={confirmDialog.onConfirm}>
                Discard Changes
              </button>
              {confirmDialog.onSaveAndContinue && (
                <button className="btn btn-primary" onClick={confirmDialog.onSaveAndContinue}>
                  Save & Continue
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">
            <img src="/assets/icon.png" alt="ScreenSafe" />
          </div>
          <span className="header-title">ScreenSafe</span>
        </div>

        <div className="header-actions">
          {mode === 'editing' && (
            <>
              <button className="btn btn-ghost" onClick={handleNewProjectWithConfirm} title="Start a new project (clean slate)">
                🆕 New
              </button>
              <button className="btn btn-ghost" onClick={handleLoadNewVideo} title="Load another video (keep current settings)">
                📼 Load Video
              </button>
              <button className="btn btn-ghost" onClick={handleOpenProject} title="Load a previously saved ScreenSafe project">
                📁 Open Project
              </button>
              <button className="btn btn-ghost" onClick={handleSaveProject} title="Save project with detections and settings">
                💾 Save Project
              </button>
              <button
                className={`btn ${previewBlur ? 'btn-secondary' : 'btn-ghost'}`}
                onClick={() => setPreviewBlur(!previewBlur)}
                title={previewBlur ? 'Blur preview ON - click to disable' : 'Blur preview OFF - click to enable'}
              >
                {previewBlur ? '👁️ Preview ON' : '👁️‍🗨️ Preview OFF'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={handleScan}
                title="Scan video for blur regions (faster than encoding)"
              >
                🔍 Scan
              </button>
              <button className="btn btn-primary" onClick={handleExport} title="Export video with applied redactions">
                📹 Export
              </button>
              <button className="btn btn-ghost" onClick={() => setShowSettingsModal(true)} title="Configure application preferences">
                ⚙️ Settings
              </button>
              <button className="btn btn-ghost" onClick={() => setShowAboutModal(true)} title="About ScreenSafe">
                ℹ️ About
              </button>
              {/* GPU Status Indicator */}
              <div
                className={`gpu-indicator ${gpuStatus.available ? 'gpu-active' : 'gpu-inactive'}`}
                title={gpuStatus.available ? `GPU: ${gpuStatus.name || 'Active'} (${gpuStatus.type || 'Unknown'})` : 'GPU: Not detected (CPU mode)'}
              >
                {gpuStatus.available ? '🟢 GPU' : '⚪ CPU'}
              </div>
            </>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {mode === 'idle' && (
          <DropZone onFileSelect={handleFileSelect} onPathSelect={handlePathSelect} onOpenProject={handleOpenProject} />
        )}

        {mode === 'editing' && (
          <div className="editor">
            <div className="editor-main">
              {/* Video wrapper to scope overlays to video area */}
              <div className="video-wrapper">
                <VideoPlayer
                  ref={videoPlayerRef}
                  videoUrl={videoUrl}
                  detections={detections}
                  currentTime={videoState.currentTime}
                  onTimeUpdate={handleTimeUpdate}
                  onDurationChange={handleDurationChange}
                  onVideoStateChange={handleVideoStateChange}
                  isSelectionMode={isSelectionMode}
                  onSelectionComplete={handleAddBlur}
                  previewBlur={previewBlur}
                  selectedDetection={selectedDetectionId ? detections.find(d => d.id === selectedDetectionId) : null}
                  onBboxChange={handleBboxChange}
                  onResizeStart={handleResizeStart}
                  onResizeEnd={handleResizeEnd}
                  onDetectionClick={handleDetectionClick}
                  onDeselect={() => setSelectedDetectionId(null)}
                />

                {/* Anchor Pick Mode Overlay - Step 1: Draw box around anchor text */}
                {anchorPickStep === 'pickText' && (() => {
                  const videoEl = videoPlayerRef.current?.getVideoElement?.();
                  if (!videoEl) return null;

                  // Get video element rect and compute letterbox dimensions
                  const containerRect = videoEl.getBoundingClientRect();
                  const wrapperEl = videoEl.closest('.video-wrapper');
                  const wrapperRect = wrapperEl?.getBoundingClientRect();
                  const videoWidth = videoEl.videoWidth;
                  const videoHeight = videoEl.videoHeight;

                  if (!videoWidth || !videoHeight || !wrapperRect) return null;

                  const displayedRatio = containerRect.width / containerRect.height;
                  const videoRatio = videoWidth / videoHeight;

                  let displayedWidth: number, displayedHeight: number, offsetX: number, offsetY: number;

                  if (videoRatio > displayedRatio) {
                    // Letterbox top/bottom
                    displayedWidth = containerRect.width;
                    displayedHeight = containerRect.width / videoRatio;
                    offsetX = 0;
                    offsetY = (containerRect.height - displayedHeight) / 2;
                  } else {
                    // Letterbox left/right
                    displayedHeight = containerRect.height;
                    displayedWidth = containerRect.height * videoRatio;
                    offsetX = (containerRect.width - displayedWidth) / 2;
                    offsetY = 0;
                  }

                  // Calculate overlay position relative to wrapper
                  const overlayLeft = containerRect.left - wrapperRect.left + offsetX;
                  const overlayTop = containerRect.top - wrapperRect.top + offsetY;

                  // Simple coordinate conversion: mouse relative to overlay = video coords
                  const getVideoCoords = (e: React.MouseEvent) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    return {
                      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
                      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
                    };
                  };

                  return (
                    <div
                      className="anchor-draw-overlay anchor-step1"
                      style={{
                        position: 'absolute',
                        left: overlayLeft,
                        top: overlayTop,
                        width: displayedWidth,
                        height: displayedHeight,
                      }}
                      onMouseDown={(e) => {
                        const coords = getVideoCoords(e);
                        console.log('[Anchor Step1] Start draw at video coords:', coords);
                        setAnchorDrawBox({ startX: coords.x, startY: coords.y, endX: coords.x, endY: coords.y });
                      }}
                      onMouseMove={(e) => {
                        if (!anchorDrawBox) return;
                        const coords = getVideoCoords(e);
                        setAnchorDrawBox(prev => prev ? { ...prev, endX: coords.x, endY: coords.y } : null);
                      }}
                      onMouseUp={() => {
                        if (!anchorDrawBox) return;
                        const x = Math.min(anchorDrawBox.startX, anchorDrawBox.endX);
                        const y = Math.min(anchorDrawBox.startY, anchorDrawBox.endY);
                        const width = Math.abs(anchorDrawBox.endX - anchorDrawBox.startX);
                        const height = Math.abs(anchorDrawBox.endY - anchorDrawBox.startY);

                        console.log('[Anchor Step1] Finish draw:', { x, y, width, height });

                        if (width > 0.01 && height > 0.01) {
                          handleAnchorBoxDrawn({ x, y, width, height });
                        }
                        setAnchorDrawBox(null);
                      }}
                      onMouseLeave={() => setAnchorDrawBox(null)}
                    >
                      {/* Show drawing rectangle for anchor box - simple percentage coords */}
                      {anchorDrawBox && (
                        <div
                          className="anchor-draw-rect anchor-box"
                          style={{
                            left: `${Math.min(anchorDrawBox.startX, anchorDrawBox.endX) * 100}%`,
                            top: `${Math.min(anchorDrawBox.startY, anchorDrawBox.endY) * 100}%`,
                            width: `${Math.abs(anchorDrawBox.endX - anchorDrawBox.startX) * 100}%`,
                            height: `${Math.abs(anchorDrawBox.endY - anchorDrawBox.startY) * 100}%`,
                          }}
                        />
                      )}
                    </div>
                  );
                })()}


                {/* Anchor Pick Mode - Step 2: Draw blur box */}
                {anchorPickStep === 'drawBox' && (() => {
                  const videoEl = videoPlayerRef.current?.getVideoElement?.();
                  if (!videoEl) return null;

                  // Get video element rect and compute letterbox dimensions
                  const containerRect = videoEl.getBoundingClientRect();
                  const wrapperEl = videoEl.closest('.video-wrapper');
                  const wrapperRect = wrapperEl?.getBoundingClientRect();
                  const videoWidth = videoEl.videoWidth;
                  const videoHeight = videoEl.videoHeight;

                  if (!videoWidth || !videoHeight || !wrapperRect) return null;

                  const displayedRatio = containerRect.width / containerRect.height;
                  const videoRatio = videoWidth / videoHeight;

                  let displayedWidth: number, displayedHeight: number, offsetX: number, offsetY: number;

                  if (videoRatio > displayedRatio) {
                    displayedWidth = containerRect.width;
                    displayedHeight = containerRect.width / videoRatio;
                    offsetX = 0;
                    offsetY = (containerRect.height - displayedHeight) / 2;
                  } else {
                    displayedHeight = containerRect.height;
                    displayedWidth = containerRect.height * videoRatio;
                    offsetX = (containerRect.width - displayedWidth) / 2;
                    offsetY = 0;
                  }

                  // Calculate overlay position relative to wrapper
                  const overlayLeft = containerRect.left - wrapperRect.left + offsetX;
                  const overlayTop = containerRect.top - wrapperRect.top + offsetY;

                  // Simple coordinate conversion: mouse relative to overlay = video coords
                  const getVideoCoords = (e: React.MouseEvent) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    return {
                      x: Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)),
                      y: Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
                    };
                  };

                  return (
                    <div
                      className="anchor-draw-overlay"
                      style={{
                        position: 'absolute',
                        left: overlayLeft,
                        top: overlayTop,
                        width: displayedWidth,
                        height: displayedHeight,
                      }}
                      onMouseDown={(e) => {
                        const coords = getVideoCoords(e);
                        console.log('[Anchor Step2] Start draw at video coords:', coords);
                        setAnchorDrawBox({ startX: coords.x, startY: coords.y, endX: coords.x, endY: coords.y });
                      }}
                      onMouseMove={(e) => {
                        if (!anchorDrawBox) return;
                        const coords = getVideoCoords(e);
                        setAnchorDrawBox(prev => prev ? { ...prev, endX: coords.x, endY: coords.y } : null);
                      }}
                      onMouseUp={() => {
                        if (!anchorDrawBox) return;
                        const x = Math.min(anchorDrawBox.startX, anchorDrawBox.endX);
                        const y = Math.min(anchorDrawBox.startY, anchorDrawBox.endY);
                        const width = Math.abs(anchorDrawBox.endX - anchorDrawBox.startX);
                        const height = Math.abs(anchorDrawBox.endY - anchorDrawBox.startY);

                        console.log('[Anchor Step2] Finish draw:', { x, y, width, height });

                        if (width > 0.01 && height > 0.01) {
                          handleBlurBoxDrawnForAnchor({ x, y, width, height });
                        }
                        setAnchorDrawBox(null);
                      }}
                      onMouseLeave={() => setAnchorDrawBox(null)}
                    >
                      {/* Show anchor position indicator - simple percentage coords */}
                      {pickedAnchorBbox && (
                        <div
                          className="anchor-position-indicator"
                          style={{
                            left: `${pickedAnchorBbox.x * 100}%`,
                            top: `${pickedAnchorBbox.y * 100}%`,
                            width: `${pickedAnchorBbox.width * 100}%`,
                            height: `${pickedAnchorBbox.height * 100}%`,
                          }}
                        />
                      )}

                      {/* Show drawing rectangle - simple percentage coords */}
                      {anchorDrawBox && (
                        <div
                          className="anchor-draw-rect"
                          style={{
                            left: `${Math.min(anchorDrawBox.startX, anchorDrawBox.endX) * 100}%`,
                            top: `${Math.min(anchorDrawBox.startY, anchorDrawBox.endY) * 100}%`,
                            width: `${Math.abs(anchorDrawBox.endX - anchorDrawBox.startX) * 100}%`,
                            height: `${Math.abs(anchorDrawBox.endY - anchorDrawBox.startY) * 100}%`,
                          }}
                        />
                      )}
                    </div>
                  );
                })()}
              </div>
              {/* End video-wrapper */}

              <VideoControls
                videoState={videoState}
                onPlay={handlePlay}
                onPause={handlePause}
                onSeek={handleSeek}
                onMagicWandClick={() => {
                  setSelectionType('blur');
                  handleSelectionModeToggle();
                }}
                isMagicWandActive={isSelectionMode && selectionType === 'blur'}
                onBlackoutClick={() => {
                  setSelectionType('blackout');
                  if (!isSelectionMode) {
                    handleSelectionModeToggle();
                  }
                }}
                isBlackoutActive={isSelectionMode && selectionType === 'blackout'}
                enableContentDetection={enableContentDetection}
                onContentDetectionToggle={() => setEnableContentDetection(prev => !prev)}
              />
            </div>

            <div className="editor-sidebar">
              <div className="sidebar-tabs">
                <button
                  className={`sidebar-tab ${sidebarTab === 'config' ? 'active' : ''}`}
                  onClick={() => setSidebarTab('config')}
                >
                  🎯 Filters
                </button>
                <button
                  className={`sidebar-tab ${sidebarTab === 'detections' ? 'active' : ''}`}
                  onClick={() => setSidebarTab('detections')}
                >
                  ✓ Matches ({detections.length})
                </button>
              </div>

              {sidebarTab === 'config' ? (
                <ConfigPanel
                  watchList={watchList}
                  anchors={anchors}
                  scanZones={scanZones}
                  videoDuration={videoState.duration}
                  currentTime={videoState.currentTime}
                  onWatchListChange={handleWatchListChange}
                  onAnchorsChange={handleAnchorsChange}
                  onScanZonesChange={setScanZones}
                  onSavePreset={handleSavePreset}
                  onLoadPreset={handleLoadPreset}
                  onPickFromVideo={handlePickFromVideo}
                  isPickingAnchor={anchorPickStep !== 'idle'}
                  pickedAnchorText={pickedAnchorText}
                />
              ) : (
                <DetectionSidebar
                  detections={detections}
                  selectedDetectionId={selectedDetectionId}
                  onDetectionSelect={handleDetectionSelect}
                  onToggleRedaction={handleToggleRedaction}
                  onDeleteDetection={handleDeleteDetection}
                />
              )}
            </div>

            <div className="editor-timeline">
              <Timeline
                detections={detections}
                duration={videoState.duration || 60}
                currentTime={videoState.currentTime}
                onSeek={handleSeek}
                onDetectionClick={handleDetectionClick}
                selectedDetectionId={selectedDetectionId}
                onDetectionTimeChange={handleDetectionTimeChange}
                onToggleRedaction={handleToggleRedaction}
                onDeleteDetection={handleDeleteDetection}
              />
            </div>
          </div>
        )}
      </main>

      {/* Analysis Overlay */}
      {mode === 'analyzing' && (
        <AnalysisOverlay
          progress={analysisProgress}
          onCancel={handleCancelAnalysis}
        />
      )}

      {/* Export/Scan Progress Overlay */}
      {exportProgress.isExporting && (
        <div className="export-progress-overlay">
          <div className="export-progress-card">
            <div className="export-progress-header">
              <h3>🔄 {exportProgress.stage.includes('Scan') ? 'Scanning' : 'Processing'}</h3>
            </div>
            <div className="export-progress-stage">{exportProgress.stage}</div>
            <div className="export-progress-bar">
              <div
                className="export-progress-fill"
                style={{ width: `${exportProgress.progress}%` }}
              />
            </div>
            <div className="export-progress-percent">
              {Math.round(exportProgress.progress)}%
              {exportProgress.eta && <span> • {exportProgress.eta}</span>}
            </div>
            <button
              className="btn btn-ghost"
              style={{ marginTop: 'var(--space-md)' }}
              onClick={() => setExportProgress({ isExporting: false, progress: 0, stage: '', eta: '' })}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Selection Mode Indicator */}
      {isSelectionMode && (
        <div className="selection-mode-active">
          🎯 Selection Mode — Click and drag to select region (Esc to cancel)
        </div>
      )}

      {/* Toast Message */}
      {toastMessage && (
        <div className="toast-notification">
          {toastMessage}
        </div>
      )}

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettingsModal}
        settings={appSettings}
        onSettingsChange={(newSettings) => setAppSettings((prev: typeof appSettings) => ({ ...prev, ...newSettings }))}
        onClose={() => setShowSettingsModal(false)}
      />

      {/* About Modal */}
      {showAboutModal && (
        <div className="settings-modal-overlay" onClick={() => setShowAboutModal(false)}>
          <div className="about-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-header">
              <h2>About</h2>
              <button className="btn btn-ghost btn-icon" onClick={() => setShowAboutModal(false)}>
                ✕
              </button>
            </div>
            <div className="about-content">
              <img src="/assets/icon.png" alt="ScreenSafe" className="about-icon" />
              <h3>ScreenSafe</h3>
              <p className="about-version">Version 1.1.0</p>
              <p className="about-tagline">Privacy-first video redaction tool</p>
              <p className="about-description">
                Automatically detect and blur sensitive information in screen recordings before sharing.
              </p>
              <div className="about-copyright">
                <p>© 2026 xersbtt</p>
                <p>Released under the MIT License</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
