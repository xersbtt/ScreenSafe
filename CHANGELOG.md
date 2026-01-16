# Changelog

All notable changes to ScreenSafe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.1] - 2026-01-16

### Added
- **Include Audio Toggle**: New setting to export video with or without audio track. Useful for creating silent videos or when audio is not needed.
- **VFR (Variable Frame Rate) Support**: Improved compatibility across the export pipeline.

### Fixed
- **macOS H.265 Export**: Fixed H.265 encoding on macOS by using `hevc_videotoolbox` hardware encoder instead of missing `libx265`.
- **macOS FFmpeg Detection**: Fixed FFmpeg not being found when running from the app bundle. Now checks common Homebrew paths (`/opt/homebrew/bin`, `/usr/local/bin`).
- **App Crash on Close**: Added `CloseRequested` event handler to properly stop the Python sidecar before WebView destruction.
- **Python 3.9 Compatibility**: Added `from __future__ import annotations` for type hint compatibility with Python 3.9.6.
- **Encoder Detection**: Fixed encoder detection to check both stdout and stderr from FFmpeg.

### Changed
- **Encoder Selection**: Dynamic encoder detection now supports macOS VideoToolbox, NVIDIA NVENC, and software fallback encoders.
- **FFmpeg Path Resolution**: Uses detected FFmpeg path globally instead of relying on PATH environment variable.

---

## [1.1.0] - 2026-01-16

### Added
- **macOS Support**: Added Apple Silicon (M1/M2/M3/M4) support with MPS (Metal Performance Shaders) GPU acceleration.
- **Cross-platform GPU Detection**: Backend automatically detects and uses the best available GPU backend (CUDA for NVIDIA, MPS for Apple Silicon, CPU fallback). Status displayed in UI header.
- **Adaptive Sampling**: Intelligent frame skipping during scans - static frames (<1% visual change) are skipped unless 2 seconds have passed, dramatically improving scan speed on videos with pauses or slow sections.
- **Motion Boost**: When high motion is detected (scrolling, transitions), extra OCR scans are triggered to capture fast-moving text that might otherwise be missed.
- **Content Detection Toggle**: Manual blur/blackout tools can now optionally skip automatic content detection, useful for blurring non-text elements.
- **Responsive Video Preview**: Video player now adapts to window size, expanding/shrinking dynamically while maintaining aspect ratio.

### Fixed
- **VFR Video Support**: Improved Variable Frame Rate video handling by using timestamp-based operations instead of frame-based, preventing sync issues.

### Changed
- **App Icons**: Updated application icons with improved design.
- **Scan Performance**: Combined Adaptive Sampling and Motion Boost provide faster scans with better accuracy - static content scanned less frequently, moving content scanned more frequently.
- **Settings Storage**: Settings now saved to a proper config file in the user's app data directory instead of browser storage. OCR cache also moved to user cache directory for cross-platform consistency.

---

## [1.0.1] - 2026-01-12

### Fixed
- **Export**: Fixed VFR video sync issues ("ghost blurs") by switching to pure timestamp-based rendering.
- **Export**: Fixed blackout regions rendering as blurs instead of solid black boxes.
- **Playback**: Fixed video reloading bug where loading a second video resulted in a black screen.
- **Timeline**: Fixed navigation - clicking a segment now seeks to the exact click position.
- **Timeline**: Fixed resizing - dragging segment edges no longer causes the playhead to jump.
- **UI**: Fixed "Unsaved Changes" prompt reliability when closing app or loading new files.

### Changed
- **UI**: Sidebar tabs now remain fixed at the top while scrolling content.
- **UI**: Sidebar automatically switches to "Detections" tab when selecting an item or completing a scan.
- **UI**: Renamed "Advanced Export Settings" to "Advanced Scan Settings" for clarity.

## [1.0.0] - 2026-01-12

### Added
- **Core Features**
  - Video loading with drag-and-drop support
  - Real-time blur preview in video player
  - GPU-accelerated export with NVIDIA NVENC

- **Detection Methods**
  - Watchlist-based text detection
  - Anchor-based detection (blur relative to label text)
  - Manual blur tool with motion tracking
  - Blackout tool for solid black overlays

- **Timeline & Editing**
  - Multi-track redaction timeline grouped by detection type
  - Draggable segment edges for time range adjustment
  - Click-to-select detection segments
  - Spatial resize handles on video overlays
  - Scan zones for targeted analysis

- **Project Management**
  - Save/load project files (.json)
  - Unsaved changes warning on close
  - Settings persistence

- **UI/UX**
  - Modern dark theme with glassmorphism
  - Keyboard shortcuts (Space, Arrow keys)
  - Context menus for detection actions
  - Auto-scroll sidebar to selected detection
  - Comprehensive tooltips on all controls

### Technical
- Built with Tauri + React + TypeScript
- Python sidecar for OCR and motion tracking
- EasyOCR integration for text detection
- FFmpeg for video processing
