# Changelog

All notable changes to ScreenSafe will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
