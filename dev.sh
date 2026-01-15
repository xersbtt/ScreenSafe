#!/bin/bash
# ScreenSafe Development Server - macOS/Linux
# Starts the Tauri app with Python sidecar

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ No .venv found. Run setup.sh first or create venv manually:"
    echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check for required tools
if ! command -v npm &> /dev/null; then
    echo "✗ npm not found. Install Node.js: brew install node"
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "✗ cargo not found. Install Rust: brew install rust"
    exit 1
fi

echo "🚀 Starting ScreenSafe..."
npm run tauri dev
