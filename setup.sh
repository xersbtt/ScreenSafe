#!/bin/bash
# ScreenSafe Setup Script - macOS/Linux
# Run this once after cloning/copying the project

set -e  # Exit on error

echo "🔧 ScreenSafe Setup"
echo "==================="

# Check for Homebrew (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    echo "📦 Installing system dependencies..."
    brew install node rust ffmpeg python@3.11 || true
fi

# Create Python virtual environment
echo ""
echo "🐍 Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Created virtual environment"
else
    echo "✓ Virtual environment already exists"
fi

source .venv/bin/activate

# Install Python dependencies (using pinned requirements for cross-platform compatibility)
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r python/requirements-pinned.txt

# Install PyTorch with MPS support (Apple Silicon)
echo ""
echo "🔥 Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use CPU/MPS version (smaller download)
    pip install torch torchvision
else
    # Linux - check for CUDA
    if command -v nvidia-smi &> /dev/null; then
        pip install torch torchvision
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
fi

# Install EasyOCR (will use MPS on Apple Silicon)
echo ""
echo "👁️ Installing EasyOCR..."
pip install easyocr

# Install Node dependencies
echo ""
echo "📦 Installing Node.js packages..."
npm install

# Success!
echo ""
echo "✅ Setup complete!"
echo ""
echo "To run ScreenSafe:"
echo "  ./dev.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  npm run tauri dev"
