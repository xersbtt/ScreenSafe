#!/bin/bash

# ================================================
# ScreenSafe Release Build Script (macOS)
# ================================================

set -e

echo "================================================"
echo "ScreenSafe Release Build Script (macOS)"
echo "================================================"
echo ""

# Get version from tauri.conf.json
VERSION=$(grep '"version"' src-tauri/tauri.conf.json | head -1 | sed 's/.*: *"\([^"]*\)".*/\1/')
echo "Building version: v$VERSION"
echo ""

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    TARGET="aarch64-apple-darwin"
    ARCH_NAME="arm64"
else
    TARGET="x86_64-apple-darwin"
    ARCH_NAME="x64"
fi
echo "Target architecture: $ARCH_NAME ($TARGET)"
echo ""

# Clean previous builds
echo "[1/6] Cleaning previous builds..."
rm -rf release
mkdir -p release

# Build the Tauri app
echo "[2/6] Building Tauri app (this may take a few minutes)..."
npm run tauri build -- --target $TARGET

# Find the .app bundle
echo "[3/6] Locating app bundle..."
APP_PATH=$(find src-tauri/target/$TARGET/release/bundle/macos -name "*.app" 2>/dev/null | head -1)
if [ -z "$APP_PATH" ]; then
    echo "ERROR: Could not find .app bundle!"
    exit 1
fi
echo "Found: $APP_PATH"

# Bundle Python sidecar into the app
echo "[4/6] Bundling Python sidecar..."
cp -R python "$APP_PATH/Contents/MacOS/python"
echo "Copied Python sidecar into $APP_PATH/Contents/MacOS/python"

# Copy assets if they exist
if [ -d "assets" ]; then
    cp -R assets "$APP_PATH/Contents/MacOS/assets"
    echo "Copied assets into $APP_PATH/Contents/MacOS/assets"
fi

# Rebuild the DMG with the modified .app bundle
echo "[5/6] Creating DMG..."
DMG_DIR="src-tauri/target/$TARGET/release/bundle/dmg"
rm -f "$DMG_DIR"/*.dmg
hdiutil create -volname "ScreenSafe" -srcfolder "$(dirname $APP_PATH)" -ov -format UDZO "$DMG_DIR/ScreenSafe-$ARCH_NAME.dmg"
cp "$DMG_DIR/ScreenSafe-$ARCH_NAME.dmg" "release/ScreenSafe-v$VERSION-macos-$ARCH_NAME.dmg"

# Create portable ZIP
echo "[6/6] Creating portable ZIP..."
mkdir -p release/temp
cp -R "$APP_PATH" release/temp/
cd release
zip -r "ScreenSafe-v$VERSION-macos-$ARCH_NAME-portable.zip" temp
rm -rf temp
cd ..

echo ""
echo "================================================"
echo "Build Complete!"
echo "================================================"
echo ""
echo "Release files created in: release/"
ls -la release/
echo ""
echo "Ready to upload to GitHub release!"
echo ""
echo "NOTE: For a universal release, run this script on both:"
echo "  - Apple Silicon Mac (M1/M2/M3/M4)"
echo "  - Intel Mac (or use Rosetta)"
