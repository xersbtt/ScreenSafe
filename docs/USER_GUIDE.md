# ScreenSafe User Guide

**ScreenSafe** is a privacy-first video redaction tool designed to automatically detect and blur sensitive information in screen recordings. Whether you need to hide passwords, emails, or customer data, ScreenSafe helps you secure your videos before sharing.

---

## 🚀 Getting Started

### Installation
1.  **Download** the latest release for your platform:
    - **Windows**: `ScreenSafe-vX.X.X-windows-x64-portable.zip` or installer
    - **macOS**: `ScreenSafe-vX.X.X-macos-arm64.dmg` (Apple Silicon) or `macos-x64.dmg` (Intel)
2.  **Run** the executable. No installation wizard is required.
3.  **macOS Note**: For unsigned builds, right-click the app and select "Open" to bypass Gatekeeper.

### First Launch
When you open ScreenSafe, you'll see a clean interface waiting for your content.
-   **Dark Mode** is enabled by default to reduce eye strain.
-   **Python Backend**: The app automatically starts a lightweight Python sidecar for AI analysis (OCR and tracking).

---

## 🖥️ Interface Overview

The main window is divided into four key areas:

1.  **Video Player (Center)**
    *   Your main workspace. View the video, draw manual redactions, and preview blurs in real-time.
    *   **Responsive sizing**: Video adapts to your window size while maintaining aspect ratio.
    *   **Controls**: Play/Pause (`Space`), Skip 10s (`Arrow Keys`), Volume.
    *   **Resize Handles**: Drag corners of any blur box to resize it spatially.

2.  **Configuration Panel (Right Sidebar - "Config" Tab)**
    *   Setup your detection rules before scanning.
    *   **Watch List**: specific text to hunt for.
    *   **Anchors**: relative positions (e.g., "blur the box *below* this label").
    *   **Scan Zones**: limit processing to specific time ranges.

3.  **Detections List (Right Sidebar - "Detections" Tab)**
    *   A list of all found redactions.
    *   Click any item to jump to it in the video.
    *   Toggle individual redactions on/off (👁️ icon).

4.  **Timeline (Bottom)**
    *   Visual overview of the entire video.
    *   **Tracks**: Detections are grouped by type (Email, Watchlist, Manual).
    *   **Editing**: Drag the edges of any colored bar to adjust when the blur starts or stops.

---

## 🛠️ Core Workflows

### 1. Automatic Redaction (Smart Scan)
Use this for detecting specific text (like emails, API keys) or form fields.

#### A. Watch List
Perfect for specific text you know exists in the video.
1.  Go to the **Config** tab.
2.  Under **Watch List**, type the text (e.g., `john.doe@example.com`).
3.  Press **Add**.
4.  *Tip: You can add multiple items separated by commas.*

#### B. Anchors
Perfect for dynamic values that change but are always near a static label (e.g., a "Password" field).
1.  Click **+ Add Anchor** in the Config tab.
2.  **The Easy Way (Pick from Video)**:
    *   Click **Pick**.
    *   **Step 1**: Draw a box around the *Label* text (e.g., the word "Password"). The app will OCR it.
    *   **Step 2**: Draw a box where the *Sensitive Data* appears (e.g., the input field).
    *   The app automatically calculates the direction (e.g., "Below") and gap.
3.  **Manual Mode**: Enter the Label text and configure direction/dimensions manually.

#### C. Run the Scan
1.  Click the **Scan** button in the header.
2.  Wait for the analysis to complete (Extracting -> Detecting -> Tracking).
3.  Review results in the **Detections** tab.
4.  Toggle **👁️ Preview ON/OFF** in the header to compare the original video with the redacted version.

### 2. Manual Redaction
Use this for random objects or when automatic detection misses something.

1.  **Pause** the video where the sensitive content appears.
2.  Select the tool from the toolbar (top left of video):
    *   **Blur Tool** (💧): Standard Gaussian blur.
    *   **Blackout Tool** (⬛): Solid black box (good for complete redaction).
3.  **Draw** a box over the sensitive area.
4.  **Content Detection Toggle**: Click the magic wand icon (✨) to enable/disable automatic content detection. Disable it when blurring non-text elements.
5.  **Motion Tracking**:
    *   The app will attempt to forward-track the object.
    *   If it drifts, pause, resize/move the box, and it will update the path keyframe.

### 3. Using Presets (Save/Load)
Don't reinvent the wheel! If you regularly record the same application (e.g., "Salesforce Weekly Update"), you don't need to re-add anchors and watchlists every time.

1.  **Configure** your Watch List and Anchors as usual.
2.  Scroll to the **bottom** of the Config Panel.
3.  **Save**: Click `💾 Save` to export your settings to a JSON file.
4.  **Load**: Next time, click `📂 Load` to instantly restore your Watch List and Anchors.


---

## 🎨 Refining Results

Automatic detection isn't always 100% perfect. ScreenSafe gives you tools to fix it.

### Adjusting Timing (Timeline)
*   **Trim**: Hover over the left or right edge of a timeline segment. Drag to change start/end times.
*   **Move**: Click and drag the center of a segment to shift it in time.
*   **Seek**: Click any segment to jump the video to that moment.

### Adjusting Position (Spatial)
*   At any point in the video, **click** a blurred region overlay.
*   **Resize**: Drag the white corner handles to change size.
*   **Move**: Drag the box to move it.
*   These changes apply from that frame onward until the next keyframe.

### Selecting & Deleting
*   **Select**: Click a box in the video or an item in the sidebar.
*   **Delete**: Press `Delete` key or click the trash icon in the sidebar to remove a detection.

---

## 📤 Exporting

Once you're happy with the preview:

1.  Click the **Export** button (header).
2.  Review settings:
    *   **Codec**: H.264 (Standard), H.265 (High Efficiency), VP9 (Web).
    *   **Quality**: Low (~2 Mbps), Medium (~4 Mbps), High (~8 Mbps).
    *   **Resolution**: Original, 1080p, 720p, or 480p. Lower resolutions export faster and create smaller files.
3.  **Process**:
    *   ScreenSafe uses **FFmpeg** with **NVENC** (NVIDIA GPU) acceleration if available for fast rendering.
    *   It re-encodes the video, burning in the blurs permanently.

### Cancelling Scans/Exports (v1.1.2+)

If you need to cancel a scan or export:
1.  Click the **Cancel** button in the progress overlay.
2.  Confirm: "Are you sure?" — Choose **Yes, Cancel** or **Continue**.
3.  If detections were found before cancelling, you'll be asked: "Keep Detections?"
    *   **Keep Detections**: Adds the partial results to your project.
    *   **Discard All**: Removes partial detections and returns to editing.

---

## ⚙️ Advanced Settings

Click the **Settings (⚙️)** icon for global configuration.

*   **Scan Interval**: How often the AI checks for text (Frames). Lower = more accurate but slower. Default: `30`.
*   **Motion Threshold**: Sensitivity for tracking and Motion Boost. Default: `30`.
*   **OCR Scale**: Upscales video frames before reading text. Increase to `1.0` or higher if it misses small text.
*   **Auto-detect PII Patterns**: Enable to automatically detect emails, phones, SSNs via regex.

### Scan Optimizations (v1.1.0+)

*   **Adaptive Sampling**: Static frames (< 1% visual change) are automatically skipped to speed up scans.
*   **Motion Boost**: When scrolling or high motion is detected, extra OCR scans are triggered to catch fast-moving text.

## ❓ Troubleshooting

**Q: The blur "drifts" off the object.**
A: The object might be moving too fast. Try manually adjusting the box position at the point it drifts. The tracker will interpolate between your adjustments.

**Q: VFR Video Sync Issues?**
A: ScreenSafe v1.1.0+ uses timestamp-based rendering to handle Variable Frame Rate videos correctly, ensuring blurs stay in sync with the video time.

**Q: App is slow to scan?**
A: Use **Scan Zones** in the Config panel to limit scanning to only the seconds where sensitive data actually appears.
