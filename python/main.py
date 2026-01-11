"""
ScreenSafe AI Sidecar - WebSocket Server

Main entry point for the Python AI sidecar.
Communicates with Tauri frontend via WebSocket.
"""

import asyncio
import json
import logging
import sys
import signal
from pathlib import Path

# Add analysis module to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    VideoAnalyzer, AnalysisSettings, AnalysisProgress, 
    Detection, WSMessageType, WSMessage
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("screensafe")

# Global state
analyzer: VideoAnalyzer | None = None
current_task: asyncio.Task | None = None

# PII Config (watch_list, anchors)
pii_config = {
    "watch_list": [],
    "anchors": {}
}


def load_pii_config(config_path: str = None) -> dict:
    """Load PII detection config from JSON file"""
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent / "config.json"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {"watch_list": [], "anchors": {}}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        watch_list = data.get("watch_list", [])
        
        # Parse anchors (convert from [dir, gap, w, h] to dict format)
        anchors = {}
        for label, config in data.get("anchors", {}).items():
            if isinstance(config, list) and len(config) >= 4:
                anchors[label.lower()] = {
                    "direction": config[0],
                    "gap": config[1],
                    "width": config[2],
                    "height": config[3]
                }
            elif isinstance(config, dict):
                anchors[label.lower()] = config
        
        logger.info(f"Loaded PII config: {len(watch_list)} watch items, {len(anchors)} anchors")
        return {"watch_list": watch_list, "anchors": anchors}
    
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"watch_list": [], "anchors": {}}


async def handle_message(websocket, message: str):
    """Handle incoming WebSocket message from Tauri"""
    global analyzer, current_task, pii_config
    
    try:
        data = json.loads(message)
        msg_type = data.get("type")
        payload = data.get("payload", {})
        
        logger.info(f"Received message: {msg_type}")
        logger.info(f"  Expected TRACK_OBJECT value: {WSMessageType.TRACK_OBJECT.value}")
        logger.info(f"  Match: {msg_type == WSMessageType.TRACK_OBJECT.value}")
        
        # Branch check
        logger.info(f"  Checking START_ANALYSIS: {msg_type == WSMessageType.START_ANALYSIS.value}")
        
        if msg_type == WSMessageType.START_ANALYSIS.value:
            logger.info(">>> Entering START_ANALYSIS branch")
            # Start video analysis
            video_path = payload.get("video_path")
            settings_dict = payload.get("settings", {})
            
            if not video_path:
                await send_error(websocket, "No video path provided")
                return
            
            # Load PII config (watch_list, anchors)
            global pii_config
            pii_config = load_pii_config()
            
            # Create settings
            settings = AnalysisSettings(**settings_dict) if settings_dict else AnalysisSettings()
            
            # Initialize analyzer with PII config
            analyzer = VideoAnalyzer(
                settings=settings,
                watch_list=pii_config["watch_list"],
                anchors=pii_config["anchors"]
            )
            use_gpu = payload.get("use_gpu", True)  # Default to True for speed
            analyzer.initialize(use_gpu=use_gpu)
            
            logger.info(f"Analyzer initialized with {len(pii_config['watch_list'])} watch items, {len(pii_config['anchors'])} anchors")
            
            # Start analysis in background
            current_task = asyncio.create_task(
                run_analysis(websocket, video_path)
            )
            
        elif msg_type == WSMessageType.CANCEL_ANALYSIS.value:
            # Cancel ongoing analysis
            if analyzer:
                analyzer.cancel()
            if current_task and not current_task.done():
                current_task.cancel()
            
            logger.info("Analysis cancelled by user")
        
        elif msg_type == WSMessageType.START_EXPORT.value:
            # Start video export with redactions
            video_path = payload.get("video_path")
            output_path = payload.get("output_path")
            detections_data = payload.get("detections", [])
            anchors_data = payload.get("anchors", [])
            watch_list = payload.get("watch_list", [])
            scan_interval = payload.get("scan_interval", 90)
            motion_threshold = payload.get("motion_threshold", 30.0)
            ocr_scale = payload.get("ocr_scale", 1.0)
            scan_zones = payload.get("scan_zones", [])
            codec = payload.get("codec", "h264")
            quality = payload.get("quality", "high")
            preview_mode = payload.get("preview", False)  # Low-res preview mode
            
            if not video_path:
                await send_error(websocket, "No video path provided for export")
                return
            
            if not output_path:
                # Default output path
                from pathlib import Path
                vp = Path(video_path)
                suffix = "_preview" if preview_mode else "_redacted"
                output_path = str(vp.parent / f"{vp.stem}{suffix}{vp.suffix}")
            
            mode_str = "PREVIEW" if preview_mode else "FULL"
            logger.info(f"Starting {mode_str} export: {video_path} -> {output_path}")
            logger.info(f"Config: {len(detections_data)} detections, {len(anchors_data)} anchors, {len(watch_list)} watch items")
            logger.info(f"Settings: scan_interval={scan_interval}, motion_threshold={motion_threshold}, codec={codec}, quality={quality}")
            
            # Start export in background
            current_task = asyncio.create_task(
                run_export(
                    websocket, video_path, output_path, detections_data, anchors_data,
                    watch_list, scan_interval, motion_threshold, ocr_scale, scan_zones,
                    codec, quality, preview_mode
                )
            )
        
        elif msg_type == WSMessageType.TRACK_OBJECT.value:
            # Track a region bidirectionally through the video
            video_path = payload.get("video_path")
            bbox = payload.get("bbox")  # Normalized {x, y, width, height}
            frame_number = payload.get("frame_number", 0)
            detection_id = payload.get("detection_id")
            
            logger.info(f"=== TRACK_OBJECT received ===")
            logger.info(f"  video_path: {video_path}")
            logger.info(f"  bbox: {bbox}")
            logger.info(f"  frame_number: {frame_number}")
            logger.info(f"  detection_id: {detection_id}")
            
            if not video_path or not bbox:
                await send_error(websocket, "Missing video_path or bbox for tracking")
                return
            
            logger.info(f"Starting region tracking: frame {frame_number}, bbox {bbox}")
            
            # Start tracking in background
            current_task = asyncio.create_task(
                run_track(websocket, video_path, bbox, frame_number, detection_id)
            )
        
        elif msg_type == "start_scan":
            # Scan video for blur regions WITHOUT encoding
            video_path = payload.get("video_path")
            anchors_data = payload.get("anchors", [])
            watch_list = payload.get("watch_list", [])
            scan_interval = payload.get("scan_interval", 30)
            motion_threshold = payload.get("motion_threshold", 30.0)
            ocr_scale = payload.get("ocr_scale", 1.0)
            scan_zones = payload.get("scan_zones", [])
            
            if not video_path:
                await send_error(websocket, "No video path provided for scan")
                return
            
            logger.info(f"Starting video scan: {video_path}")
            logger.info(f"Config: {len(anchors_data)} anchors, {len(watch_list)} watch items, scan_interval={scan_interval}")
            
            # Start scan in background
            current_task = asyncio.create_task(
                run_scan(
                    websocket, video_path, anchors_data, watch_list,
                    scan_interval, motion_threshold, ocr_scale, scan_zones
                )
            )
        
        elif msg_type == WSMessageType.UPDATE_CONFIG.value:
            # Update PII wizard configuration (watch list, anchors, zones)
            pii_config = {
                "watch_list": payload.get("watch_list", []),
                "anchors": payload.get("anchors", {}),
                "scan_zones": payload.get("scan_zones", []),
                "enable_keyboard": payload.get("enable_keyboard", False),
                "scan_interval": payload.get("scan_interval", 15),
                "motion_threshold": payload.get("motion_threshold", 3.0),
                "ocr_scale": payload.get("ocr_scale", 0.5)
            }
            logger.info(f"Config updated: {len(pii_config['watch_list'])} watch items, {len(pii_config['anchors'])} anchors")
            
            await send_message(websocket, WSMessageType.PROGRESS_UPDATE, {
                "progress": 100,
                "stage": "Configuration updated"
            })
        
        elif msg_type == WSMessageType.PREVIEW_FRAME.value:
            # Analyze a single frame with current config
            video_path = payload.get("video_path")
            frame_number = payload.get("frame_number", 0)
            
            if not video_path:
                await send_error(websocket, "No video path provided")
                return
            
            # Run preview analysis in background
            current_task = asyncio.create_task(
                run_preview_frame(websocket, video_path, frame_number)
            )
        
        elif msg_type == WSMessageType.GET_TEXT_AT_CLICK.value:
            # Get OCR text at clicked position for anchor selection
            video_path = payload.get("video_path")
            frame_number = payload.get("frame_number", 0)
            click_x = payload.get("click_x", 0.5)  # Normalized 0-1
            click_y = payload.get("click_y", 0.5)  # Normalized 0-1
            
            if not video_path:
                await send_error(websocket, "No video path provided")
                return
            
            # Run text detection in background
            current_task = asyncio.create_task(
                run_get_text_at_click(websocket, video_path, frame_number, click_x, click_y)
            )
        
        elif msg_type == WSMessageType.GET_TEXT_IN_REGION.value:
            # Get OCR text from a drawn region
            video_path = payload.get("video_path")
            frame_number = payload.get("frame_number", 0)
            region = payload.get("region", {})  # {x, y, width, height} normalized
            
            if not video_path:
                await send_error(websocket, "No video path provided")
                return
            
            # Run text detection in background
            current_task = asyncio.create_task(
                run_get_text_in_region(websocket, video_path, frame_number, region)
            )
            
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        await send_error(websocket, f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await send_error(websocket, str(e))


async def run_analysis(websocket, video_path: str):
    """Run analysis and stream progress/results"""
    global analyzer
    
    try:
        # Get the event loop for thread-safe scheduling
        loop = asyncio.get_running_loop()
        
        def on_progress(progress: AnalysisProgress):
            """Send progress update (thread-safe)"""
            # Schedule coroutine from executor thread
            asyncio.run_coroutine_threadsafe(
                send_message(
                    websocket, 
                    WSMessageType.PROGRESS_UPDATE,
                    progress.model_dump()
                ),
                loop
            )
        
        def on_detection(detection: Detection):
            """Send new detection (thread-safe)"""
            asyncio.run_coroutine_threadsafe(
                send_message(
                    websocket,
                    WSMessageType.DETECTION_FOUND,
                    detection.model_dump()
                ),
                loop
            )
        
        # Run analysis in thread pool
        result = await loop.run_in_executor(
            None,
            lambda: analyzer.analyze(
                video_path,
                progress_callback=on_progress,
                detection_callback=on_detection
            )
        )
        
        # Send final result
        await send_message(
            websocket,
            WSMessageType.ANALYSIS_COMPLETE,
            result.model_dump()
        )
        
        logger.info(f"Analysis complete: {len(result.detections)} detections")
        
    except asyncio.CancelledError:
        logger.info("Analysis task cancelled")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        await send_error(websocket, str(e))
    finally:
        if analyzer:
            analyzer.cleanup()
            analyzer = None

async def run_export(websocket, video_path: str, output_path: str, detections_data: list, 
                     anchors_data: list = None, watch_list: list = None,
                     scan_interval: int = 90, motion_threshold: float = 30.0,
                     ocr_scale: float = 1.0, scan_zones: list = None,
                     codec: str = 'h264', quality: str = 'high',
                     preview_mode: bool = False):
    """Run video export with redactions applied"""
    from analysis import export_video_optimized, preview_export_video, Detection as DetectionModel, BoundingBox
    
    try:
        loop = asyncio.get_running_loop()
        
        # Convert detection data back to Detection objects
        detections = []
        for d in detections_data:
            # Only include redacted detections
            if d.get("isRedacted", True):
                bbox_data = d.get("bbox", {})
                detection = DetectionModel(
                    id=d.get("id", ""),
                    type=d.get("type", "manual"),
                    content=d.get("content", ""),
                    confidence=d.get("confidence", 1.0),
                    bbox=BoundingBox(
                        x=bbox_data.get("x", 0),
                        y=bbox_data.get("y", 0),
                        width=bbox_data.get("width", 0.1),
                        height=bbox_data.get("height", 0.1)
                    ),
                    start_time=d.get("startTime", 0),
                    end_time=d.get("endTime", 0),
                    start_frame=d.get("frameStart", 0),
                    end_frame=d.get("frameEnd", 0),
                    track_id=d.get("trackId")
                )
                detections.append(detection)
        
        # Process anchors - convert to list of dicts
        anchors = anchors_data if anchors_data else []
        watch_items = watch_list if watch_list else []
        zones = scan_zones if scan_zones else []
        
        logger.info(f"Exporting with {len(detections)} detections, {len(anchors)} anchors, {len(watch_items)} watch items, codec={codec}, quality={quality}")
        
        def on_progress(progress: float, stage: str):
            """Send export progress (thread-safe)"""
            asyncio.run_coroutine_threadsafe(
                send_message(
                    websocket,
                    WSMessageType.EXPORT_PROGRESS,
                    {"progress": progress, "stage": stage}
                ),
                loop
            )
        
        # Run export using convenience function (preview or full)
        if preview_mode:
            result = await loop.run_in_executor(
                None,
                lambda: preview_export_video(
                    video_path=video_path,
                    output_path=output_path,
                    detections=detections,
                    anchors=anchors,
                    watch_list=watch_items,
                    scan_interval=scan_interval,
                    motion_threshold=motion_threshold,
                    ocr_scale=ocr_scale,
                    scan_zones=zones,
                    progress_callback=on_progress,
                    blur_strength=51
                )
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: export_video_optimized(
                    video_path=video_path,
                    output_path=output_path,
                    detections=detections,
                    anchors=anchors,
                    watch_list=watch_items,
                    scan_interval=scan_interval,
                    motion_threshold=motion_threshold,
                    ocr_scale=ocr_scale,
                    scan_zones=zones,
                    progress_callback=on_progress,
                    use_cache=True,
                    use_gpu=True,
                    codec=codec,
                    quality=quality
                )
            )
        
        # Extract result (now returns dict with success + detected_blurs)
        success = result.get('success', result) if isinstance(result, dict) else result
        detected_blurs = result.get('detected_blurs', []) if isinstance(result, dict) else []
        
        # Normalize bbox coordinates for frontend (convert from pixels to 0-1)
        # Get video dimensions for normalization
        import cv2 as cv_temp
        cap_temp = cv_temp.VideoCapture(video_path)
        vid_w = cap_temp.get(cv_temp.CAP_PROP_FRAME_WIDTH)
        vid_h = cap_temp.get(cv_temp.CAP_PROP_FRAME_HEIGHT)
        cap_temp.release()
        normalized_blurs = []
        for blur in detected_blurs:
            bbox = blur.get('bbox', {})
            if isinstance(bbox, dict):
                # bbox values are already normalized (0-1) from the exporter
                # No need to divide by video dimensions again!
                normalized_blurs.append({
                    'frame': blur['frame'],
                    'time': blur['time'],
                    'end_time': blur.get('end_time', blur['time'] + 1/30),
                    'bbox': bbox,  # Already normalized
                    'type': blur['type'],
                    'source': blur['source']
                })
        
        logger.info(f"Sending {len(normalized_blurs)} detected blur regions to frontend")
        
        # Send completion with detected regions
        await send_message(
            websocket,
            WSMessageType.EXPORT_COMPLETE,
            {
                "output_path": output_path, 
                "success": success,
                "detected_blurs": normalized_blurs
            }
        )
        
        logger.info(f"Export complete: {output_path}")
        
    except asyncio.CancelledError:
        logger.info("Export task cancelled")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.EXPORT_ERROR, {"error": str(e)})


async def run_scan(websocket, video_path: str, anchors: list, watch_list: list,
                   scan_interval: int, motion_threshold: float, ocr_scale: float, scan_zones: list):
    """Run OCR scan on video WITHOUT encoding - returns blur regions for overlay display."""
    try:
        from analysis import scan_video
        
        loop = asyncio.get_running_loop()
        
        def on_progress(progress: float, message: str):
            """Send scan progress (thread-safe)"""
            asyncio.run_coroutine_threadsafe(
                send_message(
                    websocket,
                    WSMessageType.EXPORT_PROGRESS,
                    {"progress": progress, "stage": message}
                ),
                loop
            )
        
        # Run scan in thread pool
        result = await loop.run_in_executor(
            None,
            lambda: scan_video(
                video_path,
                anchors=anchors,
                watch_list=watch_list,
                scan_interval=scan_interval,
                motion_threshold=motion_threshold,
                ocr_scale=ocr_scale,
                scan_zones=scan_zones,
                progress_callback=on_progress,
                use_cache=True,
                use_gpu=True
            )
        )
        
        # Extract results
        success = result.get('success', False)
        detected_blurs = result.get('detected_blurs', [])
        
        logger.info(f"Scan complete: {len(detected_blurs)} blur regions detected")
        
        # Send scan complete with detected regions
        await send_message(
            websocket,
            WSMessageType.SCAN_COMPLETE,
            {
                "success": success,
                "detected_blurs": detected_blurs
            }
        )
        
    except asyncio.CancelledError:
        logger.info("Scan task cancelled")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.PROGRESS_UPDATE, {
            "progress": 0,
            "stage": f"Scan failed: {e}",
            "error": str(e)
        })


async def run_track(websocket, video_path: str, bbox: dict, frame_number: int, detection_id: str):
    """Run content-aware tracking to find when content appears/disappears"""
    try:
        from analysis.region_tracker import track_region_in_video
        
        loop = asyncio.get_running_loop()
        
        def on_progress(progress: float, message: str):
            """Send tracking progress (thread-safe)"""
            asyncio.run_coroutine_threadsafe(
                send_message(
                    websocket,
                    WSMessageType.PROGRESS_UPDATE,
                    {"progress": progress, "stage": message, "tracking": True}
                ),
                loop
            )
        
        # Convert bbox dict to tuple
        bbox_tuple = (bbox["x"], bbox["y"], bbox["width"], bbox["height"])
        
        logger.info(f"Starting content tracking at frame {frame_number}")
        
        # Run content-aware tracking (OCR-based) in executor
        result = await loop.run_in_executor(
            None,
            lambda: track_region_in_video(
                video_path=video_path,
                bbox=bbox_tuple,
                start_frame=frame_number,
                progress_callback=on_progress
            )
        )
        
        logger.info(f"Content tracking complete: {result.start_time:.2f}s - {result.end_time:.2f}s")
        
        # Send result (no frame_positions for static tracking)
        await send_message(
            websocket,
            WSMessageType.TRACK_RESULT,
            {
                "detection_id": detection_id,
                "start_frame": result.start_frame,
                "end_frame": result.end_frame,
                "start_time": result.start_time,
                "end_time": result.end_time,
                "confidence": result.confidence,
                "frame_positions": None  # Static blur, no motion tracking
            }
        )
        
        logger.info(f"Tracking complete: {result.start_time:.2f}s - {result.end_time:.2f}s")
        
    except asyncio.CancelledError:
        logger.info("Tracking task cancelled")
    except Exception as e:
        logger.error(f"Tracking failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.ANALYSIS_ERROR, {"error": str(e)})


async def run_preview_frame(websocket, video_path: str, frame_number: int):
    """Analyze a single frame with current pii_wizard config and return blur regions"""
    try:
        from analysis.pii_wizard import analyze_frame, ProcessingConfig, Anchor
        
        global pii_config
        
        loop = asyncio.get_running_loop()
        
        # Build ProcessingConfig from current config
        watch_list = pii_config.get("watch_list", [])
        anchors_dict = pii_config.get("anchors", {})
        anchors = [
            Anchor(
                label=label,
                direction=config[0],
                gap=config[1],
                width=config[2],
                height=config[3]
            )
            for label, config in anchors_dict.items()
        ]
        
        config = ProcessingConfig(
            watch_list=watch_list,
            anchors=anchors,
            enable_keyboard=pii_config.get("enable_keyboard", False),
            ocr_scale=pii_config.get("ocr_scale", 0.5)
        )
        
        # Read the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            await send_error(websocket, f"Could not read frame {frame_number}")
            return
        
        # Analyze frame
        logger.info(f"Preview analyzing frame {frame_number}")
        rects, keyboard_detected = await loop.run_in_executor(
            None,
            lambda: analyze_frame(frame, config)
        )
        
        # Convert rects to detection format
        height, width = frame.shape[:2]
        detections = []
        for (bx, by, bw, bh) in rects:
            detections.append({
                "id": f"preview-{len(detections)}",
                "type": "manual",
                "bbox": {
                    "x": bx / width,
                    "y": by / height,
                    "width": bw / width,
                    "height": bh / height
                }
            })
        
        logger.info(f"Preview found {len(detections)} regions, keyboard={keyboard_detected}")
        
        await send_message(websocket, WSMessageType.PROGRESS_UPDATE, {
            "progress": 100,
            "stage": f"Found {len(detections)} regions",
            "detections": detections,
            "keyboard_detected": keyboard_detected
        })
        
    except Exception as e:
        logger.error(f"Preview analysis failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.ANALYSIS_ERROR, {"error": str(e)})


async def run_get_text_at_click(websocket, video_path: str, frame_number: int, click_x: float, click_y: float):
    """Get OCR text at clicked position for anchor selection"""
    try:
        from analysis.pii_wizard import get_text_at_position
        import cv2
        
        loop = asyncio.get_running_loop()
        
        # Read the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            await send_error(websocket, f"Could not read frame {frame_number}")
            return
        
        # Log frame dimensions for debugging
        h, w = frame.shape[:2]
        logger.info(f"Frame dimensions: {w}x{h}")
        logger.info(f"Click position in pixels: ({int(click_x * w)}, {int(click_y * h)})")
        
        # Get text at click position (1.0 scale for best accuracy)
        logger.info(f"Getting text at click ({click_x:.3f}, {click_y:.3f}) on frame {frame_number}")
        text_regions = await loop.run_in_executor(
            None,
            lambda: get_text_at_position(frame, click_x, click_y, ocr_scale=1.0)
        )
        
        # Return the text regions (closest first)
        logger.info(f"Found {len(text_regions)} text regions near click")
        
        # Debug: Log top 5 detected texts with bbox info
        for i, region in enumerate(text_regions[:5]):
            bbox = region['bbox']
            logger.info(f"  #{i+1}: '{region['text']}' (dist={region['distance']:.1f}, conf={region['confidence']:.2f})")
            logger.info(f"       bbox: x={bbox['x']:.3f} y={bbox['y']:.3f} w={bbox['width']:.3f} h={bbox['height']:.3f}")
        
        await send_message(websocket, WSMessageType.PROGRESS_UPDATE, {
            "progress": 100,
            "stage": f"Found {len(text_regions)} text regions",
            "text_regions": text_regions[:10]  # Return top 10 closest
        })
        
    except Exception as e:
        logger.error(f"Text detection failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.ANALYSIS_ERROR, {"error": str(e)})


async def run_get_text_in_region(websocket, video_path: str, frame_number: int, region: dict):
    """Get OCR text from a drawn region for box-based anchor selection"""
    try:
        from analysis.pii_wizard import get_text_in_region
        import cv2
        
        loop = asyncio.get_running_loop()
        
        # Read the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            await send_error(websocket, f"Could not read frame {frame_number}")
            return
        
        # Get region params
        x = region.get("x", 0)
        y = region.get("y", 0)
        width = region.get("width", 0.1)
        height = region.get("height", 0.1)
        
        logger.info(f"Getting text in region: x={x:.3f} y={y:.3f} w={width:.3f} h={height:.3f}")
        
        # OCR the region
        text = await loop.run_in_executor(
            None,
            lambda: get_text_in_region(frame, x, y, width, height)
        )
        
        logger.info(f"Detected text in region: '{text}'")
        
        await send_message(websocket, WSMessageType.PROGRESS_UPDATE, {
            "progress": 100,
            "stage": f"Detected: {text}",
            "region_text": text,
            "region": region  # Return the region back so frontend knows it
        })
        
    except Exception as e:
        logger.error(f"Region text detection failed: {e}")
        import traceback
        traceback.print_exc()
        await send_message(websocket, WSMessageType.ANALYSIS_ERROR, {"error": str(e)})


async def send_message(websocket, msg_type: WSMessageType, payload: dict):
    """Send WebSocket message to Tauri"""
    message = WSMessage(type=msg_type, payload=payload)
    await websocket.send(message.model_dump_json())


async def send_error(websocket, error: str):
    """Send error message"""
    await send_message(websocket, WSMessageType.ANALYSIS_ERROR, {"error": error})


async def websocket_handler(websocket, path, stop_event):
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            await handle_message(websocket, message)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Client disconnected, shutting down server...")
        
        # Cleanup
        global analyzer, current_task
        if current_task and not current_task.done():
            current_task.cancel()
        if analyzer:
            analyzer.cleanup()
            analyzer = None
        
        # Trigger server shutdown (prevent zombie process)
        stop_event.set()


async def main():
    """Main entry point"""
    import websockets
    
    host = "127.0.0.1"
    port = 9876
    
    logger.info(f"Starting ScreenSafe AI Sidecar on ws://{host}:{port}")
    
    # Handle shutdown gracefully
    stop = asyncio.Event()
    
    def handle_shutdown(sig):
        logger.info(f"Received signal {sig.name}, shutting down...")
        stop.set()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass
    
    async with websockets.serve(
        lambda ws, path: websocket_handler(ws, path, stop),
        host, port
    ):
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await stop.wait()
    
    logger.info("Sidecar shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
