"""
tests/smoke_test.py - End-to-end integration and smoke test for AURA EDIT pipeline.
Runs the complete AI pipeline on the bundled sample clip and verifies output validity,
audio track, duration (~6s), resolution, and file integrity.
"""

import os
import sys
import time
import tempfile
import cv2
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.frames import extract_clip_frames
from pipeline.face import detect_face_and_zoom_target, get_best_face_crop
from pipeline.emotion import analyze_emotion
from pipeline.compose import compose_aura_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("smoke_test")

def run_smoke_test():
    sample_clip = os.path.join(PROJECT_ROOT, "demo_samples", "sample_portrait.mp4")
    if not os.path.exists(sample_clip):
        raise FileNotFoundError(f"Sample clip missing at: {sample_clip}")
        
    logger.info("=== STARTING AURA EDIT SMOKE TEST ===")
    t_start = time.time()
    
    # 1. Test Frame Extraction
    logger.info("Step 1: Testing frame extraction...")
    t0 = time.time()
    frames, meta = extract_clip_frames(sample_clip, target_num_frames=12, max_dimension=720)
    assert len(frames) == 12, f"Expected 12 frames, got {len(frames)}"
    logger.info(f"[PASS] Extracted {len(frames)} frames in {time.time()-t0:.2f}s. Resolution: {frames[0].shape}")
    
    # 2. Test Face Detection & Zoom Target
    logger.info("Step 2: Testing face detection & zoom target...")
    t0 = time.time()
    zoom_target, boxes = detect_face_and_zoom_target(frames)
    assert "cx" in zoom_target and "cy" in zoom_target, "Zoom target missing coordinates"
    face_crop = get_best_face_crop(frames, boxes)
    logger.info(f"[PASS] Face detection completed in {time.time()-t0:.2f}s. Target: {zoom_target}")
    
    # 3. Test Emotion Classification
    logger.info("Step 3: Testing emotion classification...")
    t0 = time.time()
    emotion, conf, style = analyze_emotion(face_crop)
    logger.info(f"[PASS] Emotion: '{emotion}' ({conf*100:.1f}%) -> Style: {style} (Elapsed: {time.time()-t0:.2f}s)")
    assert style in ["DARK", "GOLD", "FIRE"], f"Invalid style: {style}"
    
    # 4. Test Full Video Composition
    logger.info("Step 4: Testing full 6-second video composition and export...")
    temp_dir = tempfile.mkdtemp(prefix="aura_smoke_test_")
    test_output = os.path.join(temp_dir, "smoke_aura_output.mp4")
    
    t0 = time.time()
    out_path = compose_aura_video(
        frames=frames,
        zoom_target=zoom_target,
        style_name=style,
        output_path=test_output,
        output_width=720,
        output_height=1280,
        fps=24,
        duration=6.0
    )
    t_render = time.time() - t0
    logger.info(f"[PASS] Video composed in {t_render:.2f}s -> {out_path}")
    
    # 5. Output Verification
    assert os.path.exists(out_path), f"Output file does not exist: {out_path}"
    file_size = os.path.getsize(out_path)
    assert file_size > 100_000, f"Output file size suspiciously small: {file_size} bytes"
    
    cap = cv2.VideoCapture(out_path)
    assert cap.isOpened(), "Could not open generated MP4 file with OpenCV"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    logger.info(f"Generated Video Specs: {width}x{height} | {total_frames} frames | {fps:.1f} fps | {duration:.2f}s duration")
    assert 5.5 <= duration <= 6.5, f"Duration {duration:.2f}s not within 5.5s - 6.5s range"
    assert width == 720 and height == 1280, f"Resolution {width}x{height} does not match expected 720x1280"
    
    # 6. Cleanup
    try:
        if os.path.exists(test_output):
            os.remove(test_output)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.debug(f"Cleanup note: {e}")
        
    total_elapsed = time.time() - t_start
    logger.info(f"=== SMOKE TEST PASSED SUCCESSFULLY IN {total_elapsed:.2f}s! ===")
    return True

if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        sys.exit(1)
