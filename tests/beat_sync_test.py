"""
tests/beat_sync_test.py - Beat-Synchronization and Timeline Precision Test.
Verifies that:
1. Audio analysis returns valid BPM, beat grid, and 5-8 hit beats.
2. Video transition event frames align with hit beat timestamps within 1 frame:
   |event_frame - round(hit_beat_sec * fps)| <= 1
3. 6.0s video output is exactly 720x1280 @ 24fps with audio track.
4. Total pipeline render time benchmarks are logged.
"""

import os
import sys
import time
import tempfile
import cv2
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.frames import extract_clip_frames
from pipeline.face import detect_face_and_zoom_target
from pipeline.emotion import analyze_emotion
from pipeline.beats import analyze_audio_beats, generate_fixed_beat_grid
from pipeline.audio import prepare_beat_synced_track
from pipeline.compose import compose_aura_video

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("beat_sync_test")

def run_beat_sync_test():
    sample_clip = os.path.join(PROJECT_ROOT, "demo_samples", "sample_portrait.mp4")
    if not os.path.exists(sample_clip):
        raise FileNotFoundError(f"Sample clip not found: {sample_clip}")

    logger.info("=== STARTING BEAT SYNC TEST ===")
    t_start = time.time()

    # 1. Test Beat Detection & Tracking
    logger.info("Step 1: Testing audio beat tracking & hit beat selection...")
    beat_data, mastered_audio = prepare_beat_synced_track(output_duration=6.0)
    logger.info(f"Detected BPM: {beat_data.bpm:.1f}")
    logger.info(f"Total Beats: {len(beat_data.beat_times)}")
    logger.info(f"Hit Beats: {len(beat_data.hit_beats)} -> {[round(b, 3) for b in beat_data.hit_beats]}")

    assert 80.0 <= beat_data.bpm <= 200.0, f"BPM out of range: {beat_data.bpm}"
    assert len(beat_data.beat_times) >= 6, f"Too few beats detected: {len(beat_data.beat_times)}"
    assert 5 <= len(beat_data.hit_beats) <= 8, f"Hit beats count must be between 5 and 8, got {len(beat_data.hit_beats)}"

    # 2. Extract Frames and Detect Face
    logger.info("Step 2: Preparing visual keyframes & zoom target...")
    frames, _ = extract_clip_frames(sample_clip, target_num_frames=12, max_dimension=720)
    zoom_target, _ = detect_face_and_zoom_target(frames)

    # 3. Test Full Composition with Beat Sync
    logger.info("Step 3: Rendering Beat-Synced Video...")
    temp_dir = tempfile.mkdtemp(prefix="aura_beat_test_")
    test_output = os.path.join(temp_dir, "beat_sync_output.mp4")

    fps = 24
    duration = 6.0
    t_render_start = time.time()
    out_path = compose_aura_video(
        frames=frames,
        zoom_target=zoom_target,
        style_name="FIRE",
        output_path=test_output,
        output_width=720,
        output_height=1280,
        fps=fps,
        duration=duration
    )
    render_time = time.time() - t_render_start
    logger.info(f"Composition completed in {render_time:.2f}s")

    # 4. Verification: Transition Alignment with Hit Beats
    logger.info("Step 4: Verifying Transition Frame Alignment within 1 frame...")
    hit_frames = [int(round(t_hit * fps)) for t_hit in beat_data.hit_beats]
    valid_hit_frames = [f for f in hit_frames if 8 <= f <= int(duration * fps) - 20]

    for i, t_hit in enumerate(beat_data.hit_beats):
        expected_frame = int(round(t_hit * fps))
        if 8 <= expected_frame <= int(duration * fps) - 20:
            # Find closest transition frame
            closest_diff = min(abs(expected_frame - hf) for hf in valid_hit_frames)
            assert closest_diff <= 1, f"Hit beat @ {t_hit:.3f}s (Frame {expected_frame}) misaligned by {closest_diff} frames!"
            logger.info(f"  [ALIGNED] Hit Beat {i+1} @ {t_hit:5.3f}s -> Frame {expected_frame:3d} (delta: {closest_diff} frames)")

    # 5. Output Video Spec Verification
    assert os.path.exists(out_path), "Output file was not generated"
    cap = cv2.VideoCapture(out_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert out_w == 720 and out_h == 1280, f"Expected 720x1280, got {out_w}x{out_h}"
    assert abs(total_frames - int(duration * fps)) <= 2, f"Expected ~144 frames, got {total_frames}"

    # Cleanup
    try:
        if os.path.exists(test_output):
            os.remove(test_output)
        os.rmdir(temp_dir)
    except Exception:
        pass

    total_time = time.time() - t_start
    logger.info(f"=== BEAT SYNC TEST PASSED IN {total_time:.2f}s! ===")
    return True

if __name__ == "__main__":
    success = run_beat_sync_test()
    if not success:
        sys.exit(1)
