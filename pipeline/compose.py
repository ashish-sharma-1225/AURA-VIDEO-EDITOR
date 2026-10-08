"""
pipeline/compose.py - Beat-Synced Video Composition Engine for AURA EDIT.
Composes 720x1280 (9:16 vertical HD) 6.0s edits synchronized to music beats:
- Multi-shot timeline cut on hit beats (Shot A, B, C, D, E)
- 6 dynamic transitions on hit beats: White Flash, Zoom Punch, RGB Split, Glitch Slice, Whip Pan, Shake
- 3% breathing pulse on every beat
- Style-specific rendering: DARK (red chromatic, vignette), GOLD (gold flash, bloom), FIRE (shake, contrast, glitch)
- Fast vectorized NumPy/OpenCV rendering (< 25s render time)
- FFmpeg H.264 + AAC muxing with exact 6.0s duration and zero audio-video offset
"""

import os
import time
import math
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import logging

import imageio_ffmpeg

from .styles import get_style_config
from .background import (
    extract_foreground_matte,
    generate_dynamic_background,
    composite_foreground_with_glow
)
from .audio import prepare_beat_synced_track, get_or_synthesize_audio
from .beats import BeatSyncData

logger = logging.getLogger("aura_edit.compose")

# ---------------------------------------------------------------------------
# Typography Helpers
# ---------------------------------------------------------------------------
def _get_font(font_size: int, is_title: bool = True) -> ImageFont.ImageFont:
    """Loads system bold fonts or falls back to Pillow default."""
    candidates = []
    if is_title:
        candidates = [
            "C:/Windows/Fonts/impact.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/impact.ttf",
        ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, font_size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_styled_text_overlay(
    frame_rgb: np.ndarray,
    title_text: str,
    tagline_text: str,
    style_config: Dict[str, Any],
    alpha_fade: float = 1.0,
    scale_pop: float = 1.0
) -> np.ndarray:
    """Renders bold phonk typography with outer glow and shadow."""
    if alpha_fade <= 0.01:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    base_title_size = int(w * 0.125 * scale_pop)
    tagline_size = int(w * 0.045)

    title_font = _get_font(base_title_size, is_title=True)
    tag_font = _get_font(tagline_size, is_title=False)

    glow_col = tuple(style_config.get("glow_color", (255, 255, 255)))
    text_fill = tuple(style_config.get("text_color", (255, 255, 255)))
    outline_col = tuple(style_config.get("text_outline", (0, 0, 0)))

    bbox_title = draw.textbbox((0, 0), title_text, font=title_font)
    t_w = bbox_title[2] - bbox_title[0]
    t_h = bbox_title[3] - bbox_title[1]
    title_x = (w - t_w) // 2
    title_y = int(h * 0.72) - (t_h // 2)

    # Outer glow
    glow_alpha = int(90 * alpha_fade)
    for radius in [12, 8, 4]:
        draw.text(
            (title_x, title_y),
            title_text,
            font=title_font,
            fill=(*glow_col, glow_alpha),
            stroke_width=radius,
            stroke_fill=(*glow_col, glow_alpha)
        )

    # Black outline & fill
    outline_alpha = int(240 * alpha_fade)
    draw.text(
        (title_x, title_y),
        title_text,
        font=title_font,
        fill=(*text_fill, int(255 * alpha_fade)),
        stroke_width=4,
        stroke_fill=(*outline_col, outline_alpha)
    )

    # Subtitle Tagline
    bbox_tag = draw.textbbox((0, 0), tagline_text, font=tag_font)
    tag_w = bbox_tag[2] - bbox_tag[0]
    tag_x = (w - tag_w) // 2
    tag_y = title_y + t_h + 16

    draw.text(
        (tag_x, tag_y),
        tagline_text,
        font=tag_font,
        fill=(255, 255, 255, int(180 * alpha_fade))
    )

    base_img = Image.fromarray(frame_rgb).convert("RGBA")
    combined = Image.alpha_composite(base_img, overlay)
    return np.array(combined.convert("RGB"))

# ---------------------------------------------------------------------------
# Fast Transformation & VFX Primitives
# ---------------------------------------------------------------------------
def apply_zoom_and_crop(
    frame_rgb: np.ndarray,
    zoom_scale: float,
    center_norm: Tuple[float, float]
) -> np.ndarray:
    """Applies a high-quality centered zoom crop."""
    if abs(zoom_scale - 1.0) < 0.005:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    cx = int(w * center_norm[0])
    cy = int(h * center_norm[1])

    crop_w = max(10, int(w / zoom_scale))
    crop_h = max(10, int(h / zoom_scale))

    x1 = np.clip(cx - crop_w // 2, 0, w - crop_w)
    y1 = np.clip(cy - crop_h // 2, 0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def apply_rgb_split(frame_rgb: np.ndarray, offset: int, red_only: bool = False) -> np.ndarray:
    """Horizontally offsets color channels for chromatic aberration."""
    if offset <= 0:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    out = frame_rgb.copy()

    # Red shifted right
    out[:, offset:, 0] = frame_rgb[:, :-offset, 0]
    if not red_only:
        # Blue shifted left
        out[:, :-offset, 2] = frame_rgb[:, offset:, 2]
    return out

def apply_glitch_slice(frame_rgb: np.ndarray, intensity: float, seed: int = 42) -> np.ndarray:
    """Slices frame into horizontal strips and offsets them randomly."""
    if intensity <= 0.01:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    out = frame_rgb.copy()
    rng = np.random.RandomState(seed)

    num_slices = 16
    slice_h = h // num_slices
    max_shift = int(28 * intensity)

    # Shift 4 random slices
    slice_indices = rng.choice(num_slices, size=4, replace=False)
    for idx in slice_indices:
        y1 = idx * slice_h
        y2 = min(h, y1 + slice_h)
        shift = rng.randint(-max_shift, max_shift + 1)
        if shift != 0:
            out[y1:y2] = np.roll(out[y1:y2], shift, axis=1)
            # Slight color tint on slice
            out[y1:y2, :, (idx % 3)] = np.clip(out[y1:y2, :, (idx % 3)] * 1.25, 0, 255)
    return out

def apply_whip_pan(
    current_frame: np.ndarray,
    prev_frame: np.ndarray,
    progress: float
) -> np.ndarray:
    """Smooth directional horizontal whip pan transition between two frames."""
    h, w = current_frame.shape[:2]
    # Ease cubic
    ease = 3.0 * (progress ** 2) - 2.0 * (progress ** 3)
    dx = int(w * (1.0 - ease))

    composed = np.zeros_like(current_frame)
    if dx > 0 and dx < w:
        composed[:, :w - dx] = current_frame[:, dx:]
        composed[:, w - dx:] = prev_frame[:, :dx]
    else:
        composed = current_frame.copy()

    # Apply directional motion blur
    blur_k = max(1, int(21 * math.sin(progress * math.pi)))
    if blur_k % 2 == 0:
        blur_k += 1
    if blur_k > 1:
        composed = cv2.blur(composed, (blur_k, 1))
    return composed

def apply_camera_shake(frame_rgb: np.ndarray, max_amp: int, seed: int) -> np.ndarray:
    """Applies decaying camera shake offset with reflected border."""
    if max_amp <= 0:
        return frame_rgb
    h, w = frame_rgb.shape[:2]
    rng = np.random.RandomState(seed)
    dx = rng.randint(-max_amp, max_amp + 1)
    dy = rng.randint(-max_amp, max_amp + 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_vignette(frame_rgb: np.ndarray, strength: float = 0.55) -> np.ndarray:
    """Applies dark aesthetic cinematic vignette."""
    h, w = frame_rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_d = math.hypot(cx, cy)

    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / max_d
    vig_mask = np.clip(1.0 - (dist ** 1.8) * strength, 0.0, 1.0)[:, :, None]
    return (frame_rgb.astype(np.float32) * vig_mask).astype(np.uint8)

# ---------------------------------------------------------------------------
# Shot Rendering Engine
# ---------------------------------------------------------------------------
def render_base_shot(
    shot_type: str,
    fitted_frames: List[np.ndarray],
    cutout_frames: List[np.ndarray],
    zoom_target: Dict[str, Any],
    style_config: Dict[str, Any],
    p: float,
    seed: int = 42
) -> np.ndarray:
    """
    Renders one of the 5 distinct camera angles:
    - Shot A: slow push-in on the face
    - Shot B: tight punch-in crop on eyes/face (zoom 1.6x to 2x)
    - Shot C: wide shot with the cutout person over generated background, slight rotation
    - Shot D: mirrored or reversed segment with a color shift
    - Shot E: slow-motion segment (speed 0.5x using frame interpolation)
    """
    num_frames = len(fitted_frames)
    cx = zoom_target.get("cx", 0.5)
    cy = zoom_target.get("cy", 0.45)
    style_name = style_config.get("name", "DARK")

    if shot_type == "SHOT_A":
        # Slow push-in on face (1.05x to 1.28x)
        idx = int(p * (num_frames - 1))
        base = fitted_frames[idx]
        zoom = 1.05 + 0.23 * (p ** 1.2)
        return apply_zoom_and_crop(base, zoom, (cx, cy))

    elif shot_type == "SHOT_B":
        # Tight punch-in crop on eyes/face (1.65x to 1.95x)
        idx = int(p * (num_frames - 1))
        base = fitted_frames[idx]
        cy_eyes = max(0.18, cy - 0.05)
        zoom = 1.70 + 0.20 * math.sin(p * math.pi)
        return apply_zoom_and_crop(base, zoom, (cx, cy_eyes))

    elif shot_type == "SHOT_C":
        # Wide shot with cutout person over procedural background, slight rotation
        cut_idx = min(len(cutout_frames) - 1, int(p * len(cutout_frames)))
        comp = cutout_frames[cut_idx].copy()
        # Slight dynamic rotation (+- 3 deg)
        rot_angle = 3.2 * math.sin(p * math.pi * 2.0)
        h, w = comp.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_angle, 1.02)
        return cv2.warpAffine(comp, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    elif shot_type == "SHOT_D":
        # Mirrored / reversed with color shift
        rev_idx = int((1.0 - p) * (num_frames - 1))
        base = cv2.flip(fitted_frames[rev_idx], 1) # Horizontal mirror
        out = base.copy()
        if style_name == "DARK":
            # Red tint boost + slight desaturation
            out[:, :, 0] = np.clip(out[:, :, 0] * 1.28, 0, 255) # Red boost
            out[:, :, 1] = np.clip(out[:, :, 1] * 0.88, 0, 255)
        elif style_name == "GOLD":
            # Golden amber tint
            out[:, :, 0] = np.clip(out[:, :, 0] * 1.15, 0, 255)
            out[:, :, 1] = np.clip(out[:, :, 1] * 1.10, 0, 255)
            out[:, :, 2] = np.clip(out[:, :, 2] * 0.80, 0, 255)
        else: # FIRE
            # High-energy contrast boost
            out = cv2.convertScaleAbs(out, alpha=1.22, beta=-10)
        return apply_zoom_and_crop(out, 1.12, (1.0 - cx, cy))

    elif shot_type == "SHOT_E":
        # Slow-motion segment (speed 0.5x by frame duplication / half-speed ramp)
        slow_idx = int(0.5 * p * (num_frames - 1))
        base = fitted_frames[slow_idx]
        return apply_zoom_and_crop(base, 1.14, (cx, cy))

    else:
        # Default fallback
        idx = int(p * (num_frames - 1))
        return fitted_frames[idx]

# ---------------------------------------------------------------------------
# Precomputing Fast Cutouts
# ---------------------------------------------------------------------------
def precompute_cutouts_fast(
    fitted_frames: List[np.ndarray],
    style_config: Dict[str, Any],
    target_count: int = 4
) -> List[np.ndarray]:
    """
    Extracts high-fidelity background cutouts for keyframes rapidly.
    To ensure render time stays strictly under 25s on CPU, processes a small
    representative set of keyframes and upscales the clean neural alpha mask.
    """
    logger.info(f"Precomputing {target_count} cutout keyframes with glow composite...")
    t0 = time.time()
    h, w = fitted_frames[0].shape[:2]
    num_src = len(fitted_frames)

    # Sample keyframe indices
    indices = np.round(np.linspace(0, num_src - 1, target_count)).astype(int)
    cutout_results = []

    for k_idx, src_i in enumerate(indices):
        frame = fitted_frames[src_i]

        # Downsample slightly for 3x faster U2Net inference on CPU
        small_w = w // 2
        small_h = h // 2
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Extract matte
        rgba_small = extract_foreground_matte(small_frame)
        alpha_small = rgba_small[:, :, 3]
        alpha_full = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
        fg_rgba = np.dstack([frame, alpha_full])

        # Generate procedural background for this timestamp progress
        t_prog = k_idx / max(1, target_count - 1)
        bg_rgb = generate_dynamic_background(w, h, style_config, t_prog)

        # Composite with neon aura glow
        comp = composite_foreground_with_glow(fg_rgba, bg_rgb, style_config)
        cutout_results.append(comp)

    logger.info(f"Precomputed {len(cutout_results)} cutouts in {time.time() - t0:.2f}s")
    return cutout_results

# ---------------------------------------------------------------------------
# Main Video Composition Engine
# ---------------------------------------------------------------------------
def compose_aura_video(
    frames: List[np.ndarray],
    zoom_target: Dict[str, Any],
    style_name: str,
    output_path: str,
    output_width: int = 720,
    output_height: int = 1280,
    fps: int = 24,
    duration: float = 6.0,
    audio_path: Optional[str] = None
) -> str:
    """
    Beat-Synced Video Composition Pipeline.
    
    1. Prepares 6.0s beat-synced audio with ducking and hit-beat SFX.
    2. Builds an edit timeline cut at hit beats cycling through Shot A-E.
    3. Fires dynamic transitions on every hit beat and 3% breathing pulses on all beats.
    4. Renders all 144 frames in vectorized NumPy/OpenCV.
    5. Exports via FFmpeg rawvideo pipe for lightning-fast H.264/AAC muxing (< 25s total).
    """
    t_start_total = time.time()
    style_config = get_style_config(style_name)
    total_output_frames = int(duration * fps) # 144 frames

    logger.info(f"Starting Beat-Synced Composition: Style={style_config['name']}, Canvas={output_width}x{output_height} @ {fps}fps")

    # -----------------------------------------------------------------------
    # Stage 1: Audio Analysis & Beat Synchronization
    # -----------------------------------------------------------------------
    t0 = time.time()
    beat_data, mastered_audio_path = prepare_beat_synced_track(
        output_duration=duration
    )
    t_stage1 = time.time() - t0
    logger.info(f"[Stage 1/4 Completed] Beat Detection & SFX Mix in {t_stage1:.2f}s (BPM={beat_data.bpm:.1f}, Hits={len(beat_data.hit_beats)})")

    # -----------------------------------------------------------------------
    # Stage 2: Fit Input Frames & Precompute Cutouts
    # -----------------------------------------------------------------------
    t0 = time.time()
    fitted_frames = []
    for frame in frames:
        f_h, f_w = frame.shape[:2]
        scale = max(output_width / f_w, output_height / f_h)
        new_w = int(f_w * scale)
        new_h = int(f_h * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_start = (new_w - output_width) // 2
        y_start = (new_h - output_height) // 2
        fitted = scaled_frame[y_start:y_start + output_height, x_start:x_start + output_width]
        fitted_frames.append(fitted)

    cutout_frames = precompute_cutouts_fast(
        fitted_frames,
        style_config,
        target_count=4
    )
    t_stage2 = time.time() - t0
    logger.info(f"[Stage 2/4 Completed] Cutouts ready in {t_stage2:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 3: Build Edit Schedule & Precompute Timeline
    # -----------------------------------------------------------------------
    hit_frames = [int(round(t_hit * fps)) for t_hit in beat_data.hit_beats]
    # Filter valid hit frames (ensuring margin at start and end)
    hit_frames = [f for f in hit_frames if 8 <= f <= total_output_frames - 20]
    hit_frames = sorted(list(set(hit_frames)))

    # Beat pulse frames (all beats)
    all_beat_frames = [int(round(b * fps)) for b in beat_data.beat_times if 0 <= b < duration]

    # Partition into segments at hit beats
    # Segment boundaries: [0, h_1, h_2, ..., h_k, total_frames]
    boundaries = [0] + hit_frames + [total_output_frames]
    available_shots = ["SHOT_A", "SHOT_C", "SHOT_B", "SHOT_D", "SHOT_E"]

    segments = []
    last_shot = None
    for seg_idx in range(len(boundaries) - 1):
        s_start = boundaries[seg_idx]
        s_end = boundaries[seg_idx + 1]

        if seg_idx == len(boundaries) - 2:
            # Final segment is dramatic freeze frame with style text
            shot_type = "SHOT_FREEZE"
        else:
            # Cycle through shots without repeating consecutive shots
            candidates = [s for s in available_shots if s != last_shot]
            shot_type = candidates[seg_idx % len(candidates)]
            last_shot = shot_type

        segments.append({
            "start": s_start,
            "end": s_end,
            "shot": shot_type
        })

    # Schedule Transitions at each hit beat
    TRANSITIONS_LIST = ["white_flash", "zoom_punch", "rgb_split", "glitch_slice", "whip_pan", "shake"]
    if style_name == "DARK":
        # DARK has fewer flashes, favoring whip pan & rgb split
        TRANSITIONS_LIST = ["whip_pan", "rgb_split", "zoom_punch", "shake", "glitch_slice", "white_flash"]

    transition_schedule = []
    last_trans = None
    for idx, h_frame in enumerate(hit_frames):
        candidates = [tr for tr in TRANSITIONS_LIST if tr != last_trans]
        trans_type = candidates[idx % len(candidates)]
        last_trans = trans_type

        # Durations per transition
        dur_map = {
            "white_flash": 3,
            "zoom_punch": 8,
            "rgb_split": 6 if style_name != "DARK" else 8,
            "glitch_slice": 4,
            "whip_pan": 6,
            "shake": 6
        }
        transition_schedule.append({
            "frame": h_frame,
            "time": h_frame / fps,
            "type": trans_type,
            "duration": dur_map.get(trans_type, 6)
        })

    logger.info(f"Edit Schedule: {len(segments)} Segments, {len(transition_schedule)} Transitions")
    for s_i, seg in enumerate(segments):
        logger.info(f"  Segment {s_i}: Frames [{seg['start']:3d}-{seg['end']:3d}] ({seg['start']/fps:.2f}s - {seg['end']/fps:.2f}s) -> {seg['shot']}")
    for tr in transition_schedule:
        logger.info(f"  Transition @ Frame {tr['frame']:3d} ({tr['time']:.2f}s): {tr['type']} ({tr['duration']} frames)")

    # -----------------------------------------------------------------------
    # Stage 4: Frame-by-Frame Timeline Rendering
    # -----------------------------------------------------------------------
    t0 = time.time()
    rendered_frames = []
    flash_color = np.array(style_config.get("flash_color", (255, 255, 255)), dtype=np.float32)
    flash_canvas = np.full((output_height, output_width, 3), flash_color, dtype=np.uint8)

    cx = zoom_target.get("cx", 0.5)
    cy = zoom_target.get("cy", 0.45)
    rng_global = np.random.RandomState(42)

    current_seg_idx = 0
    prev_rendered = None

    for f in range(total_output_frames):
        t_sec = f / fps

        # Find active segment
        while current_seg_idx < len(segments) - 1 and f >= segments[current_seg_idx]["end"]:
            current_seg_idx += 1
        active_seg = segments[current_seg_idx]
        seg_start = active_seg["start"]
        seg_end = active_seg["end"]
        seg_dur = max(1, seg_end - seg_start)
        p_seg = (f - seg_start) / float(seg_dur)

        # 1. Render Base Shot
        shot_type = active_seg["shot"]
        if shot_type == "SHOT_FREEZE":
            # Hold last cutout frame at dynamic zoom 1.32x
            base = cutout_frames[-1]
            frame_rgb = apply_zoom_and_crop(base, 1.32, (cx, cy))
        else:
            frame_rgb = render_base_shot(
                shot_type=shot_type,
                fitted_frames=fitted_frames,
                cutout_frames=cutout_frames,
                zoom_target=zoom_target,
                style_config=style_config,
                p=p_seg,
                seed=42 + f
            )

        # 2. Check and Apply Active Hit Beat Transition
        for tr in transition_schedule:
            tr_start = tr["frame"]
            tr_dur = tr["duration"]
            if tr_start <= f < tr_start + tr_dur:
                k = f - tr_start
                tr_name = tr["type"]

                if tr_name == "white_flash":
                    decay = (1.0 - k / float(tr_dur)) * 0.88
                    frame_rgb = cv2.addWeighted(frame_rgb, 1.0 - decay, flash_canvas, decay, 0)

                elif tr_name == "zoom_punch":
                    punch_scale = 1.0 + 0.16 * ((1.0 - k / float(tr_dur)) ** 2.0)
                    frame_rgb = apply_zoom_and_crop(frame_rgb, punch_scale, (cx, cy))

                elif tr_name == "rgb_split":
                    split_px = int(22.0 * (1.0 - k / float(tr_dur)))
                    red_only = (style_name == "DARK")
                    frame_rgb = apply_rgb_split(frame_rgb, split_px, red_only=red_only)

                elif tr_name == "glitch_slice":
                    g_int = 1.0 - k / float(tr_dur)
                    if style_name == "FIRE":
                        g_int *= 1.4
                    frame_rgb = apply_glitch_slice(frame_rgb, g_int, seed=f * 17)

                elif tr_name == "whip_pan":
                    w_prog = k / float(tr_dur)
                    prev_f = prev_rendered if prev_rendered is not None else frame_rgb
                    frame_rgb = apply_whip_pan(frame_rgb, prev_f, w_prog)

                elif tr_name == "shake":
                    env = 1.0 - k / float(tr_dur)
                    amp_factor = 0.045 if style_name == "FIRE" else 0.030
                    shake_amp = int(output_width * amp_factor * env)
                    frame_rgb = apply_camera_shake(frame_rgb, shake_amp, seed=f * 31)

                # Contrast spike on FIRE on frame 0 of hit beat
                if style_name == "FIRE" and k == 0:
                    frame_rgb = cv2.convertScaleAbs(frame_rgb, alpha=1.25, beta=10)
                break

        # 3. Apply Beat Breathing Pulse on EVERY beat (decays over 0.15s ~ 4 frames)
        for bf in all_beat_frames:
            if 0 <= f - bf < 4:
                b_offset = f - bf
                pulse_decay = 1.0 - (b_offset / 4.0)
                # 3% zoom pulse
                z_pulse = 1.0 + 0.03 * pulse_decay
                frame_rgb = apply_zoom_and_crop(frame_rgb, z_pulse, (cx, cy))
                # Subtle brightness boost
                bright_val = int(8 * pulse_decay)
                frame_rgb = cv2.add(frame_rgb, np.full_like(frame_rgb, bright_val))
                # GOLD bloom on beat pulse
                if style_name == "GOLD" and pulse_decay > 0.5:
                    bloom_layer = cv2.GaussianBlur(frame_rgb, (21, 21), 10)
                    frame_rgb = cv2.addWeighted(frame_rgb, 0.85, bloom_layer, 0.15, 0)
                break

        # 4. Style-Specific Post-Processing
        if style_name == "DARK":
            # Heavy cinematic vignette
            frame_rgb = apply_vignette(frame_rgb, strength=0.55)

        # 5. Phonk Text Overlay on Freeze Segment
        if shot_type == "SHOT_FREEZE":
            freeze_prog = p_seg
            alpha_text = min(1.0, freeze_prog * 2.5) # Fast snap fade-in
            # Scale pop bounce
            scale_pop = 1.25 - 0.25 * min(1.0, freeze_prog * 3.0) + 0.02 * math.sin(freeze_prog * 12.0)
            frame_rgb = draw_styled_text_overlay(
                frame_rgb,
                title_text=style_config["title"],
                tagline_text=style_config["tagline"],
                style_config=style_config,
                alpha_fade=alpha_text,
                scale_pop=scale_pop
            )

        prev_rendered = frame_rgb.copy()
        rendered_frames.append(frame_rgb)

    t_stage3 = time.time() - t0
    logger.info(f"[Stage 3/4 Completed] Rendered {len(rendered_frames)} VFX frames in {t_stage3:.2f}s")

    # -----------------------------------------------------------------------
    # Stage 4: FFmpeg Fast Video Encoding & Audio Muxing
    # -----------------------------------------------------------------------
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    logger.info(f"Exporting video via FFmpeg pipe ({output_width}x{output_height} @ {fps}fps, {len(rendered_frames)} frames)...")

    cmd = [
        ffmpeg_exe, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{output_width}x{output_height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-i", mastered_audio_path,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", f"{duration:.2f}",
        "-movflags", "+faststart",
        output_path
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in rendered_frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    stdout_data, stderr_data = proc.communicate()

    if proc.returncode != 0:
        logger.error(f"FFmpeg encoding failed with code {proc.returncode}: {stderr_data.decode(errors='replace')}")
        # Fallback to MoviePy if pipe fails
        from moviepy.editor import ImageSequenceClip, AudioFileClip
        v_clip = ImageSequenceClip(rendered_frames, fps=fps)
        a_clip = AudioFileClip(mastered_audio_path).subclip(0, duration)
        out_clip = v_clip.set_audio(a_clip)
        out_clip.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
        v_clip.close()
        a_clip.close()
        out_clip.close()

    t_stage4 = time.time() - t0
    total_time = time.time() - t_start_total

    logger.info(f"[Stage 4/4 Completed] FFmpeg Export in {t_stage4:.2f}s -> {output_path}")
    logger.info(f"=== FULL BEAT-SYNC PIPELINE FINISHED IN {total_time:.2f}s! ===")

    # Timing Summary Printout
    print("\n" + "=" * 60)
    print(f">> AURA EDIT BEAT-SYNC BENCHMARK ({style_name}):")
    print(f"   BPM:                               {beat_data.bpm:6.1f}")
    print(f"   Total Beats Detected:              {len(beat_data.beat_times):6d}")
    print(f"   Hit Beats Count:                   {len(beat_data.hit_beats):6d}")
    print(f"   Stage 1 (Beat Detection & SFX):    {t_stage1:6.2f}s")
    print(f"   Stage 2 (Cutouts & Compositing):   {t_stage2:6.2f}s")
    print(f"   Stage 3 (Timeline VFX Assembly):   {t_stage3:6.2f}s")
    print(f"   Stage 4 (FFmpeg H.264/AAC Export): {t_stage4:6.2f}s")
    print(f"  -------------------------------------------------")
    print(f"  ** TOTAL PIPELINE RENDER TIME:      {total_time:6.2f}s (Target: < 25s)")
    print("=" * 60)
    print(">> TRANSITION SCHEDULE:")
    for tr in transition_schedule:
        print(f"   Frame {tr['frame']:3d} | Time {tr['time']:5.2f}s | Type: {tr['type']:<13} | Duration: {tr['duration']} frames")
    print("=" * 60 + "\n")

    return output_path
