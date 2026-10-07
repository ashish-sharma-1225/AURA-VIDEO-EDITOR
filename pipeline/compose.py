"""
pipeline/compose.py - Video composition engine for 6-second vertical Aura Edits.
Composes 720x1280 (9:16 vertical HD) video with:
- Slow dynamic zoom focused on detected face box
- Phonk beat-synced cuts and 3-4 white flash frames
- Procedural anime/phonk backgrounds & neon aura edge glow
- Dual-layer glowing typography with fade-in and scale pop
- Final dramatic freeze frame
- Fast H.264 + AAC MP4 export via MoviePy
"""

import os
import time
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional
import logging

from .styles import get_style_config
from .background import (
    extract_foreground_matte,
    generate_dynamic_background,
    composite_foreground_with_glow
)
from .audio import get_or_synthesize_audio

logger = logging.getLogger("aura_edit.compose")

# Fonts
def _get_font(font_size: int, is_title: bool = True) -> ImageFont.ImageFont:
    """Loads system Impact/Arial Bold font, or falls back to Pillow default."""
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
    """
    Renders big bold phonk-style typography over the frame:
    - Glowing outer stroke
    - Inner bright fill
    - Drop shadow
    - Subtitle badge / tagline
    """
    if alpha_fade <= 0.01:
        return frame_rgb
        
    h, w = frame_rgb.shape[:2]
    # Work on a RGBA overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Calculate font sizes
    base_title_size = int(w * 0.125 * scale_pop)
    tagline_size = int(w * 0.045)
    
    title_font = _get_font(base_title_size, is_title=True)
    tag_font = _get_font(tagline_size, is_title=False)
    
    # Text colors
    glow_col = tuple(style_config.get("glow_color", (255, 255, 255)))
    text_fill = tuple(style_config.get("text_color", (255, 255, 255)))
    outline_col = tuple(style_config.get("text_outline", (0, 0, 0)))
    
    # 1. Main Title Box & Positioning
    # Placed in the lower third (y ~ 0.72)
    bbox_title = draw.textbbox((0, 0), title_text, font=title_font)
    t_w = bbox_title[2] - bbox_title[0]
    t_h = bbox_title[3] - bbox_title[1]
    title_x = (w - t_w) // 2
    title_y = int(h * 0.72) - (t_h // 2)
    
    # 2. Tagline Box
    bbox_tag = draw.textbbox((0, 0), tagline_text, font=tag_font)
    tag_w = bbox_tag[2] - bbox_tag[0]
    tag_h = bbox_tag[3] - bbox_tag[1]
    tag_x = (w - tag_w) // 2
    tag_y = title_y + t_h + int(h * 0.018)
    
    # Draw dark shadow
    shadow_offset = int(w * 0.01)
    shadow_col = (0, 0, 0, int(220 * alpha_fade))
    draw.text((title_x + shadow_offset, title_y + shadow_offset), title_text, font=title_font, fill=shadow_col)
    
    # Draw thick glowing outline for title
    outline_stroke = max(2, int(w * 0.018))
    draw.text(
        (title_x, title_y),
        title_text,
        font=title_font,
        fill=(*text_fill, int(255 * alpha_fade)),
        stroke_width=outline_stroke,
        stroke_fill=(*outline_col, int(255 * alpha_fade))
    )
    
    # Draw tagline with pill / badge background
    pad_x, pad_y = int(w * 0.035), int(h * 0.007)
    pill_box = [tag_x - pad_x, tag_y - pad_y, tag_x + tag_w + pad_x, tag_y + tag_h + pad_y]
    draw.rounded_rectangle(
        pill_box,
        radius=int(h * 0.01),
        fill=(10, 10, 15, int(210 * alpha_fade)),
        outline=(*glow_col, int(230 * alpha_fade)),
        width=2
    )
    draw.text(
        (tag_x, tag_y),
        tagline_text,
        font=tag_font,
        fill=(*glow_col, int(255 * alpha_fade))
    )
    
    # Top header badge ("AURA EDIT // AI POWERED")
    header_font = _get_font(int(w * 0.032), is_title=False)
    header_text = "✦ AURA EDIT // PROCESSED ✦"
    h_bbox = draw.textbbox((0, 0), header_text, font=header_font)
    h_w = h_bbox[2] - h_bbox[0]
    draw.text(
        ((w - h_w) // 2, int(h * 0.06)),
        header_text,
        font=header_font,
        fill=(255, 255, 255, int(180 * alpha_fade))
    )

    # Composite overlay onto frame
    base_img = Image.fromarray(frame_rgb).convert("RGBA")
    combined = Image.alpha_composite(base_img, overlay)
    return np.array(combined.convert("RGB"))

def apply_zoom_and_crop(
    frame_rgb: np.ndarray,
    zoom_scale: float,
    center_norm: tuple
) -> np.ndarray:
    """
    Applies a smooth digital zoom centered on (center_x, center_y).
    """
    if abs(zoom_scale - 1.0) < 0.005:
        return frame_rgb
        
    h, w = frame_rgb.shape[:2]
    cx = int(w * center_norm[0])
    cy = int(h * center_norm[1])
    
    # Crop dimensions at zoom level
    crop_w = int(w / zoom_scale)
    crop_h = int(h / zoom_scale)
    
    # Calculate top-left corner
    x1 = np.clip(cx - crop_w // 2, 0, w - crop_w)
    y1 = np.clip(cy - crop_h // 2, 0, h - crop_h)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    cropped = frame_rgb[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

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
    Main video rendering pipeline.
    
    Args:
        frames: ~12 extracted RGB frames from input clip
        zoom_target: detected face target center and dimensions
        style_name: "DARK", "GOLD", or "FIRE"
        output_path: Destination path for .mp4 video
        output_width: 720 (default vertical HD)
        output_height: 1280
        fps: 24 fps
        duration: 6.0 seconds
        audio_path: Optional path to custom audio track
        
    Returns:
        Absolute path to exported MP4 file.
    """
    t_start_total = time.time()
    style_config = get_style_config(style_name)
    total_output_frames = int(duration * fps) # 144 frames
    
    logger.info(f"Starting video composition: Style={style_config['name']}, Target={output_width}x{output_height} @ {fps}fps")
    
    # Stage 1: Matte & Composite ~12 Keyframes (Background Removal)
    t0 = time.time()
    logger.info(f"[Stage 1/4] Extracting matting for {len(frames)} frames with u2net...")
    composited_keyframes = []
    
    for i, frame in enumerate(frames):
        # Resize frame to fill vertical output canvas proportionally
        f_h, f_w = frame.shape[:2]
        scale = max(output_width / f_w, output_height / f_h)
        new_w = int(f_w * scale)
        new_h = int(f_h * scale)
        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center-crop to exact output dimensions (720x1280)
        x_start = (new_w - output_width) // 2
        y_start = (new_h - output_height) // 2
        fitted_frame = scaled_frame[y_start:y_start+output_height, x_start:x_start+output_width]
        
        # Remove background to get RGBA cutout
        fg_rgba = extract_foreground_matte(fitted_frame)
        
        # Generate initial styled background at keyframe progress
        t_key = i / max(1, len(frames) - 1)
        bg_rgb = generate_dynamic_background(output_width, output_height, style_config, t_key)
        
        # Composite subject with neon edge glow
        comp = composite_foreground_with_glow(fg_rgba, bg_rgb, style_config)
        composited_keyframes.append(comp)
        
    t_stage1 = time.time() - t0
    logger.info(f"[Stage 1/4 Completed] Matting & compositing {len(frames)} frames took: {t_stage1:.2f}s")
    
    # Stage 2: Audio Preparation
    t0 = time.time()
    logger.info("[Stage 2/4] Preparing 6.0s phonk audio track...")
    if audio_path is None or not os.path.exists(audio_path):
        audio_path = get_or_synthesize_audio(output_duration=duration)
    t_stage2 = time.time() - t0
    logger.info(f"[Stage 2/4 Completed] Audio track ready in: {t_stage2:.2f}s ({audio_path})")
    
    # Stage 3: Frame-by-Frame Animation & Effects Assembly
    t0 = time.time()
    logger.info(f"[Stage 3/4] Generating {total_output_frames} video frames with zoom, flash cuts & typography...")
    
    rendered_frames = []
    target_center = (zoom_target.get("cx", 0.5), zoom_target.get("cy", 0.45))
    
    # Define flash cut timestamps (in seconds)
    # Phonk drops at t=2.0s, t=2.7s, t=3.55s, t=4.44s
    flash_times = [2.0, 2.7, 3.55, 4.44][:style_config.get("flash_count", 3)]
    flash_frames = set()
    for ft in flash_times:
        f_idx = int(ft * fps)
        flash_frames.add(f_idx)
        flash_frames.add(f_idx + 1) # 2 frames flash duration
        
    rng_shake = np.random.RandomState(999)
    num_keys = len(composited_keyframes)
    
    for f in range(total_output_frames):
        t_sec = f / fps
        t_norm = f / total_output_frames
        
        # 1. Select Keyframe
        if t_sec < 2.0:
            # Slow intro progression across the first few keyframes
            key_idx = min(num_keys - 1, int((t_sec / 2.0) * (num_keys * 0.4)))
        elif t_sec < 4.5:
            # Fast rhythm cuts synced to beats
            beat_progress = (t_sec - 2.0) / 2.5
            key_idx = int(beat_progress * (num_keys - 1) * 2.0) % num_keys
        else:
            # Final freeze frame (dramatic hold on last keyframe)
            key_idx = num_keys - 1
            
        base_frame = composited_keyframes[key_idx].copy()
        
        # 2. Dynamic Zoom Calculation
        # Smooth zoom from 1.0x up to 1.32x focused on the face box
        if t_sec < 4.5:
            # Smooth ease-in zoom
            zoom = 1.0 + 0.32 * math.sin((t_sec / 4.5) * (math.pi / 2.0))
        else:
            # Hold maximum zoom on freeze frame
            zoom = 1.32
            
        frame_zoomed = apply_zoom_and_crop(base_frame, zoom, target_center)
        
        # 3. Camera Shake for FIRE style on drops
        if style_config.get("shake_enabled", False) and 2.0 <= t_sec <= 4.5:
            shake_amp = int(w_factor := output_width * 0.015)
            dx = rng_shake.randint(-shake_amp, shake_amp + 1)
            dy = rng_shake.randint(-shake_amp, shake_amp + 1)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame_zoomed = cv2.warpAffine(frame_zoomed, M, (output_width, output_height), borderMode=cv2.BORDER_REFLECT)
            
        # 4. White Flash Cuts
        if f in flash_frames:
            flash_col = np.array(style_config.get("flash_color", (255, 255, 255)), dtype=np.float32)
            # High-intensity flash blend
            frame_zoomed = cv2.addWeighted(frame_zoomed, 0.15, np.full_like(frame_zoomed, flash_col, dtype=np.uint8), 0.85, 0)
            
        # 5. Phonk Text Overlay with Impact Pop & Fade-in
        # Text starts fading in at t=2.3s, fully visible by t=2.8s
        if t_sec < 2.3:
            alpha_text = 0.0
            scale_pop = 1.0
        elif t_sec < 2.8:
            fade_p = (t_sec - 2.3) / 0.5
            alpha_text = fade_p
            # Impact bounce: scales from 1.25 down to 1.0
            scale_pop = 1.25 - 0.25 * fade_p
        else:
            alpha_text = 1.0
            # Subtle rhythmic breathing pulse
            scale_pop = 1.0 + 0.025 * math.sin((t_sec - 2.8) * 8.0)
            
        final_rgb = draw_styled_text_overlay(
            frame_zoomed,
            title_text=style_config["title"],
            tagline_text=style_config["tagline"],
            style_config=style_config,
            alpha_fade=alpha_text,
            scale_pop=scale_pop
        )
        
        rendered_frames.append(final_rgb)
        
    t_stage3 = time.time() - t0
    logger.info(f"[Stage 3/4 Completed] Frame generation took: {t_stage3:.2f}s")
    
    # Stage 4: Video Encoding & Audio Multiplexing via MoviePy
    t0 = time.time()
    logger.info(f"[Stage 4/4] Encoding {total_output_frames} frames to H.264 MP4 with AAC audio...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    from moviepy.editor import ImageSequenceClip, AudioFileClip
    
    video_clip = ImageSequenceClip(rendered_frames, fps=fps)
    audio_clip = AudioFileClip(audio_path).subclip(0, duration)
    final_video = video_clip.set_audio(audio_clip)
    
    # Export with ultrafast preset for lightning-fast expo render times
    final_video.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=4,
        logger=None # Suppress internal verbose logger
    )
    
    video_clip.close()
    audio_clip.close()
    final_video.close()
    
    t_stage4 = time.time() - t0
    total_time = time.time() - t_start_total
    
    logger.info(f"[Stage 4/4 Completed] MP4 encoding took: {t_stage4:.2f}s")
    logger.info(f"=== FULL PIPELINE COMPLETED IN {total_time:.2f}s! Output: {output_path} ({os.path.getsize(output_path)} bytes) ===")
    
    # Print timing summary to stdout for easy review by judges (safe for all Windows code pages)
    print("\n" + "=" * 55)
    print(f">> AURA EDIT RENDER TIMING BENCHMARK ({style_name}):")
    print(f"   Stage 1 (Matting & Compositing):  {t_stage1:6.2f}s")
    print(f"   Stage 2 (Audio Track Synth/Load): {t_stage2:6.2f}s")
    print(f"   Stage 3 (VFX Animation Assembly): {t_stage3:6.2f}s")
    print(f"   Stage 4 (H.264/AAC Export):       {t_stage4:6.2f}s")
    print(f"  ---------------------------------------------")
    print(f"  ** TOTAL PIPELINE RENDER TIME:      {total_time:6.2f}s (Target: < 25s)")
    print("=" * 55 + "\n")
    
    return output_path
