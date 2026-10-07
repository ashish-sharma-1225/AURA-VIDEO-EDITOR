"""
pipeline/background.py - Procedural background generation and subject compositing with aura glow.
100% generated with pure NumPy and OpenCV (no external images required).
Features dynamic gradients, film grain noise, moving anime light streaks, and neon aura edge glow.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("aura_edit.background")

_REMBG_SESSION = None

def get_rembg_session():
    """Returns or lazily creates a cached rembg u2net session."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        import rembg
        _REMBG_SESSION = rembg.new_session("u2net")
    return _REMBG_SESSION

def extract_foreground_matte(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Removes background using rembg u2net, returning an RGBA array.
    If rembg fails, falls back gracefully to a soft centered vignette mask.
    """
    try:
        import rembg
        session = get_rembg_session()
        rgba = rembg.remove(rgb_frame, session=session)
        if rgba.shape[2] == 4:
            return rgba
    except Exception as e:
        logger.warning(f"rembg background removal failed: {e}. Falling back to procedural mask.")
        
    # Fallback soft ellipse mask
    h, w = rgb_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, int(h * 0.55)), (int(w * 0.38), int(h * 0.45)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 20)
    rgba = np.dstack([rgb_frame, mask])
    return rgba

def generate_dynamic_background(
    width: int,
    height: int,
    style_config: Dict[str, Any],
    t: float,
    seed: int = 42
) -> np.ndarray:
    """
    Generates a stylized phonk/anime procedural background for timestamp t in [0.0, 1.0].
    
    Includes:
    - Deep gradient backdrop
    - Radial aura glow behind the subject
    - Animated diagonal light streaks / neon rays
    - Fine film grain / noise
    """
    # 1. Base vertical gradient
    base_col = np.array(style_config["bg_base_color"], dtype=np.float32)
    accent_col = np.array(style_config["bg_accent_color"], dtype=np.float32)
    
    # Linear ramp from bottom/top
    y_coords = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    gradient = (1.0 - y_coords) * base_col + y_coords * accent_col
    bg = np.tile(gradient, (1, width, 1)).astype(np.float32) # Full shape: (height, width, 3)
    
    # 2. Radial back-light / ambient aura centered around upper torso (cx=0.5, cy=0.45)
    cx, cy = int(width * 0.5), int(height * 0.45)
    max_radius = max(width, height) * 0.65
    y_grid, x_grid = np.ogrid[:height, :width]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    radial_mask = np.clip(1.0 - np.sqrt(dist_sq) / max_radius, 0.0, 1.0) ** 2.0
    
    glow_col = np.array(style_config["glow_color"], dtype=np.float32)
    bg += radial_mask[:, :, None] * glow_col * 0.45
    
    # 3. Dynamic moving light streaks (anime speed lines / neon bars)
    streak_colors = style_config.get("streak_colors", [glow_col])
    num_streaks = 5
    streak_layer = np.zeros((height, width, 3), dtype=np.float32)
    
    for i in range(num_streaks):
        col = np.array(streak_colors[i % len(streak_colors)], dtype=np.float32)
        # Position oscillates or shifts with time t
        speed = 1.2 + (i * 0.4)
        x_pos = int(((i * 0.22 + t * speed) % 1.2 - 0.1) * width)
        thickness = int(width * (0.015 + (i % 3) * 0.01))
        
        # Draw soft angled streak
        streak_mask = np.zeros((height, width), dtype=np.uint8)
        pt1 = (x_pos - int(height * 0.25), 0)
        pt2 = (x_pos + int(height * 0.25), height)
        cv2.line(streak_mask, pt1, pt2, 255, thickness)
        
        # Blur the streak for luminous neon bloom
        streak_blur = cv2.GaussianBlur(streak_mask, (31, 31), 15).astype(np.float32) / 255.0
        streak_layer += streak_blur[:, :, None] * col * 0.35
        
    bg += streak_layer
    
    # 4. Animated film grain / noise
    rng = np.random.RandomState(seed + int(t * 24))
    noise = rng.normal(0.0, 8.0, (height, width, 1)).astype(np.float32)
    bg = np.clip(bg + noise, 0.0, 255.0)
    
    return bg.astype(np.uint8)

def apply_color_grading(
    rgb_img: np.ndarray,
    saturation_factor: float,
    contrast_factor: float
) -> np.ndarray:
    """Applies saturation scaling and S-curve contrast adjustments."""
    # Convert to HSV to scale saturation
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0.0, 255.0)
    rgb_graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Contrast adjustment centered at mid-gray (128)
    rgb_graded = (rgb_graded - 128.0) * contrast_factor + 128.0
    return np.clip(rgb_graded, 0.0, 255.0).astype(np.uint8)

def apply_vignette(rgb_img: np.ndarray, intensity: float = 0.75) -> np.ndarray:
    """Applies a smooth cinematic vignette darkening around the edges."""
    h, w = rgb_img.shape[:2]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    
    radius = np.sqrt(xx ** 2 + yy ** 2)
    # Vignette falloff
    vignette = 1.0 - intensity * np.clip((radius - 0.45) / 0.8, 0.0, 1.0)
    vignette = vignette[:, :, None]
    
    return np.clip(rgb_img.astype(np.float32) * vignette, 0.0, 255.0).astype(np.uint8)

def composite_foreground_with_glow(
    fg_rgba: np.ndarray,
    bg_rgb: np.ndarray,
    style_config: Dict[str, Any]
) -> np.ndarray:
    """
    Composites the foreground cutout onto the background with:
    1. Color grading on subject (style saturation & contrast)
    2. Multi-layer neon aura edge glow around the person's silhouette
    3. Seamless alpha blending
    4. Cinematic vignette
    """
    fg_rgb = fg_rgba[:, :, :3]
    alpha = fg_rgba[:, :, 3].astype(np.float32) / 255.0
    h_bg, w_bg = bg_rgb.shape[:2]
    h_fg, w_fg = fg_rgb.shape[:2]
    
    # Resize fg to match bg if needed
    if (h_fg, w_fg) != (h_bg, w_bg):
        fg_rgb = cv2.resize(fg_rgb, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (w_bg, h_bg), interpolation=cv2.INTER_LINEAR)
        
    # 1. Color grade foreground according to style
    sat = style_config.get("saturation_factor", 1.0)
    contrast = style_config.get("contrast_factor", 1.0)
    graded_fg = apply_color_grading(fg_rgb, sat, contrast).astype(np.float32)
    
    # 2. Generate Aura Edge Glow from alpha channel
    glow_col = np.array(style_config["glow_color"], dtype=np.float32)
    
    # Multi-pass glow: tight crisp rim + wide soft bloom
    alpha_u8 = (alpha * 255.0).astype(np.uint8)
    
    # Inner/Rim glow
    rim_blur = cv2.GaussianBlur(alpha_u8, (25, 25), 8).astype(np.float32) / 255.0
    # Outer bloom
    bloom_blur = cv2.GaussianBlur(alpha_u8, (55, 55), 22).astype(np.float32) / 255.0
    
    combined_glow_mask = np.clip(rim_blur * 0.7 + bloom_blur * 0.5, 0.0, 1.0)
    
    # Glow appears outside the subject (behind the person)
    glow_behind = np.clip(combined_glow_mask - alpha * 0.8, 0.0, 1.0)[:, :, None]
    
    # Composite glow onto background
    bg_with_glow = bg_rgb.astype(np.float32) + glow_behind * glow_col * 0.85
    bg_with_glow = np.clip(bg_with_glow, 0.0, 255.0)
    
    # 3. Alpha blend person over (background + glow)
    alpha_3d = alpha[:, :, None]
    composited = graded_fg * alpha_3d + bg_with_glow * (1.0 - alpha_3d)
    composited = np.clip(composited, 0.0, 255.0).astype(np.uint8)
    
    # 4. Cinematic vignette
    vignette_int = style_config.get("vignette_intensity", 0.7)
    final_frame = apply_vignette(composited, vignette_int)
    
    return final_frame
