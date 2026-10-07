"""
AURA EDIT - Phonk & Anime Style Aura Video Generator
AI Pipeline Package
"""

from .frames import extract_clip_frames
from .face import detect_face_and_zoom_target, get_best_face_crop
from .emotion import analyze_emotion, map_emotion_to_style
from .styles import STYLES, get_style_config
from .background import generate_dynamic_background, composite_foreground_with_glow
from .audio import get_or_synthesize_audio
from .compose import compose_aura_video

__all__ = [
    "extract_clip_frames",
    "detect_face_and_zoom_target",
    "get_best_face_crop",
    "analyze_emotion",
    "map_emotion_to_style",
    "STYLES",
    "get_style_config",
    "generate_dynamic_background",
    "composite_foreground_with_glow",
    "get_or_synthesize_audio",
    "compose_aura_video",
]
