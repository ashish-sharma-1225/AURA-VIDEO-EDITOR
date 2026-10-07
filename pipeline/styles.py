"""
pipeline/styles.py - Visual styling definitions for Phonk & Anime aesthetic.
Defines parameters for DARK, GOLD, and FIRE visual themes.
"""

from typing import Dict, Any

STYLES: Dict[str, Dict[str, Any]] = {
    "DARK": {
        "name": "DARK",
        "title": "AURA +9999",
        "tagline": "ABSOLUTE FOCUS",
        "description": "Desaturated cinematic tones, deep vignette, dark red edge glow, slow dramatic zoom.",
        "glow_color": (220, 25, 45),            # RGB Crimson Red
        "bg_base_color": (15, 6, 12),           # RGB Deep dark shadow
        "bg_accent_color": (65, 10, 22),        # RGB Dark crimson
        "streak_colors": [
            (220, 25, 45),
            (160, 15, 35),
            (100, 10, 25)
        ],
        "saturation_factor": 0.65,               # Desaturated
        "contrast_factor": 1.25,                 # High contrast
        "vignette_intensity": 0.85,             # Deep vignette
        "flash_count": 3,
        "flash_color": (255, 230, 230),
        "text_color": (255, 255, 255),
        "text_outline": (180, 15, 35),
        "shake_enabled": False,
        "badge_color": "#FF2A4D"
    },
    "GOLD": {
        "name": "GOLD",
        "title": "MAIN CHARACTER",
        "tagline": "LEGENDARY ENERGY",
        "description": "Warm golden glow, bright luminous flashes, radiant light rays.",
        "glow_color": (255, 195, 30),           # RGB Rich Gold
        "bg_base_color": (25, 18, 5),           # RGB Deep warm dark
        "bg_accent_color": (75, 52, 10),        # RGB Amber gold
        "streak_colors": [
            (255, 215, 50),
            (255, 160, 20),
            (220, 120, 10)
        ],
        "saturation_factor": 1.20,               # Warm vibrancy
        "contrast_factor": 1.15,
        "vignette_intensity": 0.60,
        "flash_count": 4,
        "flash_color": (255, 250, 220),
        "text_color": (255, 250, 225),
        "text_outline": (200, 140, 15),
        "shake_enabled": False,
        "badge_color": "#FFC107"
    },
    "FIRE": {
        "name": "FIRE",
        "title": "NO MERCY MODE",
        "tagline": "UNSTOPPABLE FORCE",
        "description": "High contrast, blazing orange-red tint, camera shake, hyper-dynamic energy.",
        "glow_color": (255, 70, 15),            # RGB Blazing Ember
        "bg_base_color": (28, 8, 4),            # RGB Scorched Dark
        "bg_accent_color": (95, 22, 6),         # RGB Molten Red
        "streak_colors": [
            (255, 90, 20),
            (255, 160, 30),
            (200, 40, 10)
        ],
        "saturation_factor": 1.35,               # Vivid fiery
        "contrast_factor": 1.40,                 # Harsh dramatic contrast
        "vignette_intensity": 0.75,
        "flash_count": 4,
        "flash_color": (255, 220, 180),
        "text_color": (255, 245, 230),
        "text_outline": (220, 45, 10),
        "shake_enabled": True,                  # Impact shake
        "badge_color": "#FF5722"
    }
}

def get_style_config(style_name: str) -> Dict[str, Any]:
    """Retrieves style configuration, defaulting to DARK if unknown."""
    key = style_name.upper().strip()
    return STYLES.get(key, STYLES["DARK"])
