"""
pipeline/emotion.py - Facial expression analysis and style mapping.
Uses Hugging Face Transformers vision models to classify emotion and
maps it into one of three flattering, cinematic aura styles (DARK, GOLD, FIRE).
Never rates, ranks, or judges looks.
"""

import os
from PIL import Image
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger("aura_edit.emotion")

_EMOTION_PIPELINE = None

# Emotion to Style Mapping
# neutral / sad / fear -> DARK
# happy / surprise -> GOLD
# angry / disgust -> FIRE
EMOTION_TO_STYLE = {
    "neutral": "DARK",
    "sad": "DARK",
    "sadness": "DARK",
    "fear": "DARK",
    "fearful": "DARK",
    
    "happy": "GOLD",
    "happiness": "GOLD",
    "surprise": "GOLD",
    "surprised": "GOLD",
    
    "angry": "FIRE",
    "anger": "FIRE",
    "disgust": "FIRE",
    "disgusted": "FIRE",
}

def map_emotion_to_style(emotion_label: str) -> str:
    """Maps an emotion label to one of the 3 Aura styles: DARK, GOLD, FIRE."""
    norm = emotion_label.lower().strip()
    return EMOTION_TO_STYLE.get(norm, "DARK")

def _get_classifier_pipeline():
    global _EMOTION_PIPELINE
    if _EMOTION_PIPELINE is not None:
        return _EMOTION_PIPELINE
        
    from transformers import pipeline
    
    # Try primary model first
    primary = "dima806/facial_emotions_image_detection"
    fallback = "trpakov/vit-face-expression"
    
    try:
        logger.info(f"Loading primary emotion classifier: {primary}")
        _EMOTION_PIPELINE = pipeline(
            "image-classification",
            model=primary,
            device=-1 # CPU
        )
        logger.info(f"Primary emotion classifier loaded: {primary}")
        return _EMOTION_PIPELINE
    except Exception as e:
        logger.warning(f"Failed to load primary emotion model {primary}: {e}. Trying fallback {fallback}...")
        
    try:
        logger.info(f"Loading fallback emotion classifier: {fallback}")
        _EMOTION_PIPELINE = pipeline(
            "image-classification",
            model=fallback,
            device=-1 # CPU
        )
        logger.info(f"Fallback emotion classifier loaded: {fallback}")
        return _EMOTION_PIPELINE
    except Exception as e2:
        logger.error(f"Failed to load fallback emotion model {fallback}: {e2}")
        return None

def analyze_emotion(face_image: Image.Image) -> Tuple[str, float, str]:
    """
    Classifies the facial emotion on a cropped face image.
    
    Returns:
        (detected_emotion: str, confidence: float, chosen_style: str)
    """
    classifier = _get_classifier_pipeline()
    if classifier is None:
        logger.warning("Emotion classifier unavailable. Defaulting to neutral -> DARK.")
        return ("neutral", 1.0, "DARK")
        
    try:
        # Resize to standard input if huge
        if max(face_image.size) > 300:
            face_image = face_image.copy()
            face_image.thumbnail((224, 224))
            
        results = classifier(face_image)
        # Expected results format: [{'label': 'happy', 'score': 0.95}, ...]
        if results and isinstance(results, list):
            top_prediction = results[0]
            label = str(top_prediction.get("label", "neutral")).lower()
            score = float(top_prediction.get("score", 0.0))
            style = map_emotion_to_style(label)
            logger.info(f"Emotion detected: '{label}' (confidence: {score*100:.1f}%) -> Style: {style}")
            return (label.capitalize(), score, style)
    except Exception as e:
        logger.error(f"Error during emotion inference: {e}")
        
    return ("Neutral", 0.85, "DARK")
