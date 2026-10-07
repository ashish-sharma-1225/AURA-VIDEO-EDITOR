"""
AURA EDIT - Pre-download and cache all models for offline execution.
Ensures the app runs reliably without internet access during live expos.
"""

import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("download_models")

def download_file(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info(f"File already exists: {dest_path}")
        return
    logger.info(f"Downloading {url} -> {dest_path}")
    urllib.request.urlretrieve(url, dest_path)
    logger.info(f"Successfully downloaded {dest_path} ({os.path.getsize(dest_path)} bytes)")

def download_models():
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "assets", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. MediaPipe Face Detector model
    tflite_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    tflite_path = os.path.join(models_dir, "blaze_face_short_range.tflite")
    download_file(tflite_url, tflite_path)

    # 2. OpenCV Haar Cascade model fallback
    haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    haar_path = os.path.join(models_dir, "haarcascade_frontalface_default.xml")
    download_file(haar_url, haar_path)

    # 3. rembg u2net background removal model
    logger.info("Initializing rembg (u2net) model session...")
    try:
        import rembg
        rembg.new_session("u2net")
        logger.info("Rembg u2net model cached successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize rembg u2net: {e}")
        raise

    # 4. HuggingFace emotion classification models
    logger.info("Downloading HuggingFace emotion models...")
    from transformers import pipeline

    primary_model = "dima806/facial_emotions_image_detection"
    fallback_model = "trpakov/vit-face-expression"

    try:
        logger.info(f"Downloading primary model: {primary_model}")
        pipeline("image-classification", model=primary_model)
        logger.info(f"Primary model {primary_model} cached successfully.")
    except Exception as e:
        logger.warning(f"Error caching primary model {primary_model}: {e}")

    try:
        logger.info(f"Downloading fallback model: {fallback_model}")
        pipeline("image-classification", model=fallback_model)
        logger.info(f"Fallback model {fallback_model} cached successfully.")
    except Exception as e:
        logger.warning(f"Error caching fallback model {fallback_model}: {e}")

    logger.info("=== All models downloaded and cached for offline use! ===")

if __name__ == "__main__":
    download_models()
