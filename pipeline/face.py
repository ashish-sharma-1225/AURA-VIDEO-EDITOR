"""
pipeline/face.py - Face detection and zoom target calculation.
Uses MediaPipe FaceDetector as primary, falling back to OpenCV Haar cascade,
and finally defaulting to frame center if no face is detected.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger("aura_edit.face")

_MP_DETECTOR = None
_HAAR_CASCADE = None

def _get_models_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "assets", "models")

def _init_mediapipe_detector():
    global _MP_DETECTOR
    if _MP_DETECTOR is not None:
        return _MP_DETECTOR
        
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks import python as mp_tasks

        model_path = os.path.join(_get_models_dir(), "blaze_face_short_range.tflite")
        if not os.path.exists(model_path):
            logger.warning(f"MediaPipe model file not found at {model_path}. Trying fallback.")
            return None

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.4)
        _MP_DETECTOR = vision.FaceDetector.create_from_options(options)
        logger.info("MediaPipe FaceDetector initialized successfully.")
        return _MP_DETECTOR
    except Exception as e:
        logger.warning(f"Could not initialize MediaPipe FaceDetector: {e}. Falling back to Haar Cascade.")
        return None

def _init_haar_cascade():
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None:
        return _HAAR_CASCADE
        
    model_path = os.path.join(_get_models_dir(), "haarcascade_frontalface_default.xml")
    if os.path.exists(model_path):
        _HAAR_CASCADE = cv2.CascadeClassifier(model_path)
        logger.info("OpenCV Haar cascade initialized successfully from assets.")
        return _HAAR_CASCADE
        
    # Check default cv2 data path
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        builtin_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(builtin_path):
            _HAAR_CASCADE = cv2.CascadeClassifier(builtin_path)
            logger.info("OpenCV Haar cascade initialized from cv2.data.")
            return _HAAR_CASCADE

    logger.warning("Haar cascade XML not found.")
    return None

def detect_face_in_frame(rgb_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detects the primary face in a single RGB frame.
    Returns (x, y, w, h) in pixel coordinates, or None if no face found.
    """
    h_img, w_img = rgb_frame.shape[:2]
    
    # 1. Try MediaPipe FaceDetector
    detector = _init_mediapipe_detector()
    if detector is not None:
        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if detection_result and detection_result.detections:
                # Find largest detection box
                best_box = None
                max_area = 0
                for det in detection_result.detections:
                    bbox = det.bounding_box
                    # Ensure bbox is within frame boundaries
                    x = max(0, int(bbox.origin_x))
                    y = max(0, int(bbox.origin_y))
                    w = min(w_img - x, int(bbox.width))
                    h = min(h_img - y, int(bbox.height))
                    if w * h > max_area and w > 20 and h > 20:
                        max_area = w * h
                        best_box = (x, y, w, h)
                        
                if best_box is not None:
                    return best_box
        except Exception as e:
            logger.debug(f"MediaPipe detection failed on frame: {e}")
            
    # 2. Fall back to OpenCV Haar Cascade
    cascade = _init_haar_cascade()
    if cascade is not None:
        try:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) > 0:
                # Choose largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return (int(largest[0]), int(largest[1]), int(largest[2]), int(largest[3]))
        except Exception as e:
            logger.debug(f"Haar cascade detection failed on frame: {e}")

    return None

def detect_face_and_zoom_target(
    frames: List[np.ndarray]
) -> Tuple[Dict[str, float], List[Optional[Tuple[int, int, int, int]]]]:
    """
    Scans all frames, detects face locations, and computes a smooth relative zoom target
    (rel_center_x, rel_center_y, rel_width, rel_height) normalized to [0, 1].
    
    If no face is detected in any frame, defaults to center zoom (0.5, 0.45).
    """
    if not frames:
        return {"cx": 0.5, "cy": 0.45, "w": 0.4, "h": 0.4, "detected": False}, []
        
    h_img, w_img = frames[0].shape[:2]
    all_boxes: List[Optional[Tuple[int, int, int, int]]] = []
    valid_centers = []
    valid_sizes = []
    
    for i, frame in enumerate(frames):
        box = detect_face_in_frame(frame)
        all_boxes.append(box)
        if box is not None:
            x, y, bw, bh = box
            cx = (x + bw / 2.0) / w_img
            cy = (y + bh / 2.0) / h_img
            valid_centers.append((cx, cy))
            valid_sizes.append((bw / w_img, bh / h_img))
            
    if valid_centers:
        avg_cx = float(np.median([c[0] for c in valid_centers]))
        avg_cy = float(np.median([c[1] for c in valid_centers]))
        avg_w = float(np.median([s[0] for s in valid_sizes]))
        avg_h = float(np.median([s[1] for s in valid_sizes]))
        logger.info(f"Face detected in {len(valid_centers)}/{len(frames)} frames. Zoom target center: ({avg_cx:.2f}, {avg_cy:.2f})")
        target = {
            "cx": np.clip(avg_cx, 0.2, 0.8),
            "cy": np.clip(avg_cy, 0.2, 0.8),
            "w": np.clip(avg_w, 0.15, 0.7),
            "h": np.clip(avg_h, 0.15, 0.7),
            "detected": True
        }
    else:
        logger.info("No face detected in any frame. Defaulting zoom target to frame center.")
        target = {
            "cx": 0.5,
            "cy": 0.45,
            "w": 0.4,
            "h": 0.4,
            "detected": False
        }
        
    return target, all_boxes

def get_best_face_crop(
    frames: List[np.ndarray],
    boxes: List[Optional[Tuple[int, int, int, int]]]
) -> Image.Image:
    """
    Extracts the highest quality face crop across frames for emotion classification.
    Pads the box to capture full facial expressions (eyes, mouth, brows).
    If no face detected, extracts the central portion of the middle frame.
    """
    h_img, w_img = frames[0].shape[:2]
    best_crop = None
    best_area = 0
    
    for frame, box in zip(frames, boxes):
        if box is not None:
            x, y, w, h = box
            area = w * h
            if area > best_area:
                best_area = area
                # Add 30% padding around face
                pad_x = int(w * 0.3)
                pad_y = int(h * 0.3)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_img, x + w + pad_x)
                y2 = min(h_img, y + h + pad_y)
                crop_arr = frame[y1:y2, x1:x2]
                if crop_arr.size > 0:
                    best_crop = Image.fromarray(crop_arr)
                    
    if best_crop is not None:
        return best_crop
        
    # Fallback: Center crop of middle frame
    mid_idx = len(frames) // 2
    mid_frame = frames[mid_idx]
    c_x, c_y = w_img // 2, int(h_img * 0.45)
    box_size = min(w_img, h_img) // 2
    x1 = max(0, c_x - box_size // 2)
    y1 = max(0, c_y - box_size // 2)
    x2 = min(w_img, x1 + box_size)
    y2 = min(h_img, y1 + box_size)
    crop_arr = mid_frame[y1:y2, x1:x2]
    return Image.fromarray(crop_arr)
