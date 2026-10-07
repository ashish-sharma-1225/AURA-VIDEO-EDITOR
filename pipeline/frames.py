"""
pipeline/frames.py - Video decoding and frame extraction.
Extracts ~12 evenly spaced frames from input webcam or uploaded clip.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger("aura_edit.frames")

def extract_clip_frames(
    video_path: str,
    target_num_frames: int = 12,
    max_dimension: int = 720
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extracts evenly spaced frames from a video file and normalizes their size.
    
    Args:
        video_path: Path to the input video file (.mp4, .webm, .mov, etc.)
        target_num_frames: Desired number of keyframes (default 12)
        max_dimension: Max width or height for processing downscale (default 720)
        
    Returns:
        Tuple of (list of RGB numpy arrays, metadata dict)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Loaded video: {video_path} (Frames: {total_frames}, FPS: {fps:.1f}, Res: {orig_w}x{orig_h})")
    
    # Read all frames or index seek
    all_frames = []
    success, frame = cap.read()
    while success:
        all_frames.append(frame)
        success, frame = cap.read()
    cap.release()
    
    if len(all_frames) == 0:
        raise ValueError(f"Video file {video_path} contained no readable frames.")
        
    # Calculate sample indices
    actual_count = len(all_frames)
    if actual_count <= target_num_frames:
        indices = list(range(actual_count))
        # Pad up to target_num_frames if needed
        while len(indices) < target_num_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, actual_count - 1, target_num_frames, dtype=int).tolist()
        
    sampled_frames = []
    for idx in indices:
        bgr_frame = all_frames[idx]
        
        # Downscale proportionally if larger than max_dimension
        h, w = bgr_frame.shape[:2]
        scale = min(1.0, max_dimension / max(h, w))
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            bgr_frame = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        sampled_frames.append(rgb_frame)
        
    metadata = {
        "original_width": orig_w,
        "original_height": orig_h,
        "total_frames": actual_count,
        "fps": fps,
        "duration": actual_count / fps,
        "sampled_count": len(sampled_frames),
        "processed_resolution": (sampled_frames[0].shape[1], sampled_frames[0].shape[0]),
    }
    
    logger.info(f"Extracted {len(sampled_frames)} frames scaled to {metadata['processed_resolution']}.")
    return sampled_frames, metadata
