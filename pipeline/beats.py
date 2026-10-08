"""
pipeline/beats.py - Beat detection and rhythm analysis for AURA EDIT.
Analyzes 6-second audio tracks using librosa to extract BPM, beat timestamps,
and strongest onset hit beats for video transition synchronization.
Includes robust fallback to fixed tempo grids and exact synthesis grids.
"""

import os
import re
import random
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

logger = logging.getLogger("aura_edit.beats")

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac"}

@dataclass
class BeatSyncData:
    audio_path: str
    bpm: float
    beat_times: List[float]       # All beat timestamps in seconds
    hit_beats: List[float]        # Subset of 5-8 strongest beat timestamps
    is_synthesized: bool = False

def extract_bpm_from_filename(filename: str) -> Optional[float]:
    """Extracts BPM if present in filename like 'phonk_140.mp3' or 'beat_135.wav'."""
    match = re.search(r'(?:bpm[_\-\s]?|phonk[_\-\s]?|[_\-\s])(\d{2,3})(?:bpm)?', filename, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 70.0 <= val <= 220.0:
            return val
    return None

def generate_fixed_beat_grid(
    bpm: float,
    duration: float = 6.0,
    target_hits: int = 6
) -> Tuple[List[float], List[float]]:
    """Generates an exact mathematical beat grid for a given BPM."""
    beat_interval = 60.0 / bpm
    all_beats = []
    t = beat_interval
    while t < duration - 0.2:
        all_beats.append(round(t, 4))
        t += beat_interval

    # Select 5-8 evenly spaced hit beats starting around 0.8s up to ~4.5s
    candidate_hits = [b for b in all_beats if 0.7 <= b <= 4.6]
    if len(candidate_hits) <= target_hits:
        hit_beats = candidate_hits
    else:
        indices = np.round(np.linspace(0, len(candidate_hits) - 1, target_hits)).astype(int)
        hit_beats = [candidate_hits[i] for i in indices]

    # Ensure strictly 5-8 hit beats
    if len(hit_beats) < 5 and len(all_beats) >= 5:
        hit_beats = all_beats[:6]

    return all_beats, hit_beats

def analyze_audio_beats(
    audio_path: str,
    duration: float = 6.0,
    known_bpm: Optional[float] = None
) -> BeatSyncData:
    """
    Performs beat tracking and onset strength analysis using librosa.
    Falls back gracefully to fixed grid if detection fails or tempo looks erratic.
    """
    if known_bpm is not None and known_bpm > 0:
        logger.info(f"Using known BPM {known_bpm:.1f} for synthesized audio (exact grid, no detection required).")
        all_beats, hit_beats = generate_fixed_beat_grid(known_bpm, duration=duration, target_hits=6)
        logger.info(f">> BEAT SYNC RESULT: BPM={known_bpm:.1f} | Total Beats={len(all_beats)} | Hit Beats ({len(hit_beats)})={[round(b, 3) for b in hit_beats]}")
        return BeatSyncData(
            audio_path=audio_path,
            bpm=known_bpm,
            beat_times=all_beats,
            hit_beats=hit_beats,
            is_synthesized=True
        )

    filename = os.path.basename(audio_path)
    file_bpm = extract_bpm_from_filename(filename)

    try:
        import librosa
        # Load at 22050 Hz for fast onset & beat tracking
        y, sr = librosa.load(audio_path, sr=22050, duration=duration)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)

        if hasattr(tempo, "__iter__"):
            detected_bpm = float(tempo[0]) if len(tempo) > 0 else 130.0
        else:
            detected_bpm = float(tempo)

        detected_beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        detected_beat_times = [round(b, 4) for b in detected_beat_times if 0.2 <= b < duration - 0.2]

        # Reliability check: needs at least 6 beats and tempo between 80 and 200 BPM
        if 80.0 <= detected_bpm <= 200.0 and len(detected_beat_times) >= 6:
            bpm = detected_bpm
            all_beats = detected_beat_times

            # Evaluate onset strength for each beat to isolate 5-8 hit beats
            scored_beats = []
            for b in all_beats:
                frame_idx = librosa.time_to_frames(b, sr=sr)
                if 0 <= frame_idx < len(onset_env):
                    score = float(onset_env[frame_idx])
                else:
                    score = 0.0
                # Give higher weight to beats in the build-up window (0.8s to 4.5s)
                if 0.7 <= b <= 4.5:
                    score *= 1.3
                scored_beats.append((b, score))

            # Filter candidates in the primary edit window
            window_candidates = [item for item in scored_beats if 0.6 <= item[0] <= 4.8]
            if len(window_candidates) < 5:
                window_candidates = scored_beats

            # Sort by onset strength descending
            window_candidates.sort(key=lambda x: x[1], reverse=True)

            # Pick strongest beats while maintaining >= 0.32s separation
            selected_hits = []
            for b, _ in window_candidates:
                if all(abs(b - existing) >= 0.32 for existing in selected_hits):
                    selected_hits.append(b)
                if len(selected_hits) >= 7:
                    break

            # If still fewer than 5, fill from all_beats
            if len(selected_hits) < 5:
                for b in all_beats:
                    if b not in selected_hits and all(abs(b - existing) >= 0.32 for existing in selected_hits):
                        selected_hits.append(b)
                    if len(selected_hits) >= 6:
                        break

            hit_beats = sorted(selected_hits)
            logger.info(f">> BEAT SYNC RESULT (librosa): BPM={bpm:.1f} | Total Beats={len(all_beats)} | Hit Beats ({len(hit_beats)})={[round(b, 3) for b in hit_beats]}")
            return BeatSyncData(
                audio_path=audio_path,
                bpm=bpm,
                beat_times=all_beats,
                hit_beats=hit_beats,
                is_synthesized=False
            )
        else:
            logger.warning(f"librosa detection unreliable (BPM={detected_bpm:.1f}, Beats={len(detected_beat_times)}). Using fallback grid.")

    except Exception as e:
        logger.warning(f"librosa analysis failed: {e}. Using fallback grid.")

    # Fallback to fixed grid
    fallback_bpm = file_bpm if file_bpm is not None else 130.0
    all_beats, hit_beats = generate_fixed_beat_grid(fallback_bpm, duration=duration, target_hits=6)
    logger.info(f">> BEAT SYNC RESULT (fallback grid): BPM={fallback_bpm:.1f} | Total Beats={len(all_beats)} | Hit Beats ({len(hit_beats)})={[round(b, 3) for b in hit_beats]}")
    return BeatSyncData(
        audio_path=audio_path,
        bpm=fallback_bpm,
        beat_times=all_beats,
        hit_beats=hit_beats,
        is_synthesized=False
    )
