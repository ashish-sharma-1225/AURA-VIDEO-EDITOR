"""
pipeline/audio.py - Audio pipeline and Phonk/808 beat synthesizer.
Picks a track from assets/music/ or procedurally synthesizes an original
phonk-style 808 bass drop + beat track using NumPy.
"""

import os
import random
import numpy as np
from scipy.io import wavfile
from typing import Optional
import logging

logger = logging.getLogger("aura_edit.audio")

SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".aac", ".m4a", ".ogg", ".flac"}

def synthesize_phonk_beat(
    duration: float = 6.0,
    sample_rate: int = 44100,
    output_path: Optional[str] = None
) -> str:
    """
    Procedurally generates an original 6.0s Phonk / Trap beat with:
    - Deep distorted 808 bass slides
    - Phonk synth cowbell melodic lead
    - Crisp trap hi-hat rolls & snare hits
    - Mastered with soft-clip limiter and smooth fade-out.
    """
    total_samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, total_samples, endpoint=False)
    
    bpm = 135.0
    beat_sec = 60.0 / bpm
    
    # Left and Right stereo channels
    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)
    
    # 1. 808 Bass Line with Pitch Slide
    # Slide down on beat drops (t=0.0, 1.77, 3.55)
    drop_times = [0.0, 1.77, 3.55, 4.44]
    for drop_t in drop_times:
        start_idx = int(drop_t * sample_rate)
        end_idx = min(total_samples, start_idx + int(1.6 * sample_rate))
        dur_note = (end_idx - start_idx) / sample_rate
        if dur_note <= 0:
            continue
            
        t_note = np.linspace(0.0, dur_note, end_idx - start_idx, endpoint=False)
        # Pitch slides from 65Hz (C2) down to 38Hz (D1)
        freq_env = 65.0 - 27.0 * (t_note / dur_note) ** 0.5
        phase = 2.0 * np.pi * np.cumsum(freq_env) / sample_rate
        
        # Amplitude envelope (punchy attack, long decay)
        amp_env = np.exp(-t_note * 2.2)
        bass_wave = np.sin(phase) + 0.35 * np.sin(phase * 2.0) + 0.15 * np.sin(phase * 3.0)
        
        # Tube saturation / clipping for authentic Phonk grit
        bass_dist = np.tanh(bass_wave * 2.5) * amp_env * 0.75
        
        left[start_idx:end_idx] += bass_dist
        right[start_idx:end_idx] += bass_dist
        
    # 2. Phonk Synth / Cowbell Melody (Classic Memphis scale: D5, F5, G5, A5, C6)
    melody_notes = [
        (0.0, 587.33),   # D5
        (0.444, 784.0),  # G5
        (0.888, 698.46), # F5
        (1.333, 880.0),  # A5
        (1.777, 587.33), # D5
        (2.222, 1046.5), # C6
        (2.666, 880.0),  # A5
        (3.111, 784.0),  # G5
        (3.555, 587.33), # D5
        (4.0,   880.0),  # A5
        (4.444, 1046.5), # C6
        (4.888, 1174.6), # D6
        (5.333, 880.0),  # A5
    ]
    
    for note_t, freq in melody_notes:
        if note_t >= duration:
            continue
        start_idx = int(note_t * sample_rate)
        end_idx = min(total_samples, start_idx + int(0.35 * sample_rate))
        dur_note = (end_idx - start_idx) / sample_rate
        if dur_note <= 0:
            continue
            
        t_note = np.linspace(0.0, dur_note, end_idx - start_idx, endpoint=False)
        amp = np.exp(-t_note * 9.0)
        # Synth cowbell: square-ish tone with metallic ring
        synth_wave = (
            np.sin(2.0 * np.pi * freq * t_note) +
            0.6 * np.sin(2.0 * np.pi * freq * 1.5 * t_note) +
            0.4 * np.sin(2.0 * np.pi * freq * 2.0 * t_note)
        )
        synth_wave = np.clip(synth_wave * 1.8, -1.0, 1.0) * amp * 0.45
        
        # Slight stereo chorus pan
        left[start_idx:end_idx] += synth_wave * 0.85
        right[start_idx:end_idx] += synth_wave * 0.95

    # 3. Punchy Trap Kick and Snare/Clap
    kick_times = [0.0, 0.444, 0.888, 1.777, 2.222, 2.666, 3.555, 4.0, 4.444, 4.888, 5.333]
    snare_times = [0.888, 1.777, 2.666, 3.555, 4.444, 5.333]
    
    # Kicks
    for kt in kick_times:
        start_idx = int(kt * sample_rate)
        end_idx = min(total_samples, start_idx + int(0.22 * sample_rate))
        dur_k = (end_idx - start_idx) / sample_rate
        if dur_k <= 0:
            continue
        t_k = np.linspace(0.0, dur_k, end_idx - start_idx, endpoint=False)
        k_freq = 150.0 * np.exp(-t_k * 30.0) + 45.0
        k_wave = np.sin(2.0 * np.pi * np.cumsum(k_freq) / sample_rate) * np.exp(-t_k * 18.0) * 0.7
        left[start_idx:end_idx] += k_wave
        right[start_idx:end_idx] += k_wave
        
    # Snares / Claps (Snappy white noise burst + body)
    rng = np.random.RandomState(1337)
    for st in snare_times:
        start_idx = int(st * sample_rate)
        end_idx = min(total_samples, start_idx + int(0.25 * sample_rate))
        dur_s = (end_idx - start_idx) / sample_rate
        if dur_s <= 0:
            continue
        t_s = np.linspace(0.0, dur_s, end_idx - start_idx, endpoint=False)
        noise = rng.uniform(-1.0, 1.0, len(t_s))
        s_body = np.sin(2.0 * np.pi * 220.0 * t_s) * np.exp(-t_s * 25.0)
        s_wave = (noise * 0.7 + s_body * 0.3) * np.exp(-t_s * 14.0) * 0.55
        left[start_idx:end_idx] += s_wave * 0.9
        right[start_idx:end_idx] += s_wave * 1.0
        
    # 4. Hi-Hat 16th note rolls
    hat_step = beat_sec / 4.0
    for ht in np.arange(0.0, duration, hat_step):
        start_idx = int(ht * sample_rate)
        end_idx = min(total_samples, start_idx + int(0.05 * sample_rate))
        dur_h = (end_idx - start_idx) / sample_rate
        if dur_h <= 0:
            continue
        t_h = np.linspace(0.0, dur_h, end_idx - start_idx, endpoint=False)
        h_noise = rng.uniform(-1.0, 1.0, len(t_h)) * np.exp(-t_h * 75.0) * 0.22
        left[start_idx:end_idx] += h_noise
        right[start_idx:end_idx] += h_noise

    # 5. Smooth Fade-Out (last 0.6 seconds)
    fade_start = max(0.0, duration - 0.6)
    fade_idx = int(fade_start * sample_rate)
    fade_len = total_samples - fade_idx
    if fade_len > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32) ** 1.5
        left[fade_idx:] *= fade_curve
        right[fade_idx:] *= fade_curve
        
    # Master Limiter (Soft clip tanh)
    stereo = np.column_stack([left, right])
    stereo = np.tanh(stereo * 1.25) * 0.92
    
    # Convert to 16-bit PCM
    pcm_audio = (stereo * 32767.0).astype(np.int16)
    
    if output_path is None:
        import tempfile
        fd, output_path = tempfile.mkstemp(suffix="_aura_phonk_beat.wav")
        os.close(fd)
        
    wavfile.write(output_path, sample_rate, pcm_audio)
    logger.info(f"Synthesized original 6.0s phonk beat at: {output_path}")
    return output_path

def get_or_synthesize_audio(
    music_folder: Optional[str] = None,
    output_duration: float = 6.0,
    temp_dir: Optional[str] = None
) -> str:
    """
    Retrieves a background audio track trimmed to output_duration (6.0s).
    1. Checks music_folder (defaults to assets/music/) for audio files.
    2. If found, randomly picks one and trims with a 0.5s fade-out.
    3. If none found, synthesizes a phonk beat track.
    
    Returns path to a 6.0s audio file.
    """
    if music_folder is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        music_folder = os.path.join(project_root, "assets", "music")
        
    audio_candidates = []
    if os.path.exists(music_folder) and os.path.isdir(music_folder):
        for fname in os.listdir(music_folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_AUDIO_EXTS:
                full_path = os.path.join(music_folder, fname)
                if os.path.getsize(full_path) > 1000:
                    audio_candidates.append(full_path)
                    
    if audio_candidates:
        selected_file = random.choice(audio_candidates)
        logger.info(f"Found {len(audio_candidates)} tracks in assets/music/. Selected: {os.path.basename(selected_file)}")
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(selected_file)
            
            # Trim or loop to output_duration
            if clip.duration >= output_duration:
                # Pick a random 6s window or from start
                start_t = 0.0
                if clip.duration > output_duration + 5.0:
                    start_t = random.uniform(0.0, min(15.0, clip.duration - output_duration))
                trimmed = clip.subclip(start_t, start_t + output_duration)
            else:
                # Loop if shorter than 6s
                loops = int(np.ceil(output_duration / clip.duration))
                from moviepy.editor import concatenate_audioclips
                trimmed = concatenate_audioclips([clip] * loops).subclip(0, output_duration)
                
            # Apply fade out
            final_audio = trimmed.audio_fadeout(0.6)
            
            # Export to temp wav file
            if temp_dir is None:
                import tempfile
                temp_dir = tempfile.gettempdir()
            out_path = os.path.join(temp_dir, f"aura_audio_{random.randint(1000, 9999)}.wav")
            final_audio.write_audiofile(out_path, fps=44100, nbytes=2, codec='pcm_s16le', logger=None)
            clip.close()
            final_audio.close()
            return out_path
        except Exception as e:
            logger.warning(f"Error processing selected audio file {selected_file}: {e}. Falling back to synthesizer.")

    # Fallback to procedural synthesizer
    if temp_dir is None:
        import tempfile
        temp_dir = tempfile.gettempdir()
    synth_path = os.path.join(temp_dir, f"aura_synth_{random.randint(1000, 9999)}.wav")
    return synthesize_phonk_beat(duration=output_duration, output_path=synth_path)
