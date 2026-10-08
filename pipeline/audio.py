"""
pipeline/audio.py - Beat-synced audio pipeline and Phonk synthesizer.
Handles:
- Music track selection, 6.0-second trimming, and fallback procedural Phonk synthesis
- Beat detection and hit beat extraction via pipeline/beats.py
- Sound effects (SFX) loading from assets/sfx/ or procedural synthesis of bass thumps and whooshes
- Audio ducking (30% attenuation for 0.1s on hit beats) and mastering
"""

import os
import random
import logging
from typing import Optional, List, Tuple
import numpy as np
from scipy.io import wavfile
try:
    import soundfile as sf
except ImportError:
    sf = None

from .beats import analyze_audio_beats, BeatSyncData, SUPPORTED_AUDIO_EXTS

logger = logging.getLogger("aura_edit.audio")

KNOWN_SYNTH_BPM = 135.0

def synthesize_phonk_beat(
    duration: float = 6.0,
    sample_rate: int = 44100,
    bpm: float = KNOWN_SYNTH_BPM,
    output_path: Optional[str] = None
) -> str:
    """
    Procedurally generates an authentic 6.0s Phonk / Trap beat at a known BPM:
    - Distorted 808 bass line aligned with beat grid
    - Phonk synth cowbell melody
    - Trap kicks, snappy claps, and 16th-note hi-hat rolls
    - Soft-clip limiter and smooth fade-out
    """
    total_samples = int(duration * sample_rate)
    t = np.linspace(0.0, duration, total_samples, endpoint=False)
    beat_sec = 60.0 / bpm

    left = np.zeros(total_samples, dtype=np.float32)
    right = np.zeros(total_samples, dtype=np.float32)

    # 1. 808 Bass Line with Pitch Slide aligned on beat drops
    # Drop times on beats 0, 4, 8, 10
    drop_beats = [0, 4, 8, 10]
    for b in drop_beats:
        drop_t = b * beat_sec
        if drop_t >= duration:
            continue
        start_idx = int(drop_t * sample_rate)
        end_idx = min(total_samples, start_idx + int(1.6 * sample_rate))
        dur_note = (end_idx - start_idx) / sample_rate
        if dur_note <= 0:
            continue

        t_note = np.linspace(0.0, dur_note, end_idx - start_idx, endpoint=False)
        freq_env = 65.0 - 27.0 * (t_note / dur_note) ** 0.5
        phase = 2.0 * np.pi * np.cumsum(freq_env) / sample_rate
        amp_env = np.exp(-t_note * 2.2)
        bass_wave = np.sin(phase) + 0.35 * np.sin(phase * 2.0) + 0.15 * np.sin(phase * 3.0)
        bass_dist = np.tanh(bass_wave * 2.5) * amp_env * 0.75

        left[start_idx:end_idx] += bass_dist
        right[start_idx:end_idx] += bass_dist

    # 2. Phonk Synth / Cowbell Melody (Memphis scale: D5, F5, G5, A5, C6)
    # Timings snapped to exact beat grid
    melody_grid = [
        (0.0 * beat_sec, 587.33),   # D5
        (1.0 * beat_sec, 784.0),    # G5
        (2.0 * beat_sec, 698.46),   # F5
        (3.0 * beat_sec, 880.0),    # A5
        (4.0 * beat_sec, 587.33),   # D5
        (5.0 * beat_sec, 1046.5),   # C6
        (6.0 * beat_sec, 880.0),    # A5
        (7.0 * beat_sec, 784.0),    # G5
        (8.0 * beat_sec, 587.33),   # D5
        (9.0 * beat_sec, 880.0),    # A5
        (10.0 * beat_sec, 1046.5),  # C6
        (11.0 * beat_sec, 1174.6),  # D6
        (12.0 * beat_sec, 880.0),   # A5
    ]

    for note_t, freq in melody_grid:
        if note_t >= duration:
            continue
        start_idx = int(note_t * sample_rate)
        end_idx = min(total_samples, start_idx + int(0.35 * sample_rate))
        dur_note = (end_idx - start_idx) / sample_rate
        if dur_note <= 0:
            continue

        t_note = np.linspace(0.0, dur_note, end_idx - start_idx, endpoint=False)
        amp = np.exp(-t_note * 9.0)
        synth_wave = (
            np.sin(2.0 * np.pi * freq * t_note) +
            0.6 * np.sin(2.0 * np.pi * freq * 1.5 * t_note) +
            0.4 * np.sin(2.0 * np.pi * freq * 2.0 * t_note)
        )
        synth_wave = np.clip(synth_wave * 1.8, -1.0, 1.0) * amp * 0.45
        left[start_idx:end_idx] += synth_wave * 0.85
        right[start_idx:end_idx] += synth_wave * 0.95

    # 3. Kicks and Snares/Claps
    kick_beats = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12]
    snare_beats = [2, 4, 6, 8, 10, 12]

    for kb in kick_beats:
        kt = kb * beat_sec
        if kt >= duration:
            continue
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

    rng = np.random.RandomState(1337)
    for sb in snare_beats:
        st = sb * beat_sec
        if st >= duration:
            continue
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

    stereo = np.column_stack([left, right])
    stereo = np.tanh(stereo * 1.25) * 0.92
    pcm_audio = (stereo * 32767.0).astype(np.int16)

    if output_path is None:
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), f"aura_synth_{random.randint(1000, 9999)}.wav")

    wavfile.write(output_path, sample_rate, pcm_audio)
    logger.info(f"Synthesized original 6.0s phonk beat at: {output_path} (BPM: {bpm})")
    return output_path

def synthesize_bass_thump_sfx(sample_rate: int = 44100) -> np.ndarray:
    """Synthesizes a punchy sub-bass impact thump (~0.22s)."""
    dur = 0.22
    n_samples = int(dur * sample_rate)
    t = np.linspace(0.0, dur, n_samples, endpoint=False)
    # Pitch drops from 140Hz down to 42Hz
    freq = 140.0 * np.exp(-t * 22.0) + 42.0
    phase = 2.0 * np.pi * np.cumsum(freq) / sample_rate
    amp = np.exp(-t * 16.0)
    wave = np.sin(phase)
    # Warm saturation
    wave = np.tanh(wave * 2.4) * amp * 0.85
    return wave.astype(np.float32)

def synthesize_whoosh_sfx(sample_rate: int = 44100) -> np.ndarray:
    """Synthesizes a dynamic noise rush / whoosh transition sfx (~0.30s)."""
    dur = 0.30
    n_samples = int(dur * sample_rate)
    t = np.linspace(0.0, dur, n_samples, endpoint=False)
    rng = np.random.RandomState(42)
    noise = rng.uniform(-1.0, 1.0, n_samples)
    # Envelope: rises to peak at 0.08s then decays
    peak_t = 0.08
    env = np.where(t < peak_t, t / peak_t, np.exp(-(t - peak_t) * 12.0))
    # Modulated whoosh tone
    sweep_freq = 400.0 + 800.0 * np.sin(np.pi * (t / dur))
    carrier = np.sin(2.0 * np.pi * np.cumsum(sweep_freq) / sample_rate)
    whoosh = (noise * 0.7 + carrier * 0.3) * env * 0.65
    return whoosh.astype(np.float32)

def _read_audio(fpath: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """Robust audio loader supporting sf, librosa, and scipy.io.wavfile."""
    if sf is not None:
        try:
            data, sr = sf.read(fpath, dtype='float32')
            return data, sr
        except Exception:
            pass
    try:
        import librosa
        data, sr = librosa.load(fpath, sr=target_sr, mono=False)
        if data.ndim == 2:
            data = data.T
        return data, sr
    except Exception:
        sr, data = wavfile.read(fpath)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483647.0
        return data, sr

def load_or_synthesize_sfx(sample_rate: int = 44100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads SFX from assets/sfx/ if available, otherwise synthesizes bass thump and whoosh.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sfx_dir = os.path.join(project_root, "assets", "sfx")

    thump = None
    whoosh = None

    if os.path.exists(sfx_dir) and os.path.isdir(sfx_dir):
        for fname in os.listdir(sfx_dir):
            low = fname.lower()
            fpath = os.path.join(sfx_dir, fname)
            if any(k in low for k in ["bass", "thump", "hit", "impact"]) and thump is None:
                try:
                    data, sr = _read_audio(fpath, target_sr=sample_rate)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    thump = data
                    logger.info(f"Loaded custom bass hit SFX: {fname}")
                except Exception as e:
                    logger.debug(f"Could not load SFX {fname}: {e}")
            elif any(k in low for k in ["whoosh", "sweep", "transition"]) and whoosh is None:
                try:
                    data, sr = _read_audio(fpath, target_sr=sample_rate)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    whoosh = data
                    logger.info(f"Loaded custom whoosh SFX: {fname}")
                except Exception as e:
                    logger.debug(f"Could not load SFX {fname}: {e}")

    if thump is None:
        thump = synthesize_bass_thump_sfx(sample_rate)
    if whoosh is None:
        whoosh = synthesize_whoosh_sfx(sample_rate)

    return thump, whoosh

def mix_beat_synced_audio(
    base_audio_path: str,
    hit_beats: List[float],
    duration: float = 6.0,
    sample_rate: int = 44100,
    output_path: Optional[str] = None
) -> str:
    """
    Applies audio ducking (30% reduction for 0.1s) and overlays punchy SFX
    on every hit beat for maximum impact.
    """
    data, sr = _read_audio(base_audio_path, target_sr=sample_rate)

    # Resample or adapt to target sample rate and stereo
    total_target_samples = int(duration * sample_rate)
    if data.ndim == 1:
        stereo = np.column_stack([data, data])
    else:
        stereo = data[:, :2]

    # Crop or pad to exact duration
    if len(stereo) < total_target_samples:
        pad = np.zeros((total_target_samples - len(stereo), 2), dtype=np.float32)
        stereo = np.vstack([stereo, pad])
    else:
        stereo = stereo[:total_target_samples]

    thump_sfx, whoosh_sfx = load_or_synthesize_sfx(sample_rate)

    # Apply ducking and SFX overlay on each hit beat
    duck_samples = int(0.10 * sample_rate) # 0.1s ducking
    fade_ramp = int(0.015 * sample_rate)   # smooth ramp in/out

    for i, hit_t in enumerate(hit_beats):
        hit_idx = int(hit_t * sample_rate)
        if hit_idx >= total_target_samples:
            continue

        # 1. Duck base music by ~30% (multiplier = 0.70)
        start_duck = max(0, hit_idx - int(0.01 * sample_rate))
        end_duck = min(total_target_samples, start_duck + duck_samples)
        duck_len = end_duck - start_duck

        if duck_len > 0:
            env = np.ones(duck_len, dtype=np.float32)
            # Dip down to 0.70
            env[:min(fade_ramp, duck_len)] = np.linspace(1.0, 0.70, min(fade_ramp, duck_len))
            env[max(0, duck_len - fade_ramp):] = np.linspace(0.70, 1.0, min(fade_ramp, duck_len))
            env[fade_ramp:max(0, duck_len - fade_ramp)] = 0.70
            stereo[start_duck:end_duck, 0] *= env
            stereo[start_duck:end_duck, 1] *= env

        # 2. Add SFX (alternating punchy bass thump or whoosh impact)
        if i % 2 == 0:
            sfx_to_add = thump_sfx
        else:
            # Layer whoosh and thump cleanly
            mix_len = max(len(thump_sfx), len(whoosh_sfx))
            combined_sfx = np.zeros(mix_len, dtype=np.float32)
            combined_sfx[:len(thump_sfx)] += thump_sfx * 0.75
            combined_sfx[:len(whoosh_sfx)] += whoosh_sfx * 0.55
            sfx_to_add = combined_sfx
        sfx_len = min(len(sfx_to_add), total_target_samples - hit_idx)
        if sfx_len > 0:
            stereo[hit_idx:hit_idx + sfx_len, 0] += sfx_to_add[:sfx_len] * 0.55
            stereo[hit_idx:hit_idx + sfx_len, 1] += sfx_to_add[:sfx_len] * 0.55

    # Master limiter: soft clip with tanh
    stereo = np.tanh(stereo * 1.15) * 0.94
    pcm_audio = (stereo * 32767.0).astype(np.int16)

    if output_path is None:
        import tempfile
        output_path = os.path.join(tempfile.gettempdir(), f"aura_mixed_{random.randint(1000, 9999)}.wav")

    wavfile.write(output_path, sample_rate, pcm_audio)
    logger.info(f"Mastered beat-synced audio with ducking and SFX -> {output_path} (Hits: {len(hit_beats)})")
    return output_path

def prepare_beat_synced_track(
    music_folder: Optional[str] = None,
    output_duration: float = 6.0,
    temp_dir: Optional[str] = None
) -> Tuple[BeatSyncData, str]:
    """
    Main audio entry point:
    1. Selects music from assets/music/ or synthesizes fallback.
    2. Trims track to 6.0 seconds.
    3. Analyzes beats and hit beats.
    4. Mixes ducked audio with hit-synced SFX.
    Returns (BeatSyncData, mastered_audio_path).
    """
    if music_folder is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        music_folder = os.path.join(project_root, "assets", "music")

    if temp_dir is None:
        import tempfile
        temp_dir = tempfile.gettempdir()

    audio_candidates = []
    if os.path.exists(music_folder) and os.path.isdir(music_folder):
        for fname in os.listdir(music_folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_AUDIO_EXTS:
                full_path = os.path.join(music_folder, fname)
                if os.path.getsize(full_path) > 1000:
                    audio_candidates.append(full_path)

    # 1. Real music file in assets/music/
    if audio_candidates:
        selected_file = random.choice(audio_candidates)
        logger.info(f"Selected custom audio from assets/music/: {os.path.basename(selected_file)}")
        try:
            # Trim to 6.0s
            data, sr = _read_audio(selected_file)
            target_samples = int(output_duration * sr)
            if len(data) >= target_samples:
                # Pick interesting window
                max_start = max(0, len(data) - target_samples - int(2.0 * sr))
                start_s = random.randint(0, max_start) if max_start > 0 else 0
                trimmed = data[start_s:start_s + target_samples]
            else:
                repeats = int(np.ceil(target_samples / len(data)))
                trimmed = np.tile(data, (repeats, 1) if data.ndim > 1 else repeats)[:target_samples]

            trimmed_path = os.path.join(temp_dir, f"aura_trimmed_{random.randint(1000, 9999)}.wav")
            pcm_trimmed = (np.clip(trimmed, -1.0, 1.0) * 32767.0).astype(np.int16)
            wavfile.write(trimmed_path, sr, pcm_trimmed)

            beat_data = analyze_audio_beats(trimmed_path, duration=output_duration)
            mastered_path = mix_beat_synced_audio(
                base_audio_path=trimmed_path,
                hit_beats=beat_data.hit_beats,
                duration=output_duration
            )
            return beat_data, mastered_path
        except Exception as e:
            logger.warning(f"Error trimming selected music file {selected_file}: {e}. Falling back to synthesis.")

    # 2. Synthesized fallback track
    synth_path = os.path.join(temp_dir, f"aura_synth_{random.randint(1000, 9999)}.wav")
    synthesize_phonk_beat(duration=output_duration, bpm=KNOWN_SYNTH_BPM, output_path=synth_path)
    beat_data = analyze_audio_beats(synth_path, duration=output_duration, known_bpm=KNOWN_SYNTH_BPM)
    mastered_path = mix_beat_synced_audio(
        base_audio_path=synth_path,
        hit_beats=beat_data.hit_beats,
        duration=output_duration
    )
    return beat_data, mastered_path

def get_or_synthesize_audio(
    music_folder: Optional[str] = None,
    output_duration: float = 6.0,
    temp_dir: Optional[str] = None
) -> str:
    """Backward-compatible helper returning just the mastered audio path."""
    _, mastered_path = prepare_beat_synced_track(
        music_folder=music_folder,
        output_duration=output_duration,
        temp_dir=temp_dir
    )
    return mastered_path
