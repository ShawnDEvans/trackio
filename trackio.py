#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, fftconvolve
from pydub import AudioSegment
import pyloudnorm as pyln

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUALITY_PROFILES = {
    "low": {"bitrate": "32k"},
    "medium": {"bitrate": "128k"},
    "high": {"bitrate": "192k"},
    "ultra": {"bitrate": "320k"}
}

def apply_filter(data, cutoff, fs, btype='high', order=2):
    """Apply a zero-phase Butterworth filter to prevent phase cancellation."""
    sos = butter(order, cutoff, fs=fs, btype=btype, output='sos')
    min_len = 3 * (order * 2)
    if data.shape[-1] <= min_len:
        return data
    return sosfiltfilt(sos, data, axis=-1)

def apply_dynamic_eq(audio, sr, cutoff=120, threshold_percentile=85):
    """Compress sub-bass dynamically only when low-end energy exceeds normal thresholds."""
    sub = apply_filter(audio, cutoff, sr, btype='low')
    sub_env = apply_filter(np.abs(sub), 20, sr, btype='low')
    max_sub = np.percentile(sub_env, threshold_percentile)

    # Calculate compression attenuation mask
    compression = np.minimum(1.0, max_sub / (sub_env + 1e-6))
    return (audio - sub) + (sub * compression)

def apply_deesser(audio, sr, deess_factor=1.0):
    """Ducks harsh 6kHz-9kHz sibilance spikes dynamically."""
    if deess_factor <= 0:
        return audio

    # Isolate sibilant frequency band
    sibilance = apply_filter(apply_filter(audio, 6000, sr, btype='high'), 9000, sr, btype='low')
    sib_env = apply_filter(np.abs(sibilance), 30, sr, btype='low')

    thresh = np.mean(sib_env) * 2.2 / max(deess_factor, 0.1)
    reduction = np.clip(1.0 - (sib_env - thresh) / (thresh + 1e-6), 0.35, 1.0)
    return audio * reduction

def apply_transient_shaper(audio, sr, transient_boost=1.15):
    """Sharpens initial attack transients (drums/picks) before saturation."""
    if transient_boost == 1.0:
        return audio

    env = apply_filter(np.abs(audio), 50, sr, btype='low')
    diff = np.diff(env, axis=-1, prepend=0)
    attack_spikes = np.maximum(0, diff)
    return audio + (attack_spikes * (transient_boost - 1.0))

def generate_spatial_reverb_ir(sr, room_size_sec=0.6, decay_rate=6.0):
    """Generates a filtered stereo impulse response for realistic room ambience."""
    ir_len = int(sr * room_size_sec)
    t = np.linspace(0, room_size_sec, ir_len, endpoint=False)

    ir_left = np.random.normal(0, 1, ir_len) * np.exp(-decay_rate * t)
    ir_right = np.random.normal(0, 1, ir_len) * np.exp(-decay_rate * t)

    ir_left = apply_filter(apply_filter(ir_left, 300, sr, 'high'), 6000, sr, 'low')
    ir_right = apply_filter(apply_filter(ir_right, 300, sr, 'high'), 6000, sr, 'low')

    max_peak = max(np.max(np.abs(ir_left)), np.max(np.abs(ir_right)), 1e-8)
    return ir_left / max_peak, ir_right / max_peak

def apply_lookahead_limiter(audio, sr, target_norm_dbfs=-0.3, lookahead_ms=3.0):
    """Look-ahead true-peak brickwall limiter to prevent inter-sample clipping."""
    ceiling = 10 ** (target_norm_dbfs / 20.0)
    lookahead_samples = int((lookahead_ms / 1000.0) * sr)

    padded_audio = np.pad(audio, ((0, 0), (0, lookahead_samples)), mode='constant')
    delayed_audio = np.pad(audio, ((0, 0), (lookahead_samples, 0)), mode='constant')

    # Calculate peak envelope across channels
    max_env = np.max(np.abs(padded_audio), axis=0)

    # Sliding max over look-ahead window
    sliding_peaks = np.maximum.reduceat(
        max_env, np.arange(0, len(max_env), 1)
    ) if len(max_env) < 50000 else max_env  # Efficient fallback for large arrays

    gain_reduction = np.minimum(1.0, ceiling / (sliding_peaks + 1e-6))
    gain_reduction = apply_filter(gain_reduction, 200, sr, btype='low')

    limited_audio = delayed_audio[:, :audio.shape[1]] * gain_reduction[:audio.shape[1]]
    return np.clip(limited_audio, -ceiling, ceiling)

def master_audio_segment(
    sound,
    stereo_width=1.4,
    bass_boost=1.05,
    mid_boost=1.12,
    treble_boost=1.18,
    drive=1.18,
    reverb_mix=0.06,
    target_lufs=-12.0,
    target_norm_dbfs=-0.3,
    deess_factor=1.0,
    transient_boost=1.15
):
    """Adaptive mastering suite integrating dynamic EQ, de-essing, and LUFS targeting."""
    sr = sound.frame_rate
    max_sample_val = float(1 << (sound.sample_width * 8 - 1))

    samples = np.array(sound.get_array_of_samples(), dtype=np.float32)
    if sound.channels == 2:
        samples = samples.reshape((-1, 2)).T
    else:
        samples = np.vstack([samples, samples])

    samples /= max_sample_val
    left, right = samples[0], samples[1]

    # 1. Zero-Phase Mid/Side Processing
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    side_high = apply_filter(side, 120, sr, btype='high')
    side_low = side - side_high
    side = side_low + (side_high * stereo_width)

    left = mid + side
    right = mid - side

    # 2. Spectral EQ Enhancement & Tonal Safety Guardrails
    left = apply_filter(left, 28, sr, btype='high')
    right = apply_filter(right, 28, sr, btype='high')

    if bass_boost != 1.0:
        bass_L = apply_filter(left, 150, sr, btype='low')
        bass_R = apply_filter(right, 150, sr, btype='low')
        left += bass_L * (bass_boost - 1.0)
        right += bass_R * (bass_boost - 1.0)

    if mid_boost != 1.0:
        mid_L = apply_filter(apply_filter(left, 300, sr, btype='high'), 4000, sr, btype='low')
        mid_R = apply_filter(apply_filter(right, 300, sr, btype='high'), 4000, sr, btype='low')
        left += mid_L * (mid_boost - 1.0)
        right += mid_R * (mid_boost - 1.0)

    if treble_boost != 1.0:
        treble_L = apply_filter(left, 8000, sr, btype='high')
        treble_R = apply_filter(right, 8000, sr, btype='high')
        left += treble_L * (treble_boost - 1.0)
        right += treble_R * (treble_boost - 1.0)

    audio = np.vstack([left, right])

    # 3. Dynamic Low-End Control & De-Essing
    audio = apply_dynamic_eq(audio, sr, cutoff=120)
    audio = apply_deesser(audio, sr, deess_factor=deess_factor)

    # 4. Transient Shaping & Harmonic Saturation
    audio = apply_transient_shaper(audio, sr, transient_boost=transient_boost)
    if drive > 1.0:
        audio *= drive
        audio = np.tanh(audio)

    # 5. Spatial Convolution Reverb
    if reverb_mix > 0.0:
        ir_L, ir_R = generate_spatial_reverb_ir(sr)
        wet_L = fftconvolve(audio[0], ir_L, mode='full')[:len(audio[0])]
        wet_R = fftconvolve(audio[1], ir_R, mode='full')[:len(audio[1])]

        audio[0] += wet_L * reverb_mix
        audio[1] += wet_R * reverb_mix

    # 6. Integrated ITU-R BS.1770 LUFS Normalization
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.T)
        if not np.isinf(loudness):
            audio_normalized = pyln.normalize.loudness(audio.T, loudness, target_lufs)
            audio = audio_normalized.T
    except Exception as e:
        logger.warning(f"LUFS calculation skipped: {e}")

    # 7. Look-Ahead True-Peak Limiter
    audio = apply_lookahead_limiter(audio, sr, target_norm_dbfs=target_norm_dbfs)

    # Reconstruct into 16-bit PCM AudioSegment
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16).T.flatten()
    return AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=2
    )

def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        logger.info(f"Loaded {file_path} at {sr} Hz")
        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        sys.exit(1)

def normalize_audio(y):
    """Normalize audio to [-1.0, 1.0] range for analysis."""
    max_val = np.max(np.abs(y)) if len(y) > 0 else 0
    if max_val > 0:
        return y / max_val
    return y

def get_mean_db_from_samples(start_idx, end_idx, sr, y_normalized):
    """Calculate mean RMS dBFS of a segment in the normalized audio array."""
    start_idx = max(0, int(start_idx))
    end_idx = min(len(y_normalized), int(end_idx))

    if start_idx >= end_idx:
        return -np.inf

    segment = y_normalized[start_idx:end_idx]
    if len(segment) == 0:
        return -np.inf

    rms = np.sqrt(np.mean(segment**2))
    if rms > 0:
        return 20 * np.log10(rms)
    return -np.inf

def detect_tracks_intelligent(y, sr,
                              min_silence_len_ms=1500,
                              merge_gap_ms=500,
                              min_track_dur_ms=2000,
                              padding_ms=1000,
                              custom_top_db=None,
                              y_norm=None):
    if y_norm is None:
        y_norm = normalize_audio(y)

    window_size = int(sr)
    if len(y_norm) < window_size:
        window_size = len(y_norm)
    hop_length = window_size // 2

    rms_windows = librosa.feature.rms(y=y_norm, frame_length=window_size, hop_length=hop_length)[0]

    if custom_top_db is not None:
        top_db = float(custom_top_db)
    else:
        db_windows = np.where(rms_windows > 1e-10, 20 * np.log10(rms_windows), -np.inf)
        bucket_counts, _ = np.histogram(db_windows, bins=[-60, -50, -40, -30, -20, -10, 0])
        active_buckets = [(i, count) for i, count in enumerate(bucket_counts) if count > 0]
        if active_buckets:
            i_max, _ = max(active_buckets, key=lambda x: x[1])
            buckets = [(-60, -50), (-50, -40), (-40, -30), (-30, -20), (-20, -10), (-10, 0)]
            suggested_threshold = (buckets[i_max][0] + buckets[i_max][1]) / 2
            if suggested_threshold < -45:
                suggested_threshold = -40.0
            top_db = suggested_threshold
        else:
            top_db = -20.0

    thresh_linear = 10 ** (top_db / 20)
    is_active = rms_windows >= thresh_linear
    active_indices = np.where(is_active)[0]

    if len(active_indices) == 0:
        return [], top_db, rms_windows, hop_length

    split_indices = np.where(np.diff(active_indices) > 2)[0] + 1
    groups = np.split(active_indices, split_indices)
    intervals = [(g[0] * hop_length, g[-1] * hop_length) for g in groups if len(g) > 0]

    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        prev_end = merged_intervals[-1][1]
        curr_start = intervals[i][0]
        gap_ms = ((curr_start - prev_end) / sr) * 1000 if sr > 0 else 0

        if gap_ms < merge_gap_ms:
            merged_intervals[-1] = (merged_intervals[-1][0], intervals[i][1])
        else:
            merged_intervals.append(intervals[i])

    final_tracks = []
    pad_samples = int((padding_ms / 1000) * sr)

    for start_s, end_s in merged_intervals:
        track_dur_ms = int(((end_s - start_s) / sr) * 1000) if sr > 0 else 0
        if track_dur_ms < min_track_dur_ms:
            continue

        final_start = max(0, start_s - pad_samples)
        final_end = min(len(y), end_s + pad_samples)
        final_tracks.append((final_start, final_end))

    return final_tracks, top_db, rms_windows, hop_length

def analyze_audio(y, sr, min_track_dur_ms=2000, merge_gap_ms=500, padding_ms=1000, silence_thresh=None):
    print("\n" + "="*50)
    print("AUDIO ANALYSIS REPORT")
    print("="*50)

    y_norm = normalize_audio(y)
    rms_norm = np.sqrt(np.mean(y_norm**2))
    max_abs = np.max(np.abs(y)) if len(y) > 0 else 0
    peak_dbfs_raw = 20 * np.log10(max_abs) if max_abs > 0 else -np.inf
    rms_dbfs_norm = 20 * np.log10(rms_norm) if rms_norm > 0 else -np.inf

    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    print(f"Raw Peak Amplitude: {max_abs}")
    print(f"Peak dBFS (Raw): {peak_dbfs_raw:.2f} dB")
    print(f"RMS dBFS (Normalized): {rms_dbfs_norm:.2f} dB")

    print("\n--- DB DISTRIBUTION (Relative to Peak) ---")
    print("Bucket             | Seconds     | Percentage | Bar")
    print("-" * 50)

    total_seconds = len(y) / sr

    valid_tracks, calculated_thresh, rms_windows, hop_length = detect_tracks_intelligent(
        y, sr, merge_gap_ms=merge_gap_ms, min_track_dur_ms=min_track_dur_ms,
        padding_ms=padding_ms, custom_top_db=silence_thresh, y_norm=y_norm
    )

    db_windows = np.where(rms_windows > 1e-10, 20 * np.log10(rms_windows), -np.inf)
    buckets = [(-60, -50), (-50, -40), (-40, -30), (-30, -20), (-20, -10), (-10, 0)]
    bucket_labels = ["-60 to -50 dB", "-50 to -40 dB", "-40 to -30 dB", "-30 to -20 dB", "-20 to -10 dB", "-10 to 0 dB"]

    bucket_counts, _ = np.histogram(db_windows, bins=[-60, -50, -40, -30, -20, -10, 0])
    max_count = max(bucket_counts) if any(bucket_counts) else 1
    sec_per_window = hop_length / sr

    for i in range(len(buckets)):
        seconds = bucket_counts[i] * sec_per_window
        percentage = (seconds / total_seconds) * 100 if total_seconds > 0 else 0
        bar = "#" * (int((bucket_counts[i] / max_count) * 20) if max_count > 0 else 0)
        print(f"{bucket_labels[i]:18s} | {seconds:9.2f}s    | {percentage:7.1f}%     | {bar}")

    print("\n--- SUGGESTED SEGMENTATION STRATEGY ---")
    active_buckets = [(i, count) for i, count in enumerate(bucket_counts) if count > 0]
    if active_buckets:
        i_max, _ = max(active_buckets, key=lambda x: x[1])
        print(f"Most audio content lies in bucket: {bucket_labels[i_max]}")
    else:
        print("File appears to be mostly silence.")

    if silence_thresh is None:
        suggested_thresh = calculated_thresh
    else:
        if active_buckets:
            i_max, _ = max(active_buckets, key=lambda x: x[1])
            suggested_thresh = (buckets[i_max][0] + buckets[i_max][1]) / 2
            if suggested_thresh < -45:
                suggested_thresh = -40.0
        else:
            suggested_thresh = -20.0

    print(f"Suggested Threshold for Segmentation: {suggested_thresh:.1f} dBFS")

    print("\n--- VISUALIZATION OF SEGMENTATION ---")
    thresh_linear = 10 ** (calculated_thresh / 20)
    is_active = rms_windows >= thresh_linear
    active_windows = np.sum(is_active)
    total_windows = len(is_active)
    pct_detected = (active_windows / total_windows) * 100 if total_windows > 0 else 0
    print(f"Active Windows: {active_windows}/{total_windows} ({pct_detected:.1f}% of audio detected)")

    print(f"\n--- PROPOSED TRACK LOGIC (Min Track Duration: {min_track_dur_ms}ms, Padding: {padding_ms}ms) ---")
    print(f"Using Threshold: {calculated_thresh:.1f} dBFS")
    print(f"Found {len(valid_tracks)} potential segments after merging AND duration filtering.")

    for i, (s, e) in enumerate(valid_tracks):
        dur_ms = int((e - s) / sr * 1000) if sr > 0 else 0
        mean_db = get_mean_db_from_samples(s, e, sr, y_norm)
        db_str = "Silence" if np.isneginf(mean_db) else f"{mean_db:.2f} dB"
        print(f"Track {i+1}: Start={s/sr:.2f}s, End={e/sr:.2f}s, Duration={dur_ms}ms, Mean dB={db_str}")

    return valid_tracks

def export_audio_chunk(
    audio_segment,
    start_ms,
    end_ms,
    output_path,
    quality_profile="medium",
    format="wav",
    y_norm=None,
    sr=48000,
    auto_master=False,
    master_params=None
):
    start_ms = max(0, int(start_ms))
    end_ms = min(len(audio_segment), int(end_ms))

    if end_ms <= start_ms:
        return False

    chunk = audio_segment[start_ms:end_ms]

    if auto_master:
        logger.info(f"Applying adaptive mastering suite to {os.path.basename(output_path)}...")
        master_params = master_params or {}
        chunk = master_audio_segment(chunk, **master_params)

    try:
        start_s = int((start_ms / 1000) * sr)
        end_s = int((end_ms / 1000) * sr)

        mean_db = "N/A"
        if y_norm is not None and sr > 0:
            db_val = get_mean_db_from_samples(start_s, end_s, sr, y_norm)
            mean_db = "Silence" if np.isneginf(db_val) else f"{db_val:.2f} dB"

        chunk.export(output_path, format=format.lower(), bitrate=quality_profile['bitrate'])
        logger.info(f"Exported {os.path.basename(output_path)} (Mean dB: {mean_db})")
        return True
    except Exception as e:
        logger.error(f"Failed to export {output_path}: {e}")
        return False

def process_audio(
    y,
    sr,
    file_path,
    output_dir,
    prefix,
    quality="medium",
    format="wav",
    min_silence_len=1500,
    merge_gap_ms=500,
    min_track_dur_ms=2000,
    padding_ms=1000,
    silence_thresh=None,
    auto_master=False,
    master_params=None,
    disable_slicing=False
):
    y_norm = normalize_audio(y)

    if disable_slicing:
        logger.info("Slicing disabled: Processing full input as a single track.")
        track_ranges = [(0, len(y))]
    else:
        track_ranges, _, _, _ = detect_tracks_intelligent(
            y, sr,
            min_silence_len_ms=min_silence_len,
            merge_gap_ms=merge_gap_ms,
            min_track_dur_ms=min_track_dur_ms,
            padding_ms=padding_ms,
            custom_top_db=silence_thresh,
            y_norm=y_norm
        )

    if not track_ranges:
        logger.warning("No tracks found.")
        return

    try:
        audio = AudioSegment.from_file(file_path)
    except Exception as e:
        logger.error(f"Error loading file with pydub: {e}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if quality not in QUALITY_PROFILES:
        logger.error(f"Invalid quality profile: {quality}.")
        sys.exit(1)

    q_params = QUALITY_PROFILES[quality]
    processed_count = 0

    for i, (start_sample, end_sample) in enumerate(track_ranges):
        start_ms = int((start_sample / sr) * 1000) if sr > 0 else 0
        end_ms = int((end_sample / sr) * 1000) if sr > 0 else 0

        base_name = Path(file_path).stem
        ext = format.lower()

        if disable_slicing:
            suffix = "_mastered" if auto_master else "_output"
            filename = f"{prefix}{base_name}{suffix}.{ext}"
        else:
            filename = f"{prefix}{base_name}_track_{i+1:03d}.{ext}"

        output_path = os.path.join(output_dir, filename)

        if disable_slicing:
            logger.info(f"Processing single track: {filename} ({end_ms - start_ms}ms)")
        else:
            logger.info(f"Processing Track {i+1}/{len(track_ranges)}: {filename} ({end_ms - start_ms}ms)")

        if export_audio_chunk(
            audio, start_ms, end_ms, output_path, q_params, format, y_norm, sr,
            auto_master=auto_master, master_params=master_params
        ):
            processed_count += 1

    logger.info(f"Finished. Exported {processed_count} track(s).")

def main():
    parser = argparse.ArgumentParser(description="TrackIO: Intelligent Audio Slicer & Mastering Suite")
    parser.add_argument("file", help="Path to input WAV/MP3")
    parser.add_argument("--analyze", "-a", action="store_true", help="Run analysis mode only, do not export files")
    parser.add_argument("--disable-slicing", action="store_true", help="Bypass track slicing and process/master input as a single file")
    parser.add_argument("--output-dir", "-o", default="./sliced_tracks", help="Output directory")
    parser.add_argument("--prefix", "-p", default="", help="Filename prefix")
    parser.add_argument("--quality", "-q", choices=["low", "medium", "high", "ultra"], default="medium")
    parser.add_argument("--format", "-f", choices=["wav", "mp3"], default="wav")
    parser.add_argument("--min-silence-len", type=int, default=2000, help="Min silence in ms to separate major tracks")
    parser.add_argument("--merge-gap", type=int, default=1000, help="Max gap in ms between segments to merge them")
    parser.add_argument("--min-track-dur", type=int, default=30000, help="Min duration in ms for a valid track")
    parser.add_argument("--padding", type=int, default=3000, help="Buffer padding in ms added to both ends of a track")
    parser.add_argument("--silence-thresh", type=float, default=-35, help="Override automatic threshold. Lower is more sensitive.")

    # Mastering Options
    parser.add_argument("--auto-master", action="store_true", help="Enable adaptive audio mastering on output tracks")
    parser.add_argument("--width", type=float, default=1.4, help="Mastering: Stereo width expansion factor")
    parser.add_argument("--bass", type=float, default=1.16, help="Mastering: Bass gain factor for <150Hz")
    parser.add_argument("--mid", type=float, default=1.13, help="Mastering: Mid gain factor for 300Hz-4kHz")
    parser.add_argument("--treble", type=float, default=1.18, help="Mastering: Treble gain factor for >8kHz")
    parser.add_argument("--drive", type=float, default=1.22, help="Mastering: Analog harmonic saturation drive factor")
    parser.add_argument("--reverb", type=float, default=0.01, help="Mastering: Spatial reverb wet mix ratio")
    parser.add_argument("--lufs", type=float, default=-12.0, help="Mastering: Integrated ITU-R BS.1770 LUFS loudness target (default: -12.0 LUFS)")
    parser.add_argument("--norm", type=float, default=-0.3, help="Mastering: Peak normalization brickwall ceiling in dBFS")
    parser.add_argument("--deess", type=float, default=1.0, help="Mastering: Dynamic de-essing sensitivity factor (default: 1.0)")
    parser.add_argument("--transients", type=float, default=1.15, help="Mastering: Pre-saturation attack transient shaper factor (default: 1.15)")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        logger.error(f"File {args.file} does not exist.")
        sys.exit(1)

    y, sr = load_audio(args.file)

    if args.analyze:
        analyze_audio(
            y, sr, min_track_dur_ms=args.min_track_dur,
            merge_gap_ms=args.merge_gap, padding_ms=args.padding,
            silence_thresh=args.silence_thresh
        )
    else:
        master_params = {
            "stereo_width": args.width,
            "bass_boost": args.bass,
            "mid_boost": args.mid,
            "treble_boost": args.treble,
            "drive": args.drive,
            "reverb_mix": args.reverb,
            "target_lufs": args.lufs,
            "target_norm_dbfs": args.norm,
            "deess_factor": args.deess,
            "transient_boost": args.transients
        }
        process_audio(
            y=y, sr=sr, file_path=args.file, output_dir=args.output_dir, prefix=args.prefix,
            quality=args.quality, format=args.format, min_silence_len=args.min_silence_len,
            merge_gap_ms=args.merge_gap, min_track_dur_ms=args.min_track_dur, padding_ms=args.padding,
            silence_thresh=args.silence_thresh, auto_master=args.auto_master, master_params=master_params,
            disable_slicing=args.disable_slicing
        )

if __name__ == "__main__":
    main()
