import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment

# Fixed: Corrected logging syntax
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUALITY_PROFILES = {
    "low": {"bitrate": "32k"},
    "medium": {"bitrate": "128k"},
    "high": {"bitrate": "192k"},
    "ultra": {"bitrate": "320k"}
}

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
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def get_mean_db_from_samples(start_idx, end_idx, sr, y_normalized):
    """Calculate mean RMS dBFS of a segment in the normalized audio array."""
    # Ensure indices are within bounds
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

def analyze_audio(y, sr, min_track_dur_ms=2000):
    """
    Perform deep analysis with normalization and dB distribution.
    Honors min_track_dur_ms for the proposed track list.
    """
    print("\n" + "="*50)
    print("AUDIO ANALYSIS REPORT")
    print("="*50)

    # 1. Normalize for Analysis
    y_norm = normalize_audio(y)

    # 2. Calculate Stats
    rms_norm = np.sqrt(np.mean(y_norm**2))
    peak_dbfs_raw = 20 * np.log10(max(np.abs(y))) if max(np.abs(y)) > 0 else -np.inf
    rms_dbfs_norm = 20 * np.log10(rms_norm) if rms_norm > 0 else -np.inf

    print(f"Sample Rate: {sr} Hz")
    print(f"Duration: {len(y)/sr:.2f} seconds")
    print(f"Raw Peak Amplitude: {max(np.abs(y))}")
    print(f"Peak dBFS (Raw): {peak_dbfs_raw:.2f} dB")
    print(f"RMS dBFS (Normalized): {rms_dbfs_norm:.2f} dB")

    # 3. DB Distribution Histogram
    print("\n--- DB DISTRIBUTION (Relative to Peak) ---")
    print("Bucket          | Seconds     | Percentage | Bar")
    print("-" * 50)

    total_seconds = len(y) / sr
    buckets = [(-60, -50), (-50, -40), (-40, -30), (-30, -20), (-20, -10), (-10, 0)]
    bucket_labels = ["-60 to -50 dB", "-50 to -40 dB", "-40 to -30 dB", "-30 to -20 dB", "-20 to -10 dB", "-10 to 0 dB"]

    # Use 1-second windows for histogram granularity
    window_size = int(sr)
    if len(y) < window_size:
        window_size = len(y)

    hop_length = window_size // 2
    rms_windows = librosa.feature.rms(y=y_norm, frame_length=window_size, hop_length=hop_length)[0]

    # Convert to dBFS
    db_windows = np.where(rms_windows > 1e-10, 20 * np.log10(rms_windows), -np.inf)

    bucket_counts = [0] * len(buckets)
    sec_per_window = hop_length / sr

    for db_val in db_windows:
        if np.isneginf(db_val):
            continue

        for i, (low, high) in enumerate(buckets):
            if low <= db_val < high:
                bucket_counts[i] += 1
                break

    max_count = max(bucket_counts) if any(bucket_counts) else 1

    for i in range(len(buckets)):
        seconds = bucket_counts[i] * sec_per_window
        percentage = (seconds / total_seconds) * 100
        bar_len = int((bucket_counts[i] / max_count) * 20) if max_count > 0 else 0
        bar = "#" * bar_len

        print(f"{bucket_labels[i]:18s} | {seconds:9.2f}s    | {percentage:7.1f}%     | {bar}")

    # 4. Determine Best Threshold for Segmentation
    print("\n--- SUGGESTED SEGMENTATION STRATEGY ---")

    active_buckets = [(i, count) for i, count in enumerate(bucket_counts) if count > 0]
    best_threshold = -20

    if active_buckets:
        i_max, _ = max(active_buckets, key=lambda x: x[1])
        low_bound, high_bound = buckets[i_max]
        suggested_threshold = (low_bound + high_bound) / 2

        if suggested_threshold < -45:
             suggested_threshold = -40

        best_threshold = suggested_threshold
        print(f"Most audio content lies in bucket: {bucket_labels[i_max]}")
        print(f"Suggested Threshold for Segmentation: {best_threshold} dBFS")
    else:
        print("File appears to be mostly silence.")

    # 5. Visualization (Simplified)
    print("\n--- VISUALIZATION OF SEGMENTATION ---")
    thresh_linear = 10 ** (best_threshold / 20)
    rms_windows_vis = librosa.feature.rms(y=y_norm, frame_length=window_size, hop_length=hop_length)[0]
    is_active = rms_windows_vis >= thresh_linear

    # Count active windows for a simple stat instead of drawing text
    active_windows = np.sum(is_active)
    total_windows = len(is_active)
    print(f"Active Windows: {active_windows}/{total_windows} ({(active_windows/total_windows)*100:.1f}% of audio detected)")

    # 6. Proposed Tracks Logic using Manual RMS Check
    print("\n--- PROPOSED TRACK LOGIC (Min Track Duration: {}ms) ---".format(min_track_dur_ms))
    print(f"Using Threshold: {best_threshold} dBFS")

    # Find contiguous segments of active frames
    active_indices = np.where(is_active)[0]

    if len(active_indices) == 0:
        print("No tracks found even with suggested threshold.")
        return []

    intervals = []
    start_frame = active_indices[0]
    end_frame = start_frame

    for i in range(1, len(active_indices)):
        curr_frame = active_indices[i]
        if curr_frame - end_frame > 2: # Gap detected between active frames
            intervals.append((start_frame * hop_length, end_frame * hop_length))
            start_frame = curr_frame
        end_frame = curr_frame

    intervals.append((start_frame * hop_length, end_frame * hop_length))

    # Merge closely spaced segments
    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        prev_end = merged_intervals[-1][1]
        curr_start = intervals[i][0]

        gap_samples = curr_start - prev_end
        if sr > 0:
            gap_ms = (gap_samples / sr) * 1000
        else:
            gap_ms = 0

        if gap_ms < 200: # Merge gap
            merged_intervals[-1] = (merged_intervals[-1][0], intervals[i][1])
        else:
            merged_intervals.append(intervals[i])

    # Filter by duration BEFORE printing
    valid_tracks_for_print = []
    for start_s, end_s in merged_intervals:
        track_dur_ms = int(((end_s - start_s) / sr) * 1000) if sr > 0 else 0

        if track_dur_ms >= min_track_dur_ms:
            valid_tracks_for_print.append((start_s, end_s))

    print(f"Found {len(valid_tracks_for_print)} potential segments after merging AND duration filtering.")

    # Calculate and print mean dB for these tracks
    for i, (s, e) in enumerate(valid_tracks_for_print):
        dur_ms = int((e - s) / sr * 1000)
        # Use the normalized array passed to this function scope if available,
        # but here we need y_norm which is in local scope.
        mean_db = get_mean_db_from_samples(s, e, sr, y_norm)

        if np.isneginf(mean_db):
            db_str = "Silence"
        else:
            db_str = f"{mean_db:.2f} dB"

        print(f"Track {i+1}: Start={s/sr:.2f}s, End={e/sr:.2f}s, Duration={dur_ms}ms, Mean dB={db_str}")

    return valid_tracks_for_print

def detect_tracks_intelligent(y, sr,
                              min_silence_len_ms=1500,
                              merge_gap_ms=200,
                              min_track_dur_ms=2000,
                              custom_top_db=None):

    # 1. Normalize for Analysis
    y_norm = normalize_audio(y)

    if custom_top_db is not None:
        top_db = float(custom_top_db)
    else:
        # Dynamic Thresholding based on RMS
        rms_norm = np.sqrt(np.mean(y_norm**2))
        rms_dbfs = 20 * np.log10(rms_norm) if rms_norm > 0 else -np.inf

        # If RMS is very low, use a lower threshold
        if rms_dbfs < -35:
            top_db = rms_dbfs + 5
        else:
            top_db = -20.0

    logger.debug(f"Using silence detection threshold: {top_db:.2f} dBFS")

    # Use manual RMS check for robustness
    thresh_linear = 10 ** (top_db / 20)

    # Use a window size of 1 second for stability
    window_size = int(sr)
    hop_length = window_size // 2

    rms_windows = librosa.feature.rms(y=y_norm, frame_length=window_size, hop_length=hop_length)[0]
    is_active = rms_windows >= thresh_linear

    active_indices = np.where(is_active)[0]

    if len(active_indices) == 0:
        logger.warning("No audio activity detected even with dynamic threshold.")
        return []

    # Find contiguous segments of active frames
    intervals = []
    start_frame = active_indices[0]
    end_frame = start_frame

    for i in range(1, len(active_indices)):
        curr_frame = active_indices[i]
        if curr_frame - end_frame > 2:
            intervals.append((start_frame * hop_length, end_frame * hop_length))
            start_frame = curr_frame
        end_frame = curr_frame

    intervals.append((start_frame * hop_length, end_frame * hop_length))

    # Merge closely spaced segments
    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        prev_end = merged_intervals[-1][1]
        curr_start = intervals[i][0]

        gap_samples = curr_start - prev_end
        if sr > 0:
            gap_ms = (gap_samples / sr) * 1000
        else:
            gap_ms = 0

        if gap_ms < merge_gap_ms:
            merged_intervals[-1] = (merged_intervals[-1][0], intervals[i][1])
        else:
            merged_intervals.append(intervals[i])

    # Filter by duration and apply padding
    final_tracks = []

    for start_s, end_s in merged_intervals:
        track_dur_ms = int(((end_s - start_s) / sr) * 1000) if sr > 0 else 0

        if track_dur_ms < min_track_dur_ms:
            logger.debug(f"Discarded short segment: {start_s} - {end_s} ({track_dur_ms}ms)")
            continue

        pad_samples = int(0.1 * sr) # 100ms padding

        final_start = max(0, start_s - pad_samples)
        final_end = min(len(y), end_s + pad_samples)

        final_tracks.append((final_start, final_end))

    logger.info(f"Identified {len(final_tracks)} intelligently merged tracks.")
    return final_tracks

def export_audio_chunk(audio_segment, start_ms, end_ms, output_path, quality_profile="medium", format="wav", y_norm=None, sr=48000):
    """
    Export a slice of the audio segment to a file.
    Also calculates and logs mean dB if y_norm is provided.
    """
    start_ms = max(0, int(start_ms))
    end_ms = min(len(audio_segment), int(end_ms))

    if end_ms <= start_ms:
        return False

    chunk = audio_segment[start_ms:end_ms]

    try:
        # Calculate mean dB from normalized samples if available
        # Convert ms to sample indices for y_norm
        start_s = int((start_ms / 1000) * sr)
        end_s = int((end_ms / 1000) * sr)

        mean_db = "N/A"
        if y_norm is not None and sr > 0:
            db_val = get_mean_db_from_samples(start_s, end_s, sr, y_norm)
            if not np.isneginf(db_val):
                mean_db = f"{db_val:.2f} dB"
            else:
                mean_db = "Silence"

        chunk.export(output_path, format=format.lower(), bitrate=quality_profile['bitrate'])

        # Log the mean dB
        logger.info(f"Mean dB for {os.path.basename(output_path)}: {mean_db}")

        return True
    except Exception as e:
        logger.error(f"Failed to export {output_path}: {e}")
        return False

def process_audio(file_path, output_dir, prefix, quality="medium", format="wav",
                  min_silence_len=1500, merge_gap_ms=200, min_track_dur_ms=2000, silence_thresh=None):

    logger.info(f"Loading audio for analysis: {file_path}")
    y, sr = load_audio(file_path)

    # Normalize once for all calculations
    y_norm = normalize_audio(y)

    track_ranges = detect_tracks_intelligent(
        y,
        sr,
        min_silence_len_ms=min_silence_len,
        merge_gap_ms=merge_gap_ms,
        min_track_dur_ms=min_track_dur_ms,
        custom_top_db=silence_thresh
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
        filename = f"{prefix}{base_name}_track_{i+1:03d}.{ext}"
        output_path = os.path.join(output_dir, filename)

        logger.info(f"Exporting Track {i+1}/{len(track_ranges)}: {filename} ({end_ms - start_ms}ms)")

        if export_audio_chunk(audio, start_ms, end_ms, output_path, q_params, format, y_norm, sr):
            processed_count += 1

    logger.info(f"Finished. Exported {processed_count} tracks.")

def main():
    parser = argparse.ArgumentParser(description="Intelligent Audio Slicer with Analytics")
    parser.add_argument("file", help="Path to input WAV/MP3")

    # Mode Selection
    parser.add_argument("--analyze", "-a", action="store_true", help="Run analysis mode only, do not export files")

    # Output Parameters
    parser.add_argument("--output-dir", "-o", default="./sliced_tracks", help="Output directory")
    parser.add_argument("--prefix", "-p", default="", help="Filename prefix")
    parser.add_argument("--quality", "-q", choices=["low", "medium", "high", "ultra"], default="medium")
    parser.add_argument("--format", "-f", choices=["wav", "mp3"], default="wav")

    # Slicing Parameters
    parser.add_argument("--min-silence-len", type=int, default=1500, help="Min silence in ms to separate major tracks")
    parser.add_argument("--merge-gap", type=int, default=200, help="Max gap in ms between segments to merge them")
    parser.add_argument("--min-track-dur", type=int, default=2000, help="Min duration in ms for a valid track")

    # Threshold Parameter
    parser.add_argument("--silence-thresh", type=float, default=None,
                        help="Override automatic threshold. Lower is more sensitive (e.g., -35).")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        logger.error(f"File {args.file} does not exist.")
        sys.exit(1)

    # Load audio once for analysis
    y, sr = load_audio(args.file)

    if args.analyze:
        analyze_audio(y, sr, min_track_dur_ms=args.min_track_dur)
    else:
        process_audio(
            file_path=args.file,
            output_dir=args.output_dir,
            prefix=args.prefix,
            quality=args.quality,
            format=args.format,
            min_silence_len=args.min_silence_len,
            merge_gap_ms=args.merge_gap,
            min_track_dur_ms=args.min_track_dur,
            silence_thresh=args.silence_thresh
        )

if __name__ == "__main__":
    main()
