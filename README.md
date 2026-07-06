# TrackIO 

**Intelligent Audio Segmentation & Analysis Tool**

TrackIO is a Python-based utility designed to slice long-form audio recordings (WAV/MP3) into individual tracks based on silence detection and dynamic volume analysis. Unlike simple silence-slicers, TrackIO normalizes high-amplitude files, analyzes dB distribution histograms, and merges adjacent segments to prevent "chopping" continuous content into hundreds of tiny files.

Perfect for archiving analog tapes, digitizing live recordings, or organizing long podcast/lecture sessions.

## Features

*   ** Intelligent Segmentation**: Automatically detects tracks by analyzing periods of silence vs. active audio.
*   ** Dynamic Normalization**: Handles WAV files with non-standard amplitude (e.g., +12dBFS peak) by normalizing signals before processing.
*   ** Deep Analytics (`-a`)**: Visualize dB distribution histograms and preview segmentation strategies without exporting files.
*   ** Smart Merging**: Merges segments separated by short gaps (breaths/pauses) to keep musical phrases intact.
*   ** Configurable Quality**: Export to WAV or high-quality MP3 with profiles for Low, Medium, High, and Ultra quality.
*   ** Mean dB Tracking**: Reports the mean RMS level of each exported track for archival metadata.

## 📋 Prerequisites

TrackIO requires Python 3.8+ and `ffmpeg` (for audio encoding).

1.  **Install FFmpeg**:
    *   **macOS**: `brew install ffmpeg`
    *   **Ubuntu/Debian**: `sudo apt install ffmpeg`
    *   **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

2.  **Install Python Dependencies**:
    ```bash
    pip install numpy librosa pydub
    ```

## Quick Start

### Basic Usage (Auto-Detect Tracks)

Slices a WAV file into MP3s, ignoring segments shorter than 60 seconds:

```bash
python trackio_local.py ~/Music/old_recording.wav \
  --format mp3 \
  --quality high \
  --prefix archive_ \
  --output-dir ./sliced_tracks \
  --min-track-dur 60000
```

### Analytics Mode (Debug & Preview)

Run analysis on a file to see the dB distribution and proposed tracks **without** creating files:

```bash
python trackio_local.py ~/Music/old_recording.wav -a
```

This will output:
1.  Raw Audio Stats (Peak/RMS).
2.  A Histogram showing where audio energy lies (e.g., mostly in -40dB to -30dB range).
3.  A Suggested Segmentation Threshold.
4.  A list of proposed tracks with their start/end times and mean dB levels.

##  Configuration Parameters

| Parameter | Short Flag | Default | Description |
| :--- | :---: | :--- | :--- |
| `--format` | `-f` | `wav` | Output format (`wav`, `mp3`). |
| `--quality` | `-q` | `medium` | Encoding quality (`low`, `medium`, `high`, `ultra`). Only affects MP3 bitrate. |
| `--prefix` | `-p` | `` | Prefix added to every output filename (e.g., `part_1.wav`). |
| `--output-dir` | `-o` | `./sliced_tracks` | Directory where output files are saved. |
| `--min-track-dur` | | `2000` | **Min Track Duration** in ms. Segments shorter than this are discarded (filters out noise/spikes). |
| `--merge-gap` | | `200` | **Merge Gap** in ms. If two audio segments are separated by less than this, they are merged into one track. |
| `--min-silence-len` | `-s` | `1500` | **Min Silence Len** in ms. Minimum silence required to consider a "break" between major tracks. |
| `--silence-thresh` | | `None` | Override the automatic dB threshold for silence detection. Lower numbers = more sensitive (e.g., `-35`). Default is dynamic based on file peak. |

##  How It Works

1.  **Normalization**: The script first normalizes the audio signal to a [-1.0, 1.0] range. This ensures that files with weird gain levels (common in digitized analog media) are processed correctly.
2.  **Windowed RMS Analysis**: Instead of relying on global averages, TrackIO calculates Root Mean Square (RMS) energy in 1-second windows.
3.  **Histogram Generation**: It builds a histogram of dB levels to understand the "loudness profile" of the file.
4.  **Threshold Selection**:
    *   If `--silence-thresh` is not provided, TrackIO automatically selects a threshold based on the most frequent audio bucket in the histogram (usually ~5dB above the average quiet level).
5.  **Segmentation & Merging**:
    *   It identifies "active" windows where volume > Threshold.
    *   It merges consecutive active windows if the silence gap between them is smaller than `--merge-gap`.
    *   It discards any resulting track shorter than `--min-track-dur`.

## Troubleshooting

### "No audio activity detected"
*   **Cause**: Your file might be extremely quiet or entirely silent.
*   **Fix**: Run with `-a` to check the Histogram. If all energy is below -60dB, try setting `--silence-thresh -50`.

### File Output is "Clipped" or Distorted
*   **Cause**: This usually happens if you are exporting to WAV but the source MP3 was low quality, or if normalization failed (rare in this version).
*   **Fix**: Ensure `ffmpeg` is correctly installed. For best results with analog recordings, use `--format wav` for lossless archiving.

### Too many small tracks
*   **Cause**: The `--min-track-dur` is too low, or `--merge-gap` is too short.
*   **Fix**: Increase `--min-track-dur` (e.g., to 10000 for 10s) and increase `--merge-gap` (e.g., to 500ms) to combine breath pauses into the main track.

## 📄 License

MIT License. Feel free to modify and distribute.
