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

### Example Analytics Output
```bash
python trackio.py ~/Music/260623-204056.WAV --min-track-dur 60000 -a
2026-07-06 14:57:41,241 - INFO - Loaded /home/shawnevans/Music/260623-204056.WAV at 48000 Hz

==================================================
AUDIO ANALYSIS REPORT
==================================================
Sample Rate: 48000 Hz
Duration: 5592.06 seconds
Raw Peak Amplitude: 4.040073871612549
Peak dBFS (Raw): 12.13 dB
RMS dBFS (Normalized): -32.33 dB

--- DB DISTRIBUTION (Relative to Peak) ---
Bucket          | Seconds     | Percentage | Bar
--------------------------------------------------
-60 to -50 dB      |   1049.00s    |    18.8%     | #############
-50 to -40 dB      |   1368.00s    |    24.5%     | #################
-40 to -30 dB      |   1587.00s    |    28.4%     | ####################
-30 to -20 dB      |   1221.50s    |    21.8%     | ###############
-20 to -10 dB      |      0.00s    |     0.0%     |
-10 to 0 dB        |      0.00s    |     0.0%     |

--- SUGGESTED SEGMENTATION STRATEGY ---
Most audio content lies in bucket: -40 to -30 dB
Suggested Threshold for Segmentation: -35.0 dBFS

--- VISUALIZATION OF SEGMENTATION ---
Active Windows: 4296/11185 (38.4% of audio detected)

--- PROPOSED TRACK LOGIC (Min Track Duration: 60000ms) ---
Using Threshold: -35.0 dBFS
Found 13 potential segments after merging AND duration filtering.
Track 1: Start=52.00s, End=161.50s, Duration=109500ms, Mean dB=-25.51 dB
Track 2: Start=429.00s, End=576.00s, Duration=147000ms, Mean dB=-27.53 dB
Track 3: Start=658.50s, End=785.00s, Duration=126500ms, Mean dB=-28.05 dB
Track 4: Start=914.00s, End=1007.50s, Duration=93500ms, Mean dB=-27.41 dB
Track 5: Start=1206.50s, End=1334.50s, Duration=128000ms, Mean dB=-26.92 dB
Track 6: Start=1431.00s, End=1547.00s, Duration=116000ms, Mean dB=-24.74 dB
Track 7: Start=1765.00s, End=1874.00s, Duration=109000ms, Mean dB=-27.49 dB
Track 8: Start=2008.00s, End=2193.00s, Duration=185000ms, Mean dB=-32.02 dB
Track 9: Start=2671.00s, End=2778.50s, Duration=107500ms, Mean dB=-30.68 dB
Track 10: Start=2966.50s, End=3041.00s, Duration=74500ms, Mean dB=-32.73 dB
Track 11: Start=3388.00s, End=3477.00s, Duration=89000ms, Mean dB=-28.34 dB
Track 12: Start=3732.00s, End=3840.00s, Duration=108000ms, Mean dB=-28.65 dB
Track 13: Start=4018.00s, End=4115.50s, Duration=97500ms, Mean dB=-25.87 dB
```

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
| `--padding` | `1000` | **Add padding to the start and end of the detected tracks to help capture things like count ins or applause. |
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
