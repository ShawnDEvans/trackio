# TrackIO

**Intelligent Audio Segmentation, Analysis & Adaptive Mastering Suite**

TrackIO is a Python-based audio processing utility designed to analyze long-form recordings (WAV/MP3), intelligently segment them into individual tracks based on dynamic silence detection, and apply studio-grade adaptive mastering.


TrackIO goes beyond simple silence slicing: it normalizes amplitude anomalies, generates dB distribution histograms, merges continuous audio passages, and features a multi-stage DSP mastering pipeline capable of turning raw mixes into commercial-ready masters.

Whether digitizing vinyl and analog tapes, archiving live DJ sets and podcasts, or mastering single tracks, TrackIO provides end-to-end automated processing.

**AI CODE WARNING**

I'm lazy, in a couple of bands, and not an audio engineer. We use a Zoom recording device to capture jam sessions. This results in a single giant *.WAV file. I had no appetite for manually slicing the source audio into tracks, so I created this tool to save time and make sharing practices easier. It gets the job done! I hope you find it useful despite the use of AI code generation. 

---

## Key Features

* **Intelligent Segmentation**: Automatically splits continuous audio recordings into tracks by analyzing relative silence versus active audio regions.
* **Standalone Mastering Mode (`--disable-slicing`)**: Process and master single audio files without triggering segmentation.
* **Adaptive DSP Mastering (`--auto-master`)**:
  * **Zero-Phase Mid/Side Processing**: Widens soundstage while keeping sub-bass (<120Hz) locked to mono.
  * **Spectral & Dynamic Equalization**: Applies targeted low, mid, and high frequency shaping with dynamic sub-bass attenuation.
  * **Tonal Safety & De-Essing**: Dynamically ducks sibilance spikes (6kHz–9kHz) to keep high frequencies clear without harshness.
  * **Transient Shaping**: Enhances initial attack spikes (kick/snare hits) to preserve punch prior to saturation.
  * **Harmonic Saturation**: Emulates analog tube/tape saturation ($\tanh$ transfer curve) for warm glue and density.
  * **Spatial Convolution Reverb**: Introduces filtered room ambience for subtle stereo dimension.
  * **ITU-R BS.1770 LUFS Targeting**: Normalizes integrated loudness for streaming compliance.
  * **Look-Ahead True-Peak Limiter**: Prevents inter-sample peak clipping on output renders.
* **Deep Analytics (`-a`)**: Inspect raw peak/RMS energy, view dB distribution histograms, and evaluate proposed split points without exporting files.
* **Smart Merging & Padding**: Prevents abrupt cuts by adding customizable boundary padding and merging short silence gaps (e.g., breath pauses).

---

## 📋 Prerequisites

TrackIO requires Python 3.8+ and `ffmpeg` (for audio decoding and MP3 export).

### 1. Install FFmpeg
* **macOS**: `brew install ffmpeg`
* **Ubuntu/Debian**: `sudo apt install ffmpeg`
* **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

### 2. Install Python Dependencies
```bash
pip install numpy scipy librosa pydub pyloudnorm
```

---

## Quick Start

### 1. Slice and Auto-Master a Recording
Slice a long-form recording into individual MP3 tracks and apply the mastering engine:
```bash
python trackio.py input_file.wav \
  --auto-master \
  --format mp3 \
  --quality high \
  --output-dir ./mastered_tracks
```

### 2. Master a Single Track (Disable Slicing)
Bypass track slicing entirely to enhance a single audio file:
```bash
python trackio.py single_track.wav \
  --disable-slicing \
  --auto-master \
  --format wav
```

### 3. Analytics Mode (Preview Only)
Analyze dB distribution and review proposed segmentation thresholds without generating output files:
```bash
python trackio.py input_file.wav -a
```

### 4. Example output 
The output below provides an real-world useage example. The settings worked very well against Zoom files)
```bash
$ ./trackio.py --prefix 'boost_' ~/Music/260819-194621.WAV --silence-thresh -39 --min-track-dur 100000 --merge-gap 1000 --padding 10000 -f mp3 --quality ultra --auto-master
2026-09-18 11:18:56,545 - INFO - Loaded /home/Music/260819-194621.WAV at 48000 Hz
2026-09-18 11:19:00,187 - INFO - Processing Track 1/16: silverdales_boost_260819-194621_track_001.mp3 (135500ms)
2026-09-18 11:19:00,190 - INFO - Applying adaptive mastering suite to silverdales_boost_260819-194621_track_001.mp3...
2026-09-18 11:19:04,462 - INFO - Exported silverdales_boost_260819-194621_track_001.mp3 (Mean dB: -25.86 dB)
2026-09-18 11:19:04,471 - INFO - Processing Track 2/16: silverdales_boost_260819-194621_track_002.mp3 (151000ms)
2026-09-18 11:19:04,482 - INFO - Applying adaptive mastering suite to silverdales_boost_260819-194621_track_002.mp3...
2026-09-18 11:19:09,085 - INFO - Exported silverdales_boost_260819-194621_track_002.mp3 (Mean dB: -25.70 dB)
...snip...
```

---

## Analytics Mode Output Example

```text
AUDIO ANALYSIS REPORT
==================================================
Sample Rate: 48000 Hz
Duration: 5592.06 seconds
Raw Peak Amplitude: 4.040073871612549
Peak dBFS (Raw): 12.13 dB
RMS dBFS (Normalized): -32.33 dB

--- DB DISTRIBUTION (Relative to Peak) ---
Bucket              | Seconds     | Percentage | Bar
--------------------------------------------------
-60 to -50 dB       |   1049.00s    |    18.8%     | #############
-50 to -40 dB       |   1368.00s    |    24.5%     | #################
-40 to -30 dB       |   1587.00s    |    28.4%     | ####################
-30 to -20 dB       |   1221.50s    |    21.8%     | ###############
-20 to -10 dB       |      0.00s    |     0.0%     |
-10 to 0 dB         |      0.00s    |     0.0%     |

--- SUGGESTED SEGMENTATION STRATEGY ---
Most audio content lies in bucket: -40 to -30 dB
Suggested Threshold for Segmentation: -35.0 dBFS

--- VISUALIZATION OF SEGMENTATION ---
Active Windows: 4296/11185 (38.4% of audio detected)

--- PROPOSED TRACK LOGIC ---
Using Threshold: -35.0 dBFS
Found 13 potential segments after merging AND duration filtering.
Track 1: Start=52.00s, End=161.50s, Duration=109500ms, Mean dB=-25.51 dB
Track 2: Start=429.00s, End=576.00s, Duration=147000ms, Mean dB=-27.53 dB
...
```

---

## Command Line Configuration Parameters

### Core & Segmentation Options

| Parameter | Short Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `--analyze` | `-a` | Off | Run analysis mode only (no file output). |
| `--disable-slicing` | | Off | Bypass track segmentation and process the input as a single file. |
| `--format` | `-f` | `wav` | Output audio format (wav, mp3). |
| `--quality` | `-q` | `medium` | Bitrate profile for MP3 export (low, medium, high, ultra). |
| `--output-dir` | `-o` | `./sliced_tracks` | Target directory for generated audio files. |
| `--prefix` | `-p` | `""` | String prefix prepended to output filenames. |
| `--min-silence-len` | | Default | Minimum silence duration (in ms) required to register a track split. |
| `--merge-gap` | | Default | Maximum silence gap (in ms) allowed between active regions before merging. |
| `--min-track-dur` | | Default | Minimum duration (in ms) required for a slice to be saved. |
| `--padding` | | Default | Buffer padding (in ms) added to the start and end of extracted segments. |
| `--silence-thresh` | | Dynamic | Silence detection threshold in dBFS. Overrides automated detection. |

### Adaptive Mastering Options (`--auto-master`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--auto-master` | Off | Enables the adaptive multi-stage mastering engine. |
| `--width` | Default | Stereo width expansion factor for side-channel information. |
| `--bass` | Default | Low-frequency boost gain factor (<150Hz). |
| `--mid` | Default | Midrange presence gain factor (300Hz–4kHz). |
| `--treble` | Default | High-frequency sheen gain factor (>8kHz). |
| `--drive` | Default | Analog harmonic saturation multiplier. |
| `--reverb` | Default | Wet mix ratio for spatial convolution room ambience. |
| `--lufs` | Default | Target integrated loudness in LUFS (ITU-R BS.1770). |
| `--norm` | Default | True-peak ceiling in dBFS for look-ahead brickwall limiting. |
| `--deess` | Default | Dynamic de-essing sensitivity factor (6kHz–9kHz). |
| `--transients` | Default | Attack transient preservation factor. |

---

## Mastering Pipeline Architecture

When `--auto-master` is enabled, audio signals pass through the following DSP sequence:

```text
[Input Audio] 
      │
      ▼
 1. Mid/Side Split   ────────► Mono Sub-Bass (<120Hz) + Wide Side Highs
      │
      ▼
 2. Zero-Phase EQ    ────────► Subsonic Cut (<28Hz) + Bass/Mid/Treble Boosts
      │
      ▼
 3. Dynamic Control  ────────► Dynamic Sub-Bass Attenuation + De-Esser (6-9kHz)
      │
      ▼
 4. Coloration       ────────► Transient Accentuation + Harmonic Saturation (tanh)
      │
      ▼
 5. Space            ────────► Filtered Convolution Spatial Reverb
      │
      ▼
 6. Output Stage     ────────► ITU-R BS.1770 LUFS Target + Look-Ahead Peak Limiter
      │
      ▼
[Mastered Output File]
```

---

## Troubleshooting

* **No tracks detected during slicing**: Run in analytics mode (`-a`) to inspect the energy profile. If the recording is exceptionally quiet, manually specify a lower threshold (e.g., `--silence-thresh -45`).
* **Harshness or sibilance on mastered tracks**: Reduce high-frequency sheen using `--treble 1.05` or increase de-essing sensitivity using `--deess 1.5`.
* **Drums lose punch after mastering**: Increase attack transient emphasis with `--transients 1.25` or reduce drive intensity with `--drive 1.05`.

---

## 📄 License

MIT License. Free for personal and commercial use.
