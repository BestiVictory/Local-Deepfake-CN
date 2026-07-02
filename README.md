# Word-Level Video Replacement Pipeline

This project implements a word-level local video replacement pipeline.
It first selects suitable short video clips, then modifies one word or character in the audio, generates lip-synced replacement video segments, and finally replaces only the modified time range in the original video.

---

## 1. Required Components

Before running the pipeline, prepare the following components:

* FFmpeg
* FunASR model
* dlib `shape_predictor_68_face_landmarks.dat`
* InsightFace model
* CosyVoice2 model
* MFA dictionary and acoustic model
* Demucs
* Wav2Lip / `inference_torch.py`
* DeepSeek API key

---

## 2. Overall Pipeline

```text
Long video
→ semantic video clipping
→ single-face filtering
→ mouth-open filtering
→ audio extraction
→ ASR transcription
→ word/character replacement
→ MFA timestamp alignment
→ CosyVoice2 speech generation
→ new word audio extraction
→ duration adjustment
→ background audio extraction
→ final modified audio generation
→ Wav2Lip lip-sync generation
→ local video replacement
→ final output video
```

---

## 3. Script 1: `1-video_clip.py`

### Function

`1-video_clip.py` processes long videos and keeps only suitable short clips.

```text
Long video
→ Split into short clips based on semantic sentence boundaries
→ Keep clips with a single stable face
→ Keep clips where the mouth is open
```

### Variables That Need to Be Modified

#### 3.1 dlib Landmark Model Path

```python
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
```

This path points to the dlib 68-point facial landmark model.

Modify it if your model is stored somewhere else.

Example:

```python
predictor = dlib.shape_predictor("/your/project/models/shape_predictor_68_face_landmarks.dat")
```

Important: this path is resolved based on the current working directory when the script is executed.
If you run the script from a different directory, use an absolute path to avoid path errors.

---

#### 3.2 Input Video Folder

```python
parser.add_argument(
    "--input_dir",
    type=str,
    default="/home/ubuntu/mnt_4T/why/AV/original_videos/mingren",
    help="包含视频的文件夹路径"
)
```

This is the folder containing the original long videos.

Only videos inside this folder will be processed.

Recommended structure:

```text
original_videos/
├── video_001.mp4
├── video_002.mp4
└── video_003.mp4
```

Example modification:

```bash
--input_dir "/your/path/original_videos"
```

---

#### 3.3 Video Clip Output Folder

```python
parser.add_argument(
    "--result_dir",
    type=str,
    default="video_segments/mingren",
    help="输出结果文件夹"
)
```

This is the output folder for video clipping results.

For each input video, the script creates a separate subfolder.

Typical output structure:

```text
video_segments/
└── video_001/
    ├── audio.wav
    ├── output_clips/
    ├── filtered_clips/
    └── final_clips/
```

Folder meaning:

| Folder            | Meaning                               |
| ----------------- | ------------------------------------- |
| `output_clips/`   | Initial semantic video clips          |
| `filtered_clips/` | Clips that pass single-face filtering |
| `final_clips/`    | Clips that pass mouth-open filtering  |

The clips in `final_clips/` should be used as candidate input clips for the next audio modification stage.

---

#### 3.4 Log Folder

```python
parser.add_argument(
    "--log_dir",
    type=str,
    default="clip_logs/mingren",
    help="日志文件夹"
)
```

This folder stores processing logs for each video.

Example output:

```text
clip_logs/
├── video_001.log
├── video_002.log
└── completed_videos.txt
```

Use this folder to check which video failed and why.

---

#### 3.5 Number of Worker Processes

```python
parser.add_argument(
    "--workers",
    type=int,
    default=4,
    help="并发进程数"
)
```

This controls the number of parallel processes.

Recommended setting:

| Hardware             | Suggested Value |
| -------------------- | --------------: |
| Low CPU / low memory |      `1` or `2` |
| Normal workstation   |             `4` |
| High-core server     |   `8` or higher |

Do not blindly increase this value.
This script loads ASR, dlib, and face recognition models in worker processes. Too many workers may cause GPU or memory exhaustion.

---

#### 3.6 Checkpoint File

```python
parser.add_argument(
    "--checkpoint_file",
    type=str,
    default="clip_logs/mingren/completed_videos.txt",
    help="已完成视频记录文件"
)
```

This file records videos that have already been processed.

If the script is interrupted, it can skip completed videos during the next run.

Do not delete this file unless you want to reprocess all videos from the beginning.

---

### Example Command

```bash
python 1-video_clip.py \
  --input_dir "/your/path/original_videos" \
  --result_dir "./video_segments" \
  --log_dir "./clip_logs" \
  --workers 4 \
  --checkpoint_file "./clip_logs/completed_videos.txt"
```

---

## 4. Script 2: `2-audio_make.py`

### Function

`2-audio_make.py` modifies one word or character in each selected video clip and generates the final modified audio.

```text
Video clip
→ Extract audio
→ Run ASR transcription
→ Select one word or character to modify
→ Use MFA to locate the original word/character time range
→ Generate modified speech with CosyVoice2
→ Extract the newly generated word/character audio
→ Adjust its duration
→ Extract background audio
→ Merge the modified speech with background audio
→ Generate final modified audio
→ Save replacement metadata to CSV
```

---

### Variables That Need to Be Modified

#### 4.1 DeepSeek API Key

```python
deepseek_api_key = ""
```

Fill in your DeepSeek API key.

Example:

```python
deepseek_api_key = "your_deepseek_api_key"
```

For safer usage, it is better to load the key from an environment variable instead of writing it directly into the script.

Recommended version:

```python
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
```

Then set it in the terminal:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

---

#### 4.2 Input Video Folder

```python
input_folder = "./data/temp_1000/input_videos/"
```

This folder contains the short video clips that need audio modification.

Usually, these clips should come from:

```text
1-video_clip.py → final_clips/
```

Recommended structure:

```text
data/temp_1000/input_videos/
├── clip_001.mp4
├── clip_002.mp4
└── clip_003.mp4
```

All later stages rely on the video filename.
Do not rename files randomly after this stage.

---

#### 4.3 Extracted Audio Output Folder

```python
output_audio_folder = "./data/temp_1000/output_audio"
```

This folder stores the original extracted audio from each video clip.

Example output:

```text
output_audio/
├── clip_001.mp3
├── clip_002.mp3
└── clip_003.mp3
```

---

#### 4.4 Silent Video Output Folder

```python
output_video_folder = "./data/temp_1000/output_video"
```

This folder stores the video clips with audio removed.

These silent videos are later used as the visual input for Wav2Lip.

Example output:

```text
output_video/
├── clip_001_no_audio.mp4
├── clip_002_no_audio.mp4
└── clip_003_no_audio.mp4
```

If Wav2Lip requires matching filenames between video and audio, make sure the filenames are normalized before running `3-video_make.sh`.

---

#### 4.5 CSV Output File

```python
output_csv_file = "./data/temp_1000/output.csv"
```

This CSV records the metadata needed by the final replacement script.

The CSV should include:

```text
audio name,
final modified audio path,
original full text,
modified full text,
original word/character,
modified word/character,
original word/character timestamp
```

The timestamp column must be compatible with `4-final_change.py`.

Recommended timestamp format:

```text
start,end
```

Example:

```text
1.24,1.63
```

Do not use this format unless `4-final_change.py` is also modified accordingly:

```text
1.24-1.63
```

---

#### 4.6 Generated Full Speech Folder

```python
output_generated_audio_folder = "./data/temp_1000/adjust_audio"
```

This folder stores the full speech generated by CosyVoice2 after text modification.

Example output:

```text
adjust_audio/
├── clip_001.wav
├── clip_002.wav
└── clip_003.wav
```

This is an intermediate folder, not the final modified audio folder.

---

#### 4.7 Background-Mixed Modified Segment Folder

```python
output_noise_adjusted_audio_folder = "./data/temp_1000/output_noise_adjusted_audio"
```

This folder stores the modified word/character segment after merging with background audio.

It contains only the locally modified segment, not the full final audio.

---

#### 4.8 Speed-Adjusted Audio Folder

```python
output_speed_adjusted_audio_folder = "./data/temp_1000/output_speed_adjusted_audio"
```

This folder stores the new word/character audio after duration adjustment.

The purpose is to make the generated word/character match the duration of the original word/character.

This step is necessary because the generated speech duration may not match the original timestamp range.

---

#### 4.9 Final Modified Audio Folder

```python
output_final_audio_folder = "./data/temp_1000/output_final_audio"
```

This folder stores the final modified audio.

Example output:

```text
output_final_audio/
├── clip_001_final.wav
├── clip_002_final.wav
└── clip_003_final.wav
```

This folder should be used as the audio input folder for Wav2Lip.

---

### Recommended Directory Mapping

```text
2-audio_make.py input_folder
→ original short video clips

2-audio_make.py output_video_folder
→ silent video input for Wav2Lip

2-audio_make.py output_final_audio_folder
→ modified audio input for Wav2Lip

2-audio_make.py output_csv_file
→ metadata input for 4-final_change.py
```

---

### Example Configuration

```python
deepseek_api_key = "your_deepseek_api_key"

input_folder = "./data/temp_1000/input_videos/"
output_audio_folder = "./data/temp_1000/output_audio"
output_video_folder = "./data/temp_1000/output_video"
output_csv_file = "./data/temp_1000/output.csv"

output_generated_audio_folder = "./data/temp_1000/adjust_audio"
output_noise_adjusted_audio_folder = "./data/temp_1000/output_noise_adjusted_audio"
output_speed_adjusted_audio_folder = "./data/temp_1000/output_speed_adjusted_audio"
output_final_audio_folder = "./data/temp_1000/output_final_audio"
```

---

### Run

```bash
python 2-audio_make.py
```

---

## 5. Script 3: `3-video_make.sh`

### Function

`3-video_make.sh` runs Wav2Lip inference.

```text
Silent video + modified audio
→ Generate lip-synced replacement video
```

---

### Variables That Need to Be Modified

#### 5.1 Wav2Lip Project Directory

```bash
cd "/home/ubuntu/mnt_4T/why/AV/wav2lip256"
```

This should point to the directory where `inference_torch.py` is located.

Example:

```bash
cd "/your/path/wav2lip256"
```

If this path is wrong, the shell script will not find `inference_torch.py`.

---

#### 5.2 Silent Video Input Folder

```bash
--face_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000_temp/video/quarter_1"
```

This folder should contain silent videos.

It should normally correspond to:

```text
2-audio_make.py → output_video_folder
```

Example:

```bash
--face_dir "/your/project/data/temp_1000/output_video"
```

---

#### 5.3 Modified Audio Input Folder

```bash
--audio_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000_temp/audio/quarter_1"
```

This folder should contain the final modified audio files.

It should normally correspond to:

```text
2-audio_make.py → output_final_audio_folder
```

Example:

```bash
--audio_dir "/your/project/data/temp_1000/output_final_audio"
```

---

#### 5.4 Lip-Synced Output Folder

```bash
--outfile_dir "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/all_fake_results/all_fake_results1"
```

This folder stores the lip-synced replacement videos generated by Wav2Lip.

It should later be used as:

```text
4-final_change.py → FOLDER_A
```

Example:

```bash
--outfile_dir "/your/project/data/temp_1000/all_fake_results"
```

---

### Important Filename Rule

The video files and audio files used by Wav2Lip must be correctly paired.

For example:

```text
output_video/
└── clip_001.mp4

output_final_audio/
└── clip_001.wav
```

The generated Wav2Lip output should also keep a filename that can be matched by `4-final_change.py`.

For example:

```text
all_fake_results/
└── clip_001.mp4
```

If Wav2Lip changes the output filenames, rename the generated files before running `4-final_change.py`.

---

### Example Script

```bash
#!/bin/bash

cd "/your/path/wav2lip256"

python inference_torch.py \
  --face_dir "/your/project/data/temp_1000/output_video" \
  --audio_dir "/your/project/data/temp_1000/output_final_audio" \
  --outfile_dir "/your/project/data/temp_1000/all_fake_results"
```

---

### Run

```bash
bash 3-video_make.sh
```

---

## 6. Script 4: `4-final_change.py`

### Function

`4-final_change.py` replaces only the modified word/character time range in the original video.

```text
Original video segment before the modified word
+ modified segment from the lip-synced replacement video
+ original video segment after the modified word
= final locally replaced video
```

---

### Variables That Need to Be Modified

#### 6.1 CSV Path

```python
CSV_PATH = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/output.csv"
```

This should point to the CSV generated by `2-audio_make.py`.

Example:

```python
CSV_PATH = "/your/project/data/temp_1000/output.csv"
```

The CSV must contain the original word/character timestamp column.

Recommended timestamp format:

```text
start,end
```

Example:

```text
1.24,1.63
```

---

#### 6.2 Lip-Synced Replacement Video Folder

```python
FOLDER_A = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/all_fake_results"
```

This folder contains the lip-synced replacement videos generated by Wav2Lip.

It should correspond to:

```text
3-video_make.sh → --outfile_dir
```

Example:

```python
FOLDER_A = "/your/project/data/temp_1000/all_fake_results"
```

The filenames must match the original video clip basenames recorded in the CSV.

---

#### 6.3 Original Video Folder

```python
FOLDER_B = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/input_videos"
```

This folder contains the original video clips before replacement.

It should normally correspond to:

```text
2-audio_make.py → input_folder
```

Example:

```python
FOLDER_B = "/your/project/data/temp_1000/input_videos"
```

The original video must have the same basename as the generated lip-synced replacement video.

Example:

```text
input_videos/
└── clip_001.mp4

all_fake_results/
└── clip_001.mp4
```

---

#### 6.4 Final Output Folder

```python
OUTPUT_DIR = "/home/ubuntu/mnt_4T/why/AV/data/temp_1000/final_results"
```

This folder stores the final locally replaced videos.

Example:

```python
OUTPUT_DIR = "/your/project/data/temp_1000/final_results"
```

Output example:

```text
final_results/
├── clip_001.mp4
├── clip_002.mp4
└── final_error.txt
```

`final_error.txt` records failed replacement tasks.

---

#### 6.5 Number of Worker Processes

```python
MAX_WORKERS = 4
```

This controls how many videos are processed in parallel during final replacement.

Recommended setting:

| Hardware             | Suggested Value |
| -------------------- | --------------: |
| Low CPU / low memory |      `1` or `2` |
| Normal workstation   |             `4` |
| High-core server     |   `8` or higher |

If FFmpeg processes are too slow, you can increase this value.
If CPU usage or memory usage is too high, reduce this value.

---

### Example Configuration

```python
CSV_PATH = "/your/project/data/temp_1000/output.csv"
FOLDER_A = "/your/project/data/temp_1000/all_fake_results"
FOLDER_B = "/your/project/data/temp_1000/input_videos"
OUTPUT_DIR = "/your/project/data/temp_1000/final_results"
MAX_WORKERS = 4
```

---

### Run

```bash
python 4-final_change.py
```

---

## 7. Recommended Full Running Order

### Step 1: Clip Long Videos

```bash
python 1-video_clip.py \
  --input_dir "/your/path/original_videos" \
  --result_dir "./video_segments" \
  --log_dir "./clip_logs" \
  --workers 4 \
  --checkpoint_file "./clip_logs/completed_videos.txt"
```

After this step, select suitable clips from:

```text
video_segments/<video_name>/final_clips/
```

Copy them into:

```text
data/temp_1000/input_videos/
```

---

### Step 2: Generate Modified Audio

```bash
python 2-audio_make.py
```

This generates:

```text
data/temp_1000/output_audio/
data/temp_1000/output_video/
data/temp_1000/output_final_audio/
data/temp_1000/output.csv
```

---

### Step 3: Generate Lip-Synced Replacement Videos

```bash
bash 3-video_make.sh
```

This generates:

```text
data/temp_1000/all_fake_results/
```

---

### Step 4: Replace Only the Modified Segment

```bash
python 4-final_change.py
```

This generates:

```text
data/temp_1000/final_results/
```

---

## 8. Path Consistency Checklist

Before running the full pipeline, check the following mappings carefully.

| Stage                          | Output                      | Used By                                    |
| ------------------------------ | --------------------------- | ------------------------------------------ |
| `1-video_clip.py`              | `final_clips/`              | copied into `2-audio_make.py input_folder` |
| `2-audio_make.py`              | `output_video_folder`       | `3-video_make.sh --face_dir`               |
| `2-audio_make.py`              | `output_final_audio_folder` | `3-video_make.sh --audio_dir`              |
| `2-audio_make.py`              | `output_csv_file`           | `4-final_change.py CSV_PATH`               |
| `3-video_make.sh`              | `--outfile_dir`             | `4-final_change.py FOLDER_A`               |
| `2-audio_make.py input_folder` | original clips              | `4-final_change.py FOLDER_B`               |
| `4-final_change.py`            | `OUTPUT_DIR`                | final local replacement videos             |

---

## 9. Common Errors

### 9.1 dlib Model Not Found

Error cause:

```text
shape_predictor_68_face_landmarks.dat path is wrong.
```

Fix:

Use an absolute path:

```python
predictor = dlib.shape_predictor("/your/project/models/shape_predictor_68_face_landmarks.dat")
```

---

### 9.2 DeepSeek API Error

Error cause:

```text
deepseek_api_key is empty or invalid.
```

Fix:

Set the API key correctly:

```python
deepseek_api_key = "your_deepseek_api_key"
```

Or use:

```python
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
```

---

### 9.3 Wav2Lip Cannot Find Input Files

Possible causes:

* `--face_dir` is wrong
* `--audio_dir` is wrong
* video and audio filenames do not match
* silent video files still use `_no_audio` suffix while the audio files use another basename

Fix:

Make sure paired files use consistent basenames.

Example:

```text
face_dir/
└── clip_001.mp4

audio_dir/
└── clip_001.wav
```

---

### 9.4 Final Replacement Cannot Find Videos

Possible causes:

* `FOLDER_A` does not point to Wav2Lip output videos
* `FOLDER_B` does not point to original input video clips
* filenames are inconsistent
* CSV filename basenames do not match the video basenames

Fix:

Make sure these files exist:

```text
FOLDER_A/clip_001.mp4
FOLDER_B/clip_001.mp4
```

---

### 9.5 Timestamp Format Error

`4-final_change.py` expects the timestamp field to be split into two values.

Recommended CSV timestamp format:

```text
start,end
```

Example:

```text
1.24,1.63
```

If your CSV stores timestamps like this:

```text
1.24-1.63
```

then either modify `2-audio_make.py` to write `start,end`, or modify `4-final_change.py` to split by `-`.

Recommended fix in `2-audio_make.py`:

```python
data.append([
    audio_path,
    final_audio_path,
    original_text,
    modified_text,
    old_word,
    new_word,
    f"{start_time_old},{end_time_old}"
])
```

---

## 10. Responsible Use

This pipeline can generate locally modified lip-synced videos.
Use it only on authorized or consented video materials, and clearly label generated or modified outputs when necessary.
