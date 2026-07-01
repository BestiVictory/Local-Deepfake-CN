# Local-Deepfake-CN
Word-Level Video Replacement Pipeline

## 1. Required Components

You need to prepare the following components:

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

## 2. `1-video_clip.py`

Long video
→ Split into short clips based on semantic sentence boundaries
→ Keep clips with a single stable face
→ Keep clips where the mouth is open

---

## 3. `2-audio_make.py`

Video clip
→ Extract the audio
→ Run ASR transcription
→ Select one word or character to modify
→ Use MFA to locate the time range of the original word or character
→ Generate the modified speech with CosyVoice2
→ Extract the newly generated word or character audio
→ Adjust its duration to match the original segment
→ Extract the background audio
→ Merge the modified speech segment with the background audio
→ Generate the final modified audio
→ Record the replacement metadata in a CSV file

---

## 4. `3-video_make.sh`

Silent video + modified audio
→ Generate a lip-synced replacement video with Wav2Lip / `inference_torch.py`

---

## 5. `4-final_change.py`

Original video segment before the modified word

* the corresponding modified segment from the lip-synced replacement video
* original video segment after the modified word

= Final locally replaced video
