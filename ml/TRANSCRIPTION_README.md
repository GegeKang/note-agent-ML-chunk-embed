# Local Audio/Video Transcription

This project supports **local Whisper** transcription for audio/video files and flows the text into the existing Stage 2 (chunking) and Stage 4 (structured extraction) pipeline.

## Requirements

- Python 3.10+ (tested with 3.12)
- `ffmpeg` on PATH
- Local Whisper package:
  - `openai-whisper`
- For this repo, `.env` is loaded via `python-dotenv`
- This setup uses **local Whisper** only. The OpenAI API is **not** used.

### Install

```bash
python3 -m pip install -U openai-whisper
```
--- 

## Run (Bash)

Use the provided script to transcribe a file to `ml/outputs/full_transcript.txt`:

```bash
./ml/transcribe_media.sh /path/to/file.mp4
```

You can also provide a note id and/or output path:

```bash
./ml/transcribe_media.sh /path/to/file.mp4 123 /absolute/path/output.txt
```
---

## Output

The script writes the full transcript to:

```
/Users/gegekang/Desktop/note-agent-ML-chunk-embed/ml/outputs/full_transcript.txt
```

The pipeline also stores the cleaned text in `derived/<workspace>/<note_id>/extracted.txt`.
