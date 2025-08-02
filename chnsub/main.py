#!/usr/bin/env python3
"""
A subtitle generator that automatically selects the newest MP4 in a folder.
1. Scans a specified folder for the newest .mp4 file
2. Transcribes & word-aligns via WhisperX
3. Extracts per-word tokens with precise timestamps
4. Groups tokens into lines between min_tokens and max_tokens words
5. Cleans up text via OpenAI GPT-4 Turbo (adding correct punctuation)
6. Exports precise subtitles as .srt
"""
import os
import sys
import re
import argparse
import torch
import pysubs2
from dotenv import load_dotenv

# Load environment and OpenAI API key
load_dotenv(os.path.expanduser("~/.env"))
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    sys.stderr.write("Error: OPENAI_API_KEY not set.\n")
    sys.exit(1)

# WhisperX import
try:
    import whisperx
except ModuleNotFoundError:
    sys.stderr.write("Error: please install whisperx, torch, transformers.\n")
    sys.exit(1)

# Utility: sanitize filenames for output
def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/*?:\"<>|]", "_", name)

# Step 1: Find newest MP4 in folder
def get_latest_mp4(folder: str) -> str:
    files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    if not files:
        return None
    paths = [os.path.join(folder, f) for f in files]
    return max(paths, key=os.path.getmtime)

# Step 2: Transcribe and word-align with WhisperX
def transcribe_align(video_path: str, device: str):
    whisper_device = "cpu" if device == "mps" else device
    compute_type = "float16" if whisper_device == "cuda" else "float32"
    model = whisperx.load_model("base", whisper_device, compute_type=compute_type)
    result = model.transcribe(video_path, batch_size=8)
    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"], device=whisper_device
    )
    return whisperx.align(result["segments"], align_model, metadata, video_path, whisper_device)

# Step 3: Extract per-word tokens with precise timestamps
def process_segments(aligned) -> list:
    segments = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []):
            token = w.get("word", "").replace("▁", " ").strip()
            start = w.get("start"); end = w.get("end")
            if not token or start is None or end is None:
                continue
            segments.append({"token": token, "start": start, "end": end})
    return segments

# Step 4: Group tokens into lines between min_tokens and max_tokens words
def group_sentences(segments: list, min_tokens: int = 15, max_tokens: int = 25) -> list:
    groups = []
    current_tokens = []
    start_time = None
    end_time = None
    for seg in segments:
        if start_time is None:
            start_time = seg["start"]
        # if adding next token exceeds max, flush if we've reached at least min_tokens
        if len(current_tokens) >= min_tokens and len(current_tokens) + 1 > max_tokens:
            groups.append({"text": " ".join(current_tokens), "start": start_time, "end": end_time})
            current_tokens = []
            start_time = seg["start"]
        current_tokens.append(seg["token"])
        end_time = seg["end"]
    # flush remaining
    if current_tokens and start_time is not None and end_time is not None:
        groups.append({"text": " ".join(current_tokens), "start": start_time, "end": end_time})
    return groups

# Step 5: Clean up text via OpenAI (add correct punctuation)
def cleanup_text(text: str, model: str = "gpt-4-turbo") -> str:
    prompt = (
        "請在下列文字中添加恰當的中文標點符號，並修正拼寫與空格，同時保留原意：\n" + text
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一個文字校正和標點添加助手。請為輸入添加合適的中文標點，不更改詞意。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        sys.stderr.write(f"Warning: cleanup failed: {e}\n")
        return text

# Step 6: Write SRT file with precise timings
def write_srt(groups: list, output_path: str) -> str:
    subs = pysubs2.SSAFile()
    for idx, g in enumerate(groups, start=1):
        start_ms = int(g["start"] * 1000)
        end_ms = int(g["end"] * 1000)
        text = cleanup_text(g["text"])
        subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subs.save(output_path)
    return output_path

# Main entrypoint: orchestrate all steps
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate subtitles (.srt) from the newest MP4 in a folder."
    )
    parser.add_argument(
        "-d", "--dir", default=os.path.expanduser("~/Movies"),
        help="Folder to scan for .mp4 files (default: ~/Movies)"
    )
    parser.add_argument(
        "-o", "--output", help="Custom output .srt path"
    )
    parser.add_argument(
        "-n", "--min_tokens", type=int, default=15,
        help="Minimum words per subtitle line (default: 15)"
    )
    parser.add_argument(
        "-m", "--max_tokens", type=int, default=25,
        help="Maximum words per subtitle line (default: 25)"
    )
    args = parser.parse_args()

    video_path = get_latest_mp4(args.dir)
    if not video_path:
        sys.exit(f"Error: No .mp4 files found in {args.dir}")
    print(f"Processing: {video_path}")

    base = os.path.splitext(os.path.basename(video_path))[0]
    default_srt = os.path.join(os.path.dirname(video_path), sanitize_filename(base) + ".srt")
    output_path = args.output or default_srt

    # Select device: CUDA > CPU (no mps for WhisperX)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    aligned = transcribe_align(video_path, device)
    segments = process_segments(aligned)
    groups = group_sentences(segments, min_tokens=args.min_tokens, max_tokens=args.max_tokens)
    write_srt(groups, output_path)
    print(f"Done. Subtitles saved to: {output_path}")
