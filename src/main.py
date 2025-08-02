#!/usr/bin/env python3
import os
import sys
import re
import argparse
import torch
import whisperx
import pysubs2
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv(os.path.expanduser("~/.env"))
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    sys.stderr.write("Error: OPENAI_API_KEY not set.\n")
    sys.exit(1)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/*?:\"<>|]", "_", name)

def get_latest_mp4(folder: str) -> str:
    files = [f for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    return max((os.path.join(folder, f) for f in files), key=os.path.getmtime) if files else None

def transcribe_align(video_path: str):
    model = whisperx.load_model("base", device="cpu", compute_type="float32")
    result = model.transcribe(video_path, batch_size=4)
    align_model, metadata = whisperx.load_align_model(
        language_code=result['language'], device="cpu"
    )
    return whisperx.align(result['segments'], align_model, metadata, video_path, "cpu")

def punctuate_text(text: str) -> str:
    prompt = "請在下列中文文字中添加恰當的標點符號，保留原意，不進行斷行：\n" + text
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "你是一位標點助手，僅為文字添加標點，不更改詞序。"},
            {"role": "user",    "content": prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def split_lines(text: str, max_chars: int = 25) -> list:
    parts = re.split(r'(?<=[。！？])', text)
    lines = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            lines.append(part)
        else:
            for seg in re.split(r'[，、]', part):
                seg = seg.strip()
                if not seg:
                    continue
                for i in range(0, len(seg), max_chars):
                    lines.append(seg[i:i+max_chars])
    return lines

def clean_text(text: str) -> str:
    return ''.join(ch for ch in text if ('\u4e00' <= ch <= '\u9fff') or ch.isalnum())

def final_cleanup(line: str) -> str:
    prompt = "以下為一條字幕，請檢查並修正文字錯誤或不通順，保持字符長度不變：\n" + line
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "你是一位字幕校對助手，僅最小限度修正文字，保留長度。"},
            {"role": "user",    "content": prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def generate_srt(aligned, output_path: str):
    MAX_CHARS = 25
    MIN_MERGE = 8

    subs = pysubs2.SSAFile()

    for seg in aligned.get('segments', []):
        words  = seg.get('words', [])
        tokens = [w['word'].replace('▁', '') for w in words]
        times  = [(w['start'], w['end'])         for w in words]
        raw    = "".join(tokens)
        if not raw:
            continue

        # 1) First GPT pass: punctuation
        punct = punctuate_text(raw)

        # 2) Split into ≤MAX_CHARS-char lines
        lines = split_lines(punct, max_chars=MAX_CHARS)

        # 3) Merge any lines shorter than MIN_MERGE into previous,
        #    but only if result stays ≤ MAX_CHARS
        merged = []
        for line in lines:
            if merged and len(line) < MIN_MERGE:
                candidate = merged[-1] + line
                if len(candidate) <= MAX_CHARS:
                    merged[-1] = candidate
                else:
                    merged.append(line)
            else:
                merged.append(line)
        lines = merged

        # 4) Align & final cleanup
        for line in lines:
            clean = clean_text(line)
            idx = raw.find(clean)

            # Fallback: prefix/suffix match
            if idx < 0 and len(clean) > 5:
                for L in range(len(clean)-1, 4, -1):
                    pref = clean[:L]
                    idx = raw.find(pref)
                    if idx >= 0:
                        clean = pref
                        break
            if idx < 0 and len(clean) > 5:
                for L in range(len(clean)-1, 4, -1):
                    suf = clean[-L:]
                    idx = raw.find(suf)
                    if idx >= 0:
                        clean = suf
                        break

            if idx < 0:
                # Attach dropped line to previous subtitle
                if subs.events:
                    subs.events[-1].text += line
                continue

            # Map to word indices
            count = 0
            start_i = end_i = None
            for i, tok in enumerate(tokens):
                l = len(tok)
                if start_i is None and count + l > idx:
                    start_i = i
                if start_i is not None and count + l >= idx + len(clean):
                    end_i = i
                    break
                count += l

            if start_i is None or end_i is None:
                if subs.events:
                    subs.events[-1].text += line
                continue

            s = times[start_i][0]
            e = times[end_i][1]

            # 5) Second GPT pass: final cleanup
            txt = final_cleanup(line)

            subs.append(pysubs2.SSAEvent(
                start=int(s*1000), end=int(e*1000), text=txt
            ))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subs.save(output_path)
    return output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Subtitle gen: STT→punct→split→smart-merge→align→cleanup→export"
    )
    parser.add_argument('-d', '--dir', default=os.path.expanduser('~/Movies'))
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    video = get_latest_mp4(args.dir)
    if not video:
        sys.exit(f"Error: No MP4 in {args.dir}")
    out = args.output or os.path.join(
        os.path.dirname(video),
        sanitize_filename(os.path.splitext(os.path.basename(video))[0]) + '.srt'
    )

    print("Transcribing and aligning...")
    aligned = transcribe_align(video)

    print(f"Generating subtitles to {out}...")
    generate_srt(aligned, out)
    print("Done.")
