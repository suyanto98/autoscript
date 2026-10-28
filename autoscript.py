#!/usr/bin/env python3
"""Batch-generate transcript scripts from Indonesian videos, fully local.

Pipeline per video: ffmpeg extract audio -> faster-whisper (ASR Bahasa
Indonesia + timestamps) -> tulis script (.txt, opsional .srt) di folder yang
sama dengan videonya.

Usage:
    .venv/bin/python autoscript.py --source ./videos
    .venv/bin/python autoscript.py --source ./videos --srt --whisper-model small
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # progress prints visible immediately even when redirected to a log file

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}

PARAGRAPH_GAP = 1.5  # jeda (detik) antar segmen yang dianggap ganti paragraf


def tmp_for(path: Path) -> Path:
    """Same-directory scratch path for atomic writes, so a killed run never
    leaves a truncated script that a resume would mistake for "already done"."""
    return path.with_name(path.stem + ".part" + path.suffix)


def write_atomic(path: Path, content: str) -> None:
    tmp_path = tmp_for(path)
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def extract_audio(video_path: Path, wav_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", "16000", "-vn", str(wav_path),
        ],
        check=True, capture_output=True,
    )


def transcribe(wav_path: Path, whisper_model, language: str) -> list[dict]:
    segments_iter, info = whisper_model.transcribe(
        str(wav_path), language=language, vad_filter=True,
    )
    segments = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"start": seg.start, "end": seg.end, "text": text})
        print(f"\r  transkripsi {seg.end:7.1f}s / {info.duration:7.1f}s", end="")
    print()
    return segments


def build_script(segments: list[dict], with_timestamps: bool) -> str:
    """Gabungkan segmen jadi paragraf; ganti paragraf kalau ada jeda panjang."""
    paragraphs: list[str] = []
    current: list[str] = []
    prev_end = None
    para_start = 0.0
    for seg in segments:
        if prev_end is not None and seg["start"] - prev_end >= PARAGRAPH_GAP and current:
            paragraphs.append(format_paragraph(current, para_start, with_timestamps))
            current = []
        if not current:
            para_start = seg["start"]
        current.append(seg["text"])
        prev_end = seg["end"]
    if current:
        paragraphs.append(format_paragraph(current, para_start, with_timestamps))
    return "\n\n".join(paragraphs) + "\n"


def format_paragraph(texts: list[str], start: float, with_timestamps: bool) -> str:
    body = " ".join(texts)
    if not with_timestamps:
        return body
    m, s = divmod(int(start), 60)
    h, m = divmod(m, 60)
    return f"[{h:02d}:{m:02d}:{s:02d}] {body}"


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: list[dict]) -> str:
    blocks = [
        f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n{seg['text']}\n"
        for i, seg in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def process_video(video_path: Path, txt_path: Path, srt_path: Path | None, whisper_model, args) -> None:
    with tempfile.TemporaryDirectory(prefix="autoscript_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        print("  ekstrak audio ...")
        extract_audio(video_path, wav_path)
        segments = transcribe(wav_path, whisper_model, args.language)

    if not segments:
        raise RuntimeError("tidak ada ucapan terdeteksi")

    if srt_path is not None:
        write_atomic(srt_path, build_srt(segments))
    # .txt ditulis terakhir: keberadaannya jadi penanda video sudah selesai.
    write_atomic(txt_path, build_script(segments, args.timestamps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Folder berisi video sumber (dicari rekursif)")
    parser.add_argument("--whisper-model", default="medium", help="Ukuran model faster-whisper (default: medium)")
    parser.add_argument("--language", default="id", help="Kode bahasa audio (default: id)")
    parser.add_argument("--suffix", default="", help="Sufiks nama file script, mis. _script (default: kosong)")
    parser.add_argument("--srt", action="store_true", help="Simpan juga subtitle .srt")
    parser.add_argument("--timestamps", action="store_true", help="Tambahkan [jj:mm:dd] di awal tiap paragraf .txt")
    parser.add_argument("--overwrite", action="store_true", help="Proses ulang walau script sudah ada")
    args = parser.parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    if not source_dir.is_dir():
        sys.exit(f"Folder source tidak ditemukan: {source_dir}")

    def video_sort_key(path: Path) -> tuple:
        # Sort by the leading number in the title (e.g. "2. Foo" before
        # "10. Bar"), not plain alphabetical order.
        match = re.match(r"\s*(\d+)", path.stem)
        number = int(match.group(1)) if match else float("inf")
        return (str(path.parent), number, path.name)

    videos = sorted(
        (p for p in source_dir.rglob("*")
         if p.is_file() and p.suffix.lower() in VIDEO_EXTS
         and not any(part.startswith(".") for part in p.relative_to(source_dir).parts)),
        key=video_sort_key,
    )
    if not videos:
        sys.exit(f"Tidak ada file video ({', '.join(sorted(VIDEO_EXTS))}) di {source_dir}")

    print(f"Ditemukan {len(videos)} video di {source_dir}")
    print(f"[setup] loading faster-whisper ({args.whisper_model}) ...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    ok, failed, skipped = 0, 0, 0
    for i, video_path in enumerate(videos, 1):
        rel = video_path.relative_to(source_dir)
        txt_path = video_path.with_name(video_path.stem + args.suffix + ".txt")
        srt_path = video_path.with_name(video_path.stem + args.suffix + ".srt") if args.srt else None

        print(f"\n[{i}/{len(videos)}] {rel}")
        if txt_path.exists() and not args.overwrite:
            print(f"  sudah ada, dilewati -> {txt_path.name}")
            skipped += 1
            continue

        try:
            process_video(video_path, txt_path, srt_path, whisper_model, args)
            print(f"  selesai -> {txt_path.name}" + (f", {srt_path.name}" if srt_path else ""))
            ok += 1
        except Exception as exc:
            print(f"  GAGAL: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nSelesai. Berhasil: {ok}, dilewati: {skipped}, gagal: {failed}")


if __name__ == "__main__":
    main()
