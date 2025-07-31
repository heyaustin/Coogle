#!/usr/bin/env python3
"""
Split a continuous mouse‑click recording (e.g. mouse.m4a) into
separate WAV files, one per click.

Requirements:
    pip install pydub
    # And have FFmpeg installed (https://ffmpeg.org/)
"""

from pydub import AudioSegment, silence
from pathlib import Path
# import os


def split_clicks(
    input_path: str,
    export_dir: str = "clicks",
    silence_thresh_db: int = -25,    # raise (e.g. -20) if it misses clicks,
    min_silence_ms: int = 50,        # lower (e.g. 30) if clicks are very rapid
    pad_ms: int = 10                 # extra audio before & after each click
) -> None:
    """
    Detect non‑silent regions (the clicks) and save each to its own .wav.

    Parameters
    ----------
    input_path : str
        Path to the .m4a (or any FFmpeg‑readable file).
    export_dir : str
        Folder where .wav files will be written. Created if missing.
    silence_thresh_db : int
        Anything louder than this (in dBFS) counts as sound.
    min_silence_ms : int
        Length of quiet required to separate clicks.
    pad_ms : int
        Milliseconds of padding added before/after each detected click.
    """
    # Load and ensure mono (clicks are mono anyway, saves RAM & CPU)
    audio = AudioSegment.from_file(input_path).set_channels(1)

    # Detect bursts that are above the threshold
    non_silent_ranges = silence.detect_nonsilent(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db
    )

    if not non_silent_ranges:
        print("No clicks detected — try tweaking silence_thresh_db or min_silence_ms.")
        return

    Path(export_dir).mkdir(parents=True, exist_ok=True)

    for idx, (start_ms, end_ms) in enumerate(non_silent_ranges, start=1):
        clip = audio[max(0, start_ms - pad_ms): min(len(audio), end_ms + pad_ms)]
        out_path = Path(export_dir) / f"{input_path.replace(".m4a", "")}_{idx:04d}.wav"
        clip.export(out_path, format="wav")
        print(f"Saved: {out_path}")

    print(f"\nDone! Extracted {len(non_silent_ranges)} clicks to '{export_dir}/'.")


if __name__ == "__main__":
    for i in range(10):
        split_clicks(f"{i}.m4a", export_dir=f"{i}")
