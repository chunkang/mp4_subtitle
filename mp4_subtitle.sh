#!/usr/bin/env python3
"""Transcribe a video with Whisper and burn the subtitles onto the original.

Unlike its sibling mp4_silence_removal, this does no consolidation or silence
removal: it takes one or more video sources as-is, transcribes each, and writes
a subtitled copy next to the original. Accepts .mp4 / .mov / .avi (anything
ffmpeg can read, really); the burned-in output is always .mp4.

Author: Chun Kang <kurapa@kurapa.com>
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

# Quiet the "unauthenticated requests to the HF Hub / set a HF_TOKEN" advisory
# that faster-whisper emits while downloading models. Set before any
# huggingface_hub import so it takes effect. Downloads work fine without a token.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

VENV_DIR = Path.home() / ".cache" / "mp4_subtitle" / "venv"
VENV_MARKER = "MP4SUB_IN_VENV"
PIP_PACKAGES: list[str] = ["faster-whisper", "silero-vad", "numpy"]

SETTINGS_FILE = Path.home() / ".config" / "mp4_subtitle" / "settings.json"

WHISPER_MODEL = "medium"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# Confidence Silero must have to call audio human speech, used to gate out
# Whisper hallucinations over music / noise (see detect_voice / keep_voiced).
# Higher -> stricter (drops quiet talking); lower -> catches quieter speech but
# lets more background noise through.
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 300

# Marker on output files, so re-running in a directory does not re-subtitle the
# subtitled copies it already produced.
SUBTITLED_SUFFIX = ".subtitled.mp4"
VTT_SUFFIX = ".vtt"

# Whisper's word-level DTW frequently stretches a cue's first word backward into
# the pause before it, so the subtitle pops up before the speaker starts. A real
# spoken word rarely lasts longer than this; when the first word appears to, the
# excess is almost always absorbed leading silence, so we bring the cue start
# forward to at most this many seconds before the first word's end. Raise it if
# legitimately long opening words get clipped early.
SUBTITLE_LEAD_CAP_SECONDS = 0.7


# ---------- bootstrap ----------

def _ensure_venv_and_reexec() -> None:
    if os.environ.get(VENV_MARKER) == "1":
        return
    if not VENV_DIR.exists():
        print(f"[mp4sub] creating venv at {VENV_DIR}")
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        venv.create(VENV_DIR, with_pip=True)
    venv_python = VENV_DIR / "bin" / "python3"
    if PIP_PACKAGES:
        pip = VENV_DIR / "bin" / "pip"
        missing = [p for p in PIP_PACKAGES if subprocess.run(
            [str(pip), "show", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode != 0]
        if missing:
            print(f"[mp4sub] installing: {', '.join(missing)}")
            subprocess.check_call([str(pip), "install", "-q", *missing])
    env = {**os.environ, VENV_MARKER: "1"}
    os.execve(str(venv_python), [str(venv_python), os.path.abspath(__file__), *sys.argv[1:]], env)


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if shutil.which("brew"):
        print("[mp4sub] installing ffmpeg via Homebrew")
        subprocess.check_call(["brew", "install", "ffmpeg"])
        return
    sys.exit("error: ffmpeg/ffprobe not found and Homebrew is unavailable; install ffmpeg manually")


# ---------- settings ----------

def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(settings: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError as e:
        print(f"[mp4sub] warning: could not save settings: {e}")


# ---------- input discovery ----------

def is_output(p: Path) -> bool:
    return p.name.endswith(SUBTITLED_SUFFIX)


def find_videos(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and not p.name.startswith(".")
        and not is_output(p)
    )


def is_readable(video: Path) -> bool:
    """True if ffprobe can parse the file's container."""
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


# ---------- prompts ----------

class UndecodableInput(Exception):
    """Raised when a prompt receives a keystroke stdin can't decode."""


def safe_input(prompt: str) -> str:
    """input() that survives undecodable keystrokes.

    A stray IME / partial multibyte byte makes the stdin reader raise
    UnicodeDecodeError. Re-raise as UndecodableInput so looping prompts can
    re-prompt and one-shot prompts can fall back to their default. EOFError
    still propagates.
    """
    try:
        return input(prompt)
    except UnicodeDecodeError:
        print()
        raise UndecodableInput


def prompt_whisper_model(default: str) -> str:
    valid = {"tiny", "base", "small", "medium", "large-v3"}
    while True:
        try:
            answer = safe_input(f"whisper model (tiny/base/small/medium/large-v3) [{default}]: ")
        except UndecodableInput:
            continue
        except EOFError:
            return default
        answer = answer.strip()
        if not answer:
            return default
        if answer not in valid:
            print(f"[mp4sub] unknown model: {answer!r}")
            continue
        return answer


# ---------- voice activity ----------

def detect_voice(video: Path, threshold: float) -> list[tuple[float, float]]:
    """Use Silero VAD to find ranges containing human speech (in seconds)."""
    import numpy as np
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True, check=True,
    )
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(samples)

    model = load_silero_vad()
    segments = get_speech_timestamps(
        wav, model,
        sampling_rate=16000,
        threshold=threshold,
        min_speech_duration_ms=VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in segments]


def keep_voiced(
    segments: list[tuple[float, float, str]],
    voice: list[tuple[float, float]],
) -> list[tuple[float, float, str]]:
    """Drop cues that don't overlap any Silero-detected human-voice range.

    Whisper's own VAD filter is permissive enough to still transcribe (and
    hallucinate) text over music or noise. This is a second, stricter gate: a
    cue survives only if its [start, end) overlaps a detected speech range, so
    subtitles appear only where there is actually a human voice.
    """
    kept: list[tuple[float, float, str]] = []
    for start, end, text in segments:
        if any(start < ve and end > vs for vs, ve in voice):
            kept.append((start, end, text))
    return kept


# ---------- transcription ----------

def transcribe(video: Path, model: "WhisperModel") -> list[tuple[float, float, str]]:
    """Run Whisper on the video and return [(start, end, text)] in seconds."""
    print(f"[mp4sub] transcribing {video.name}")
    # vad_filter=True: skip near-silence so Whisper does not hallucinate text
    #   there and its 30s decode window does not drift; faster-whisper maps the
    #   timestamps back to the real timeline.
    # word_timestamps=True: align segment start/end to actual word onsets (DTW)
    #   instead of coarse token-attention estimates.
    # condition_on_previous_text=False: stop one bad segment from cascading a
    #   timing/repetition error into everything after it.
    # These three together are what keep the cues in sync; see README.
    segments, _ = model.transcribe(
        str(video),
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        # Prefer the first/last word's timing when available: it tracks the
        # spoken audio more tightly than the segment-level estimate.
        words = getattr(seg, "words", None) or []
        start = float(words[0].start if words else seg.start)
        end = float(words[-1].end if words else seg.end)
        # Cap the first word's stretched-back onset (see SUBTITLE_LEAD_CAP_SECONDS):
        # if it "lasts" longer than a plausible spoken word, the excess is leading
        # silence the DTW absorbed, so move the cue start up to the cap.
        if words and (words[0].end - start) > SUBTITLE_LEAD_CAP_SECONDS:
            start = float(words[0].end) - SUBTITLE_LEAD_CAP_SECONDS
        # Never let a cue appear while the previous one is still on screen.
        if out and start < out[-1][1] < end:
            start = out[-1][1]
        out.append((start, end, text))
    return out


def _fmt_vtt_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(segments: list[tuple[float, float, str]], path: Path) -> None:
    blocks: list[str] = ["WEBVTT\n"]
    for i, (start, end, text) in enumerate(segments, 1):
        blocks.append(f"{i}\n{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def burn_subtitles(source: Path, vtt: Path, output: Path) -> None:
    # Run ffmpeg from the vtt's directory so we pass a bare filename and avoid
    # the subtitles filter's painful path escaping rules.
    #
    # force_style overrides the ASS style:
    #   FontName=Apple SD Gothic Neo -> one font covering BOTH Korean and Latin.
    #     Without this, libass falls back to a separate CJK font for Hangul runs,
    #     whose ascent/descent differ from the Latin font, so the BorderStyle=3
    #     box is sized per-run and its top/bottom edges step up and down where
    #     the script changes. Pinning one font keeps the box edges straight.
    #   BorderStyle=3            -> draw a box behind the text
    #   Outline=1                -> box padding around the text (px)
    #   Shadow=0                 -> no drop shadow
    #   OutlineColour=&H80000000 -> box colour in &HAABBGGRR (alpha 0x80 ~ 50%
    #                               opaque, pure black). Alpha is inverted in
    #                               ASS: 00 = opaque, FF = fully transparent.
    #   MarginV=40               -> lift subtitles ~40px up from the bottom.
    style = "FontName=Apple SD Gothic Neo,Outline=1,Shadow=0,OutlineColour=&H80000000,MarginV=40,BorderStyle=3"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
         "-i", str(source.resolve()),
         "-vf", f"subtitles={vtt.name}:force_style='{style}'",
         "-c:a", "copy",
         str(output.resolve())],
        check=True,
        cwd=str(vtt.parent),
    )


# ---------- driver ----------

def subtitle_one(video: Path, model: "WhisperModel") -> None:
    vtt_path = video.with_name(video.stem + VTT_SUFFIX)
    subtitled = video.with_name(video.stem + SUBTITLED_SUFFIX)

    reuse_vtt = False
    if vtt_path.exists():
        try:
            answer = safe_input(f"{vtt_path.name} exists. reuse it? [Y/n]: ")
        except (UndecodableInput, EOFError):
            answer = ""
        reuse_vtt = answer.strip().lower() != "n"

    if reuse_vtt:
        print(f"[mp4sub] reusing existing {vtt_path.name}")
    else:
        print(f"[mp4sub] detecting human voice (Silero VAD) in {video.name}")
        voice = detect_voice(video, VAD_THRESHOLD)
        if not voice:
            print(f"[mp4sub] no human voice detected in {video.name}; skipping")
            return
        segments = transcribe(video, model)
        before = len(segments)
        segments = keep_voiced(segments, voice)
        dropped = before - len(segments)
        if dropped:
            print(f"[mp4sub] dropped {dropped} cue(s) outside detected voice ranges")
        if not segments:
            print(f"[mp4sub] no speech transcribed in {video.name}; skipping")
            return
        write_vtt(segments, vtt_path)
        print(f"[mp4sub] wrote {vtt_path.name} ({len(segments)} cue(s))")

    print(f"[mp4sub] burning subtitles -> {subtitled.name}")
    burn_subtitles(video, vtt_path, subtitled)
    print(f"[mp4sub] done: {subtitled}")


def resolve_inputs(args: list[str]) -> list[Path]:
    if args:
        videos: list[Path] = []
        for a in args:
            p = Path(a)
            if not p.exists():
                sys.exit(f"error: no such file: {a}")
            if not p.is_file():
                sys.exit(f"error: not a file: {a}")
            videos.append(p)
        return videos
    found = find_videos(Path.cwd())
    if not found:
        sys.exit(
            "no video given and none found in the current directory.\n"
            "usage: mp4_subtitle [video ...]"
        )
    return found


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("-h", "--help")]
    if len(args) != len(sys.argv[1:]):
        print("usage: mp4_subtitle [video ...]")
        print("  Transcribe each video with Whisper and burn the subtitles in.")
        print("  With no arguments, processes every video in the current directory.")
        return

    videos = resolve_inputs(args)

    usable: list[Path] = []
    for v in videos:
        if is_readable(v):
            usable.append(v)
        else:
            print(f"[mp4sub] skipping unreadable file (truncated / unsupported?): {v.name}")
    if not usable:
        sys.exit("no readable video files found; every input failed to probe")

    print(f"[mp4sub] {len(usable)} video(s) to subtitle:")
    for v in usable:
        print(f"        {v.name}")

    settings = load_settings()
    whisper_model = prompt_whisper_model(settings.get("whisper_model", WHISPER_MODEL))
    settings["whisper_model"] = whisper_model
    save_settings(settings)

    from faster_whisper import WhisperModel
    print(f"[mp4sub] loading whisper model: {whisper_model} (downloads on first use)")
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    for v in usable:
        subtitle_one(v, model)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
