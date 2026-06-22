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
import re
import shutil
import subprocess
import sys
import tempfile
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
# Whisper hallucinations over music / noise (see detect_voice / clip_cues_to_voice).
# Higher -> stricter (drops quiet talking); lower -> catches quieter speech but
# lets more background noise through. Kept low so quiet / secondary speakers are
# still detected and their cues survive the gate, not just the dominant voice.
VAD_THRESHOLD = 0.35

# Pad each detected voice range by this when gating/clamping cues: Silero's
# ranges sit tight against the speech, so without a little slack a cue's edge can
# fall just outside a range and get clipped mid-word or dropped outright.
VAD_RANGE_PAD_SECONDS = 0.2

VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 300

# Marker on output files, so re-running in a directory does not re-subtitle the
# subtitled copies it already produced.
SUBTITLED_SUFFIX = ".subtitled.mp4"
VTT_SUFFIX = ".vtt"

# Output stem used when merging every video in a directory into one file:
# produces <MERGED_BASENAME>.subtitled.mp4 (and .vtt) in the working directory.
MERGED_BASENAME = "merged"

# Whisper's word-level DTW frequently stretches a cue's first word backward into
# the pause before it, so the subtitle pops up before the speaker starts. A real
# spoken word rarely lasts longer than this; when the first word appears to, the
# excess is almost always absorbed leading silence, so we bring the cue start
# forward to at most this many seconds before the first word's end -- i.e. this
# is the most a subtitle is allowed to lead its first word. Lower it if cues
# still appear too early; raise it if legitimately long opening words get clipped.
SUBTITLE_LEAD_CAP_SECONDS = 0.3

# A cue is shown only while there is voice (see clip_cues_to_voice). If the
# speaker pauses for at least this long mid-cue -- or Whisper's DTW stretches a
# word into the surrounding silence -- the subtitle blanks out for the gap rather
# than hanging on the screen over silence. Shorter pauses are bridged so cues
# don't flicker on every breath.
SUBTITLE_SILENCE_GAP_SECONDS = 2.0


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


# ---------- concatenation ----------

def _probe_dims(video: Path) -> tuple[int, int]:
    """Return the (width, height) of a video's first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def _concat_list_file(videos: list[Path]) -> str:
    """Build the body of a concat-demuxer list file for the given videos."""
    lines = []
    for v in videos:
        # The 'file' directive wraps the path in single quotes, so any single
        # quote inside the path itself must be escaped as '\'' to close, escape,
        # and reopen the quoting.
        path = str(v.resolve()).replace("'", "'\\''")
        lines.append(f"file '{path}'")
    return "\n".join(lines) + "\n"


def _concat_reencode(videos: list[Path], dest: Path) -> None:
    """Join videos by re-encoding through the concat filter.

    The concat filter demands every input share frame size, SAR, and audio
    layout, so scale+pad each clip onto the first clip's frame (letterboxing
    rather than stretching) and normalise fps/audio before concatenating. Used
    only when a lossless stream copy is impossible (mixed codecs/resolutions).
    Assumes every clip carries an audio stream, which holds for videos we are
    transcribing.
    """
    w, h = _probe_dims(videos[0])
    inputs: list[str] = []
    for v in videos:
        inputs += ["-i", str(v.resolve())]
    chains: list[str] = []
    for i in range(len(videos)):
        chains.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]"
        )
        chains.append(f"[{i}:a]aresample=async=1:first_pts=0[a{i}]")
    pairs = "".join(f"[v{i}][a{i}]" for i in range(len(videos)))
    graph = ";".join(chains) + f";{pairs}concat=n={len(videos)}:v=1:a=1[v][a]"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
         *inputs, "-filter_complex", graph,
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         str(dest)],
        check=True,
    )


def _stream_signature(video: Path) -> tuple:
    """Probe the stream params that must match for a lossless concat copy.

    We can't trust the concat demuxer's exit code to tell us whether a copy is
    safe: with -c copy it just appends packets and returns success even when the
    inputs have different codecs or resolutions, producing a file that glitches
    or fails to decode past the first clip. So we compare these params ourselves
    and only stream-copy when every input agrees. Returns (video_sig, audio_sig);
    either part is None when the stream is absent.
    """
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,pix_fmt,"
         "sample_aspect_ratio,sample_rate,channels",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout)
    v = a = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and v is None:
            v = (s.get("codec_name"), s.get("width"), s.get("height"),
                 s.get("pix_fmt"), s.get("sample_aspect_ratio"))
        elif s.get("codec_type") == "audio" and a is None:
            a = (s.get("codec_name"), s.get("sample_rate"), s.get("channels"))
    return (v, a)


def _concat_copy(videos: list[Path], dest: Path) -> bool:
    """Stream-copy concat via the demuxer. True on success. -sn drops any
    subtitle streams the inputs carry (the merged file is only an intermediate)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(_concat_list_file(videos))
        list_path = f.name
    try:
        return subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", list_path,
             "-c", "copy", "-sn", str(dest)],
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    finally:
        os.unlink(list_path)


def concat_videos(videos: list[Path], dest: Path) -> None:
    """Join videos (in the given order) into a single file at dest.

    Stream-copies (no re-encode) when every input shares the same video+audio
    params -- the common case for clips from one source/encoder. Otherwise the
    copy would yield a broken mid-stream codec/resolution switch, so re-encode
    instead (see _concat_reencode). The subtitle burn re-encodes the video once
    afterward regardless, so a successful copy here means merging is free.
    """
    print(f"[mp4sub] merging {len(videos)} video(s) -> {dest.name}")
    sigs = {_stream_signature(v) for v in videos}
    uniform = len(sigs) == 1 and next(iter(sigs))[0] is not None
    if uniform and _concat_copy(videos, dest):
        print("[mp4sub] merged by stream copy (no re-encode)")
        return
    if uniform:
        print("[mp4sub] stream copy failed unexpectedly; re-encoding to merge")
    else:
        print("[mp4sub] inputs differ (codec/size/format); re-encoding to merge")
    _concat_reencode(videos, dest)


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


def prompt_merge(count: int) -> bool:
    """Ask whether to merge every discovered video into one subtitled file.

    Only reached when no filename was given on the command line and more than
    one video was found. Default (empty answer) is No: subtitle each separately,
    matching the historical behaviour.
    """
    while True:
        try:
            answer = safe_input(
                f"no filename given; {count} videos found. merge all into ONE "
                f"subtitled video, in alphabetical order? [y/N]: "
            )
        except UndecodableInput:
            continue
        except EOFError:
            return False
        answer = answer.strip().lower()
        if answer in ("", "n", "no"):
            return False
        if answer in ("y", "yes"):
            return True
        print("[mp4sub] please answer y or n")


def prompt_volume(default: float) -> tuple[float, str]:
    """Ask whether to turn the human voice up, by how much, and how.

    Returns (multiplier, method). multiplier 1.0 leaves the audio untouched
    (method "none") and lets the burn keep stream-copying it. Otherwise method
    is "gated" (boost only over detected speech, fast) or "demucs" (isolate the
    vocal stem and boost just that, slow but works under music/noise).
    """
    try:
        answer = safe_input("turn the human voice volume up? [y/N]: ")
    except (UndecodableInput, EOFError):
        return 1.0, "none"
    if answer.strip().lower() not in ("y", "yes"):
        return 1.0, "none"

    volume = default
    while True:
        try:
            raw = safe_input(f"volume multiplier (1 = no change) [{default:g}]: ")
        except UndecodableInput:
            continue
        except EOFError:
            break
        raw = raw.strip()
        if not raw:
            break
        try:
            value = float(raw)
        except ValueError:
            print(f"[mp4sub] not a number: {raw!r}")
            continue
        if value <= 0:
            print("[mp4sub] volume must be greater than 0")
            continue
        volume = value
        break

    if volume == 1.0:
        return 1.0, "none"
    return volume, prompt_volume_method()


def prompt_volume_method() -> str:
    """Pick how the voice boost is applied.

    "gated" (default): one ffmpeg volume filter enabled only over the speech
    ranges -- fast, no new dependencies, but lifts whatever else plays under the
    voice during those spans. "demucs": split off the vocal stem and boost only
    it -- isolates the voice even over music/noise, but downloads a model and is
    slow on CPU.
    """
    while True:
        try:
            answer = safe_input(
                "boost method - (g)ated gain over speech [fast, default] or "
                "(d)emucs vocal isolation [slow, best]? [G/d]: "
            )
        except UndecodableInput:
            continue
        except EOFError:
            return "gated"
        answer = answer.strip().lower()
        if answer in ("", "g", "gated"):
            return "gated"
        if answer in ("d", "demucs"):
            return "demucs"
        print("[mp4sub] please answer g or d")


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


def clip_cues_to_voice(
    segments: list[tuple[float, float, str, list[tuple[float, float, str]]]],
    voice: list[tuple[float, float]],
) -> tuple[list[tuple[float, float, str]], int]:
    """Keep each cue's text but show it only while someone is actually speaking.

    For every cue we intersect its span with the Silero voice ranges (padded by
    VAD_RANGE_PAD_SECONDS) and bridge pauses shorter than
    SUBTITLE_SILENCE_GAP_SECONDS, leaving one display interval per voiced stretch.
    A cue that fits in a single stretch is shown whole; one that straddles a long
    pause is split, and crucially its *words are dealt to the side of the gap
    they were spoken on* (by their midpoint) -- so each interval shows only its
    own words, never the whole text duplicated before and after the pause. Cues
    with no voiced overlap at all (Whisper hallucinations over music / noise) are
    dropped.

    This serves both goals at once: coverage stays wide -- quiet or secondary
    speakers are clamped to their voiced moments rather than discarded wholesale
    (a stricter all-or-nothing gate is what made it feel locked to one speaker) --
    while subtitles never linger on screen over silence. Returns (cues, dropped).
    """
    pad = VAD_RANGE_PAD_SECONDS
    out: list[tuple[float, float, str]] = []
    dropped = 0
    for start, end, text, words in segments:
        covered: list[tuple[float, float]] = []
        for vs, ve in voice:
            a, b = max(start, vs - pad), min(end, ve + pad)
            if a < b:
                covered.append((a, b))
        if not covered:
            dropped += 1
            continue
        intervals = merge_ranges(covered, SUBTITLE_SILENCE_GAP_SECONDS)
        if len(intervals) == 1 or not words:
            # One voiced stretch (or no word timing to split by): show whole text
            # across the speaking span.
            out.append((intervals[0][0], intervals[-1][1], text))
            continue
        # Multiple stretches separated by long pauses: deal each word to an
        # interval by its midpoint, splitting at the middle of each gap so every
        # word lands on exactly one side and nothing is duplicated.
        bounds = [(intervals[i][1] + intervals[i + 1][0]) / 2
                  for i in range(len(intervals) - 1)]
        buckets: list[list[str]] = [[] for _ in intervals]
        for ws, we, wtext in words:
            mid = (ws + we) / 2
            idx = next((i for i, bnd in enumerate(bounds) if mid < bnd), len(intervals) - 1)
            buckets[idx].append(wtext)
        for (a, b), bucket in zip(intervals, buckets):
            piece = "".join(bucket).strip()
            if piece:
                out.append((a, b, piece))
    out.sort(key=lambda c: c[0])
    return out, dropped


# ---------- transcription ----------

def _fmt_clock(t: float) -> str:
    """Format seconds as M:SS (or H:MM:SS past an hour) for progress lines."""
    t = max(0, int(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
    # faster-whisper decodes lazily: segments arrive only as we iterate, so the
    # loop below is where the real work happens. info.duration is the audio
    # length Whisper sees, which lets us turn each segment's end time into a
    # rough percent-complete that ticks up on one rewritten line.
    segments, info = model.transcribe(
        str(video),
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    total = float(getattr(info, "duration", 0.0) or 0.0)
    # Each cue carries its word list (start, end, text) too, so that if a cue is
    # later split across a long pause its words can be dealt to the right side
    # instead of the whole text being duplicated. See clip_cues_to_voice.
    out: list[tuple[float, float, str, list[tuple[float, float, str]]]] = []
    for seg in segments:
        if total:
            pct = min(100.0, seg.end / total * 100.0)
            print(
                f"\r[mp4sub] transcribing... {pct:5.1f}% "
                f"({_fmt_clock(seg.end)} / {_fmt_clock(total)})",
                end="", flush=True,
            )
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
        word_list = [(float(w.start), float(w.end), w.word) for w in words]
        out.append((start, end, text, word_list))
    if total:
        print()  # end the rewritten progress line
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


# ---------- voice boost ----------

_VTT_RANGE_RE = re.compile(
    r"(\d+):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})\.(\d{3})"
)


def read_vtt_ranges(path: Path) -> list[tuple[float, float]]:
    """Parse a WEBVTT file into its cue [start, end) times in seconds."""
    ranges: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _VTT_RANGE_RE.search(line)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
        ranges.append((
            h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
            h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
        ))
    return ranges


def merge_ranges(ranges: list[tuple[float, float]], gap: float = 0.2) -> list[tuple[float, float]]:
    """Coalesce overlapping ranges, and ones separated by < gap seconds.

    Fewer ranges means a shorter enable expression and fewer abrupt gain toggles
    (each toggle is a potential click), at the cost of also lifting the brief
    pauses bridged by the gap.
    """
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def gated_volume_filter(ranges: list[tuple[float, float]], volume: float) -> str:
    """Build one ffmpeg `volume` filter that is active only over `ranges`.

    The enable expression ORs the intervals by summing between() terms: each is
    1 inside its interval and 0 outside, so the sum is non-zero (truthy to
    ffmpeg) exactly when t falls in some range.
    """
    expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in ranges)
    return f"volume=volume={volume:g}:enable='{expr}'"


def _ensure_demucs() -> None:
    try:
        import demucs  # noqa: F401
        return
    except ImportError:
        pass
    print("[mp4sub] installing demucs (first use; reuses the existing torch)")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "demucs"])


def boost_voice_demucs(source: Path, volume: float, workdir: Path) -> Path:
    """Isolate the vocal stem with Demucs, amplify it, remix, return the wav.

    Demucs (htdemucs, --two-stems) splits the audio into vocals + everything
    else; we scale only the vocals and sum the stems back (amix normalize=0, so
    levels are preserved rather than averaged). The result is the original mix
    with just the voice turned up, even where it overlapped music or noise.
    """
    _ensure_demucs()
    audio = workdir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(source.resolve()), "-vn", str(audio)],
        check=True,
    )
    print("[mp4sub] separating vocals with Demucs (slow on CPU)")
    out = workdir / "demucs"
    # Write mp3 stems, not wav: demucs saves wav/flac via torchaudio.save(),
    # which on torchaudio >= 2.9 dispatches to the optional torchcodec package
    # (absent here, and with no wheels for new Pythons). Its mp3 path instead
    # uses lameenc (a demucs dependency), sidestepping that entirely. The stems
    # are throwaway intermediates and the final audio is AAC anyway, so the lossy
    # mp3 step costs nothing in practice.
    subprocess.run(
        [sys.executable, "-m", "demucs", "--two-stems", "vocals",
         "--mp3", "--mp3-bitrate", "320",
         "-o", str(out), str(audio)],
        check=True,
    )
    vocals = next(iter(out.glob("**/vocals.mp3")), None)
    if vocals is None:
        raise RuntimeError("demucs produced no vocals stem")
    no_vocals = vocals.with_name("no_vocals.mp3")
    mixed = workdir / "boosted.wav"
    # Scale the vocal stem, sum it back with the rest (normalize=0 keeps levels
    # instead of averaging), then run a peak limiter: at a high multiplier the
    # boosted vocals + backing track would otherwise exceed full scale and
    # hard-clip into distortion. alimiter catches those peaks so the voice comes
    # out loud but clean.
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(vocals), "-i", str(no_vocals),
         "-filter_complex",
         f"[0:a]volume={volume:g}[v];[v][1:a]amix=inputs=2:normalize=0[m];"
         f"[m]alimiter=limit=0.95[a]",
         "-map", "[a]", str(mixed)],
        check=True,
    )
    return mixed


def burn_subtitles(source: Path, vtt: Path, output: Path, volume: float = 1.0,
                   gain_ranges: list[tuple[float, float]] | None = None,
                   audio_source: Path | None = None) -> None:
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
    #   Outline=1                -> box padding around the text (px). Note: on a
    #     wrapped two-line cue this padding can push the per-line boxes tall
    #     enough to overlap, showing a darker seam where two 50%-opaque boxes
    #     stack (libass has no line-spacing setting to separate them). Lower it
    #     toward 0 to remove that overlap, at the cost of tighter padding.
    #   Shadow=0                 -> no drop shadow
    #   OutlineColour=&H80000000 -> box colour in &HAABBGGRR (alpha 0x80 ~ 50%
    #                               opaque, pure black). Alpha is inverted in
    #                               ASS: 00 = opaque, FF = fully transparent.
    #   MarginV=40               -> lift subtitles ~40px up from the bottom.
    style = "FontName=Apple SD Gothic Neo,Outline=1,Shadow=0,OutlineColour=&H80000000,MarginV=40,BorderStyle=3"
    sub_vf = f"subtitles={vtt.name}:force_style='{style}'"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-stats",
           "-i", str(source.resolve())]
    if audio_source is not None:
        # Demucs already applied the gain when it remixed the stems; mux that
        # processed audio over the subtitled video (video from input 0, audio 1).
        cmd += ["-i", str(audio_source.resolve()),
                "-map", "0:v", "-map", "1:a", "-vf", sub_vf]
    elif gain_ranges:
        # Time-gated gain: one volume filter active only over the speech ranges.
        cmd += ["-vf", sub_vf, "-af", gated_volume_filter(gain_ranges, volume)]
    else:
        cmd += ["-vf", sub_vf]
    # The subtitles filter always re-renders the video, so encode H.264 + AAC.
    # -sn drops any subtitle stream the source carries: the output holds only the
    # burned-in text, never a separate (soft) subtitle track.
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-sn",
            str(output.resolve())]
    subprocess.run(cmd, check=True, cwd=str(vtt.parent))


# ---------- driver ----------

def subtitle_one(video: Path, model: "WhisperModel", out_base: Path | None = None,
                 volume: float = 1.0, method: str = "none") -> None:
    # video is the file we transcribe and burn from; out_base decides where the
    # .vtt / .subtitled.mp4 land and what they are named. They differ when video
    # is a temp merged source but the outputs belong in the working directory.
    out_base = out_base or video
    vtt_path = out_base.with_name(out_base.stem + VTT_SUFFIX)
    subtitled = out_base.with_name(out_base.stem + SUBTITLED_SUFFIX)

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
        # Keep the text but show each cue only over actual voice: widens coverage
        # (quiet/secondary speakers survive) and keeps subtitles off the screen
        # during silence.
        segments, dropped = clip_cues_to_voice(segments, voice)
        if dropped:
            print(f"[mp4sub] dropped {dropped} cue(s) with no detected voice (noise/music)")
        if not segments:
            print(f"[mp4sub] no speech transcribed in {video.name}; skipping")
            return
        write_vtt(segments, vtt_path)
        print(f"[mp4sub] wrote {vtt_path.name} ({len(segments)} cue(s))")

    boost = volume != 1.0 and method in ("gated", "demucs")
    gain = "" if not boost else f" (voice volume x{volume:g}, {method})"
    print(f"[mp4sub] burning subtitles{gain} -> {subtitled.name}")

    burned = False
    if boost and method == "demucs":
        # Hold the temp dir (and stems) open across the burn that consumes them.
        # If Demucs fails, don't throw away the (possibly very long) transcription:
        # warn and fall through to gated gain instead.
        try:
            with tempfile.TemporaryDirectory(prefix="mp4sub_demucs_") as td:
                processed = boost_voice_demucs(video, volume, Path(td))
                burn_subtitles(video, vtt_path, subtitled, audio_source=processed)
            burned = True
        except Exception as e:
            print(f"[mp4sub] demucs voice boost failed ({e}); falling back to gated gain")
            method = "gated"

    if not burned and boost and method == "gated":
        # Gate the gain to the speech ranges. The cues are already clipped to
        # voice (clip_cues_to_voice), so reuse them: from the freshly built
        # segments, or by parsing the vtt when reusing an existing one.
        cue_ranges = ([(s, e) for s, e, _ in segments] if not reuse_vtt
                      else read_vtt_ranges(vtt_path))
        burn_subtitles(video, vtt_path, subtitled, volume=volume,
                       gain_ranges=merge_ranges(cue_ranges))
        burned = True

    if not burned:
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

    from_args = bool(args)
    videos = resolve_inputs(args)

    usable: list[Path] = []
    for v in videos:
        if is_readable(v):
            usable.append(v)
        else:
            print(f"[mp4sub] skipping unreadable file (truncated / unsupported?): {v.name}")
    if not usable:
        sys.exit("no readable video files found; every input failed to probe")

    # Only offer the merge when the user named no file and there is more than one
    # video to combine; ask before loading the (slow) Whisper model. usable is in
    # alphabetical order (find_videos sorts), which is the merge order.
    merge = not from_args and len(usable) > 1 and prompt_merge(len(usable))

    print(f"[mp4sub] {len(usable)} video(s) to subtitle:")
    for v in usable:
        print(f"        {v.name}")
    if merge:
        print(f"[mp4sub] mode: merge all into one -> {MERGED_BASENAME}{SUBTITLED_SUFFIX}")

    settings = load_settings()
    # WHISPER_MODEL (medium) is the standing default every run -- it suits this
    # use well -- so the prompt always offers it rather than remembering a one-off
    # choice. You can still type another model for a single run at the prompt.
    whisper_model = prompt_whisper_model(WHISPER_MODEL)
    voice_volume, voice_method = prompt_volume(float(settings.get("voice_volume", 1.0)))
    settings["voice_volume"] = voice_volume
    settings["voice_method"] = voice_method
    save_settings(settings)

    from faster_whisper import WhisperModel
    print(f"[mp4sub] loading whisper model: {whisper_model} (downloads on first use)")
    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    if merge:
        # Concatenate into a throwaway source in a temp dir, then transcribe and
        # burn from it while writing the .vtt / .subtitled.mp4 into the working
        # directory. The temp dir (and raw merged source) is removed on exit.
        with tempfile.TemporaryDirectory(prefix="mp4sub_") as td:
            merged_src = Path(td) / "merged_source.mp4"
            concat_videos(usable, merged_src)
            subtitle_one(merged_src, model, Path.cwd() / MERGED_BASENAME,
                         voice_volume, voice_method)
    else:
        for v in usable:
            subtitle_one(v, model, volume=voice_volume, method=voice_method)


if __name__ == "__main__":
    _ensure_venv_and_reexec()
    _ensure_ffmpeg()
    main()
