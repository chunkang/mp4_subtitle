# mp4_subtitle

Transcribe a video with Whisper and burn the subtitles onto the original.
Works with `.mp4`, `.mov`, `.avi` (and other formats ffmpeg can read).

## What it does

Given one or more video sources, for each video it will:

1. **Detect human voice** with the Silero voice-activity detector.
2. **Transcribe** the audio with faster-whisper (word-level timestamps, VAD
   filtering), **keep only the cues that fall within a detected human-voice
   range**, and write the survivors as a WebVTT `<name>.vtt` next to the source.
3. **Burn** those subtitles into a new `<name>.subtitled.mp4`, leaving the
   original untouched.

Step 1 is what keeps the subtitles to human speech only: Whisper's own VAD is
permissive enough to still hallucinate text over music or noise, so any cue that
doesn't overlap a Silero-detected voice range is dropped before the `.vtt` is
written.

There is no consolidation or silence removal here — the video is subtitled
as-is. (For that, see the sibling project `mp4_silence_removal`.)

If a `<name>.vtt` already exists it offers to reuse it, so you can hand-edit the
transcript and re-burn without re-running Whisper.

## Requirements

- Python 3
- `ffmpeg` / `ffprobe` (auto-installed via Homebrew if missing)

On first run the script creates its own virtualenv under
`~/.cache/mp4_subtitle/venv`, installs its Python dependencies (`faster-whisper`,
`silero-vad`, `numpy`) into it, and re-execs itself — so you don't need to
install anything by hand.

## Install

```sh
./_init.sh
```

This installs the script to `~/bin/mp4_subtitle`. Make sure `~/bin` is on your
`PATH`.

## Usage

```sh
# subtitle specific files
mp4_subtitle talk.mov interview.mp4

# or run with no arguments to subtitle every video in the current directory
cd /path/to/videos
mp4_subtitle
```

You're prompted once for the **Whisper model**
(`tiny`/`base`/`small`/`medium`/`large-v3`, default `medium`). Your answer is
saved to `~/.config/mp4_subtitle/settings.json` and used as the default next
time.

## Subtitle style

Subtitles are burned with ffmpeg's `subtitles` (libass) filter as white text on
a semi-transparent black box lifted off the bottom edge:

| Setting | Value | Effect |
| --- | --- | --- |
| `FontName` | `Apple SD Gothic Neo` | one font covering Korean + Latin, so the box edges stay straight |
| `BorderStyle` | `3` | opaque box behind the text |
| `Outline` | `1` | box padding around the glyphs |
| `Shadow` | `0` | no drop shadow |
| `OutlineColour` | `&H80000000` | 50%-opaque black box |
| `MarginV` | `40` | lift subtitles ~40px off the bottom |

The style lives in `burn_subtitles()` if you want to tweak it. To use a
different font, pick any with full Korean + Latin coverage (e.g. `NanumGothic`
or `Noto Sans CJK KR`).

## Keeping subtitles in sync

Cue timing comes entirely from Whisper. `transcribe()` runs it with VAD
filtering, word-level timestamps, and `condition_on_previous_text=False`, and
takes each cue's start/end from its first and last word rather than the coarser
segment estimate.

Whisper's word-level alignment tends to stretch a cue's *first* word backward
into the pause before it, making the subtitle pop up a beat before the speaker
starts. `transcribe()` clamps that: when the first word "lasts" longer than
`SUBTITLE_LEAD_CAP_SECONDS` (0.7s), the excess is treated as absorbed leading
silence and the cue start is brought forward to the cap. Cues also never start
while the previous one is still on screen. Raise `SUBTITLE_LEAD_CAP_SECONDS` if
legitimately long opening words get clipped early. If subtitles still look off,
try a larger Whisper model (e.g. `large-v3`).

## Output files

For an input `talk.mov`:

- `talk.vtt` — transcribed subtitles in WebVTT format
- `talk.subtitled.mp4` — the video with subtitles burned in

Files ending in `.subtitled.mp4` are skipped when scanning a directory, so
re-running won't subtitle its own output.
