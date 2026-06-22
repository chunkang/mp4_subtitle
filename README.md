# mp4_subtitle

Transcribe a video with Whisper and burn the subtitles onto the original.
Works with `.mp4`, `.mov`, `.avi` (and other formats ffmpeg can read).

## What it does

Given one or more video sources, for each video it will:

1. **Detect human voice** with the Silero voice-activity detector.
2. **Transcribe** the audio with faster-whisper (word-level timestamps, VAD
   filtering), **clip each cue to the human-voice ranges** it overlaps, and write
   the result as a WebVTT `<name>.vtt` next to the source.
3. **Burn** those subtitles into a new `<name>.subtitled.mp4`, leaving the
   original untouched.

Step 1 is what keeps the subtitles to human speech only: Whisper's own VAD is
permissive enough to still hallucinate text over music or noise. Rather than an
all-or-nothing gate (which tended to lock onto the loudest speaker and drop
quieter or secondary voices), each cue's *text* is kept but only *shown* while
there is actually voice — so coverage stays wide while a cue with no voice at all
is dropped, and one straddling a long pause blanks out for the pause. See
[Keeping subtitles in sync](#keeping-subtitles-in-sync).

There is no silence removal here — each video is subtitled as-is. (For that, see
the sibling project `mp4_silence_removal`.) You can optionally **merge** every
video in a directory into one file before subtitling (see below).

If a `<name>.vtt` already exists it offers to reuse it, so you can hand-edit the
transcript and re-burn without re-running Whisper.

Long steps show progress as they run: transcription prints a live
percent-complete (segment time / total duration), and the subtitle burn shows
ffmpeg's running frame/time/speed stats.

## Requirements

- Python 3
- `ffmpeg` / `ffprobe` (auto-installed via Homebrew if missing)

On first run the script creates its own virtualenv under
`~/.cache/mp4_subtitle/venv`, installs its Python dependencies (`faster-whisper`,
`silero-vad`, `numpy`) into it, and re-execs itself — so you don't need to
install anything by hand. `demucs` is installed into the same venv only if you
pick the Demucs voice-boost method (see [Boosting the human
voice](#boosting-the-human-voice)).

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

Before processing you're prompted for a few options. The voice-boost answers are
saved to `~/.config/mp4_subtitle/settings.json` and reused as the default next
time:

- **Merge** (only when run with no filename and more than one video is present):
  whether to join every video into one before subtitling — see [Merging multiple
  videos](#merging-multiple-videos).
- **Whisper model** — `tiny`/`base`/`small`/`medium`/`large-v3`. Always defaults
  to `medium` (a good fit for this use); just press Enter to take it, or type
  another model for a one-off run.
- **Voice boost** — whether to turn the human voice up, by how much, and how —
  see [Boosting the human voice](#boosting-the-human-voice).

## Merging multiple videos

Run with **no filename** in a directory holding more than one video and it asks
whether to merge them all — in alphabetical order — into a single file, then
subtitles that. The result is `merged.subtitled.mp4` (with `merged.vtt`) in the
working directory; the individual files are left untouched.

Merging tries a lossless stream copy first (instant, the usual case when the
clips share a codec/resolution). If the inputs differ, it falls back to
re-encoding through ffmpeg's `concat` filter, scaling and padding every clip
onto the first clip's frame so they line up. Passing filenames explicitly always
subtitles them separately — the merge prompt only appears for the no-argument
case.

## Boosting the human voice

If you answer yes to the voice-boost prompt, you give a **multiplier** (`1` = no
change, e.g. `2` ≈ +6 dB) and pick a **method**:

- **Gated gain** (default, fast, no extra dependencies): applies the gain only
  over the detected speech ranges, via a single ffmpeg `volume` filter enabled
  on the cue intervals (already gated to human voice by Silero). It's
  *time*-selective — during a span where someone talks over music, it lifts the
  whole mix for that span, not the voice alone.
- **Demucs vocal isolation** (slow, best): separates the vocal stem with
  [Demucs](https://github.com/adefossez/demucs), amplifies only it, and remixes
  with the rest. This is *source*-selective — it raises the voice even where it
  overlaps music or noise — but it downloads a model on first use and is slow on
  CPU. `demucs` is pip-installed into the venv the first time you choose it.

When you pick Demucs, a follow-up prompt offers to **transcribe from the isolated
vocals** as well. Whisper otherwise reads the original mixed audio; feeding it the
clean voice-only stem can improve recognition over music or noise (loudness alone
wouldn't — Whisper normalizes levels — so this is offered only for Demucs). The
separation, the slow part, then runs once and is reused for both the transcript
and the boosted burn. If Demucs fails at any point it falls back to gated gain and
original-audio transcription, so a long run is never wasted.

The output is always encoded as **H.264 video + AAC audio**, and carries only
the burned-in text — any subtitle track in the source is dropped, since there's
no need for a separate (soft) subtitle track on top of the rendered subtitles.

## Subtitle style

Subtitles are burned with ffmpeg's `subtitles` (libass) filter as white text on
a semi-transparent black box lifted off the bottom edge:

| Setting | Value | Effect |
| --- | --- | --- |
| `FontName` | `Apple SD Gothic Neo` | one font covering Korean + Latin, so the box edges stay straight |
| `BorderStyle` | `3` | opaque box behind the text |
| `Outline` | `1` | box padding around the glyphs. On wrapped two-line cues this can make the per-line boxes overlap into a dark seam (libass has no line-spacing setting); lower toward `0` to remove the overlap |
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
starts. `transcribe()` clamps that: `SUBTITLE_LEAD_CAP_SECONDS` (0.3s) is the
most a subtitle may lead its first word, so when the first word "lasts" longer
than that, the excess is treated as absorbed leading silence and the cue start
is brought forward to the cap. Cues also never start while the previous one is
still on screen. **Lower** `SUBTITLE_LEAD_CAP_SECONDS` if cues still appear too
early; **raise** it if legitimately long opening words get clipped.

The on-screen *duration* is handled separately, by `clip_cues_to_voice()`, which
intersects each cue with the Silero voice ranges (padded by
`VAD_RANGE_PAD_SECONDS`, 0.2s, so word tails aren't clipped). This does two
things at once:

- **Coverage** — quiet or secondary speakers are detected at the low
  `VAD_THRESHOLD` (0.35) and their cues are clipped to their voiced moments
  rather than dropped wholesale, so subtitles don't lock onto just the loudest
  voice. **Lower** `VAD_THRESHOLD` for even more reach (at the cost of more noise
  slipping through); **raise** it to be stricter.
- **No subtitles over silence** — Whisper stretches a cue's last word into the
  following pause, and sometimes merges a mid-sentence pause into one cue. The
  clip closes the cue when the voice stops, and if a pause lasts at least
  `SUBTITLE_SILENCE_GAP_SECONDS` (2.0s) the subtitle blanks out for the pause.
  When a cue is split this way its **words are dealt to whichever side of the gap
  they were spoken on** (by their midpoint), so each part shows only its own
  words — never the whole line duplicated before and after the pause. Shorter
  pauses are bridged so cues don't flicker on every breath. **Lower** the gap to
  blank out sooner; **raise** it to keep subtitles up through longer pauses.

If subtitles still look off, try a larger Whisper model (e.g. `large-v3`).

## Output files

For an input `talk.mov`:

- `talk.vtt` — transcribed subtitles in WebVTT format
- `talk.subtitled.mp4` — the video with subtitles burned in

Files ending in `.subtitled.mp4` are skipped when scanning a directory, so
re-running won't subtitle its own output. When merging, the pair is named
`merged.vtt` / `merged.subtitled.mp4` instead.
