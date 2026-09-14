# AI Tutor

A real-time voice tutor that runs entirely on one laptop. You talk to it through your mic, it teaches out loud
(grounded in your course notes), and it watches your webcam for focus: look at your phone or walk away for too
long and it warmly pulls you back. Video frames are processed in memory and immediately discarded; nothing is
saved or sent anywhere.

Only the LLM call (Groq) needs the network. Speech-to-text falls back to a local model, and text-to-speech,
speech detection, retrieval and focus detection all run locally.

**Setting this up for the first time?** Follow **[SETUP.md](SETUP.md)**: step-by-step, plain-language instructions for
Mac, Windows and Linux.

## Requirements

- macOS on Apple Silicon (this is what it was built and tested on)
- **Python 3.12** (Kokoro does not support 3.13+; mediapipe 1.0.x crashes on macOS, so 0.10.35 is pinned)
- Homebrew `espeak-ng` (Kokoro's fallback for words it doesn't know)
- Webcam, microphone (earphones with a mic recommended), a [Groq](https://console.groq.com) API key

## Setup

Quick version for macOS. For Windows, Linux or a slower walkthrough, see [SETUP.md](SETUP.md).

```bash
brew install python@3.12 espeak-ng
```

```bash
"$(brew --prefix)/bin/python3.12" -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put your key in `.env` (`GROQ_API_KEY=...`). Never commit `.env`.

Then download every model once, while online (~600 MB: Face Landmarker, MiniLM embeddings, spaCy English,
Kokoro-82M, faster-whisper base, Silero VAD). After this the app loads models with Hugging Face forced offline,
so venue wifi can't stall startup.

```bash
.venv/bin/python scripts/download_models.py
```

### macOS permissions

Run the app from **Terminal.app** (the `scripts/*.command` launchers open it there). The first run asks for:

- **Camera** and **Microphone** for Terminal: click Allow. If the log says `MIC IS PERFECTLY SILENT`, enable
  Terminal under System Settings > Privacy & Security > Microphone and restart.
- **Input Monitoring** for Terminal (global hotkeys). The log may still warn that Terminal isn't
  *Accessibility*-trusted; that warning is harmless if `scripts/hotkey_check.py` reports
  `global hotkeys work from this terminal`.

## Running

```bash
open scripts/run_tutor.command
```

or, from a terminal that already has camera/mic permission:

```bash
.venv/bin/python main.py
```

Look at the screen for the first 2 seconds (focus calibration), wait for `READY`, and start talking.

| Action | Global hotkey | Terminal (type + Enter) |
|---|---|---|
| Interrupt the tutor | Ctrl+Option+I | `i` |
| Force a focus intervention now | Ctrl+Option+F | `f` |
| Suppress / re-enable automatic interventions | Ctrl+Option+S | `s` |
| Recalibrate "looking at the screen" | Ctrl+Option+C | `c` |
| Microphone on / off | Ctrl+Option+M | `m` |
| Camera on / off | Ctrl+Option+V | `v` |
| Quit | | `q` |

You can also interrupt just by talking over the tutor. Logs go to the console and `logs/tutor.log`, including
the focus score every 2 s, every state transition, and a per-turn `LATENCY` / `E2E` line.

### The page

`main.py` also opens a local page at **http://127.0.0.1:8765** (`UI_PORT` in `config.py`; set
`UI_OPEN_BROWSER = False` to stop it opening automatically). It's a dark, full-screen view that shows:

- a **3D tutor avatar** (three.js, built from primitives, no model files) whose mouth moves with the loudness of
  the audio actually playing, which leans in while you talk, looks up while thinking, turns concerned when your
  focus drops, and follows your mouse. If WebGL isn't available, a glowing orb that pulses with the voice replaces it,
- a live **focus card** (score, threshold marker, focused / drifting / distracted and why),
- what the app is doing (listening, you're speaking, thinking, tutor speaking) and whether auto nudges are paused,
- the conversation, with the tutor's text streaming in and equations rendered by KaTeX,
- a chip whenever a focus nudge fires (only the reason is shown, never the hidden prompt),
- a **Meet-style control bar**: microphone and camera toggles (red when off), Interrupt, Force nudge,
  Pause/Resume auto nudges, Recalibrate, and a shield button for the privacy panel,
- a **microphone and speaker menu** (the arrow next to the mic button): pick any combination, such as headphones for
  sound with the laptop's mic, with a live mic level and a Test sound. The switch happens instantly, speech already
  playing moves to the new speaker, and the choice is remembered on this computer in `local_settings.json`,
- your **self-view** in a floating tile: drag it and it snaps to a corner, resize it from its inner corner, or
  minimize it to a pill. The browser opens the camera for this preview itself, so that video never reaches Python,
  the network or the AI. The tile remembers where you left it,
- a **privacy panel** (shield button) listing what is sent to the AI service and what stays on this computer, with
  live status for the camera, the microphone and where transcription is happening.

Camera off and mic off mean off: the device is released. With the camera off, focus tracking and automatic nudges
pause (Force nudge still works). With the mic off, nothing is heard or transcribed.

The page reconnects on its own and replays the conversation after a reload. It only receives scores and text,
never video. KaTeX and three.js are vendored in `ui/static/`, so equations and the avatar work offline. The server listens on
localhost only; if the port is busy, the tutor keeps running without the page and logs an error.

## Adding course materials

Drop `.md`, `.txt` or `.pdf` files into `./materials/`. They are chunked, embedded and stored in a local
ChromaDB (`./.chroma/`) the next time the app starts; only new or changed files are re-indexed, and deleted
files are removed. A sample calculus file is included; replace it with your own.

Tips:

- Markdown headings (`#`, `##`) become sections. The matched section name is used as the "current topic" in focus
  interventions, so descriptive headings help.
- Write math as LaTeX in `$...$`; the tutor reads it aloud as words ("d y by d x equals ...").
- To test retrieval without voice: `.venv/bin/python scripts/text_chat.py` (type `/reindex` after editing files).

## Tuning focus detection

All focus settings are at the top of `config.py`. The ones you'll touch at a venue:

| Setting | Default | What it does |
|---|---|---|
| `FOCUS_THRESHOLD` | 50 | Score (0–100) below this counts as unfocused |
| `DISTRACTED_SECONDS` | 10 | How long the score must stay low before an intervention |
| `ABSENT_SECONDS` | 10 | No face this long counts as walked away |
| `PITCH_DOWN_OK_DEG` / `PITCH_DOWN_MAX_DEG` | 10 / 20 | Head tilt where the "looking down" penalty starts / maxes out |
| `YAW_OK_DEG` / `YAW_MAX_DEG` | 20 / 40 | Same for turning away |
| `EAR_CLOSED` | 0.12 | Eye openness below this counts as closed |
| `INTERVENTION_COOLDOWN_S` | 45 | Minimum gap between spoken interventions |

Measured on the dev laptop: normal posture sat at −7° to −12° of head pitch, phone glances at −14° to −26°, eyes
open 0.21–0.30 and closed ~0.03.

To retune:

1. `open scripts/run_focus_check.command`: a spoken, 2-minute guided check (look down, turn, close eyes, leave)
   that prints PASS/FAIL per step with the measured angles to `logs/focus_check.txt`.
2. `.venv/bin/python scripts/focus_debug.py`: a live overlay of score, head pose and eye openness
   (`c` recalibrates, `q` quits).
3. Change the values above and restart.

If venue lighting breaks detection, press **Ctrl+Option+S** to stop automatic interventions and use
**Ctrl+Option+F** to trigger the moment by hand. Forcing works even if the camera fails entirely.

## Tuning voice

Also in `config.py`:

- Devices: pick them in the page's microphone/speaker menu. `INPUT_DEVICE` / `OUTPUT_DEVICE` only set the starting
  choice on a computer that hasn't picked yet (None = system default). Devices are matched by name, so plugging
  something in can't silently switch them. List devices with `.venv/bin/python -m sounddevice`.
- `END_SILENCE_MS` (300): silence that ends your turn. Every millisecond here adds latency; raise it if the
  tutor cuts you off mid-sentence.
- `MIN_SPEECH_RMS` (0.02): loudness gate that ignores background voices. Every utterance's loudness is logged
  (`loudness p90 rms=...`, and `ignored quiet utterance` for rejected ones); set it between the room and your voice.
- Barge-in (`BARGE_IN_*`, `ECHO_TAIL_MS`): defaults assume **earphones**. On laptop or external speakers, use
  the stricter values in the comment above them so the tutor doesn't interrupt itself.

## What happens when things fail

- **No network / Groq STT down:** the local faster-whisper transcript (computed in parallel) is used with no
  extra delay, and Groq is skipped for 30 s.
- **Groq rate limit (429):** that turn is retried on `LLM_FALLBACK_MODEL` (gpt-oss-20b), which has its own quota.
- **LLM unreachable:** the tutor says "Sorry, I lost my connection for a moment" and keeps listening.
- **Camera missing or blocked:** focus score holds at 100 (no false interventions); the force hotkey still works.
- Every failure is logged loudly (`ERROR` / uppercase message); none of them crash the app.

## Project layout

```
main.py                 full app: voice loop + focus detector + interventions + hotkeys
config.py               every tunable
local_settings.json     microphone/speaker picked in the page (per computer, not in git)
tutor/
  focus.py              MediaPipe focus detector and rolling score (runs on its own thread)
  interventions.py      decides when to redirect a distracted student; force/suppress overrides
  voice.py              conversation loop: STT -> tutor -> sentence chunks -> TTS, interruption
  tutor.py              RAG context + history + streaming LLM
  llm.py                Groq client (streaming, cancellable, rate-limit fallback)
  rag.py                ChromaDB + MiniLM retrieval over ./materials
  prompts.py            system prompt and hidden app messages
  speech_text.py        sentence chunking and LaTeX-to-speech
  hotkeys.py            global hotkeys (physical key matching, works with Option)
  state.py              state shared between the loops
  ui_server.py          FastAPI + WebSocket server for the page (background thread)
  audio/                mic + VAD, STT, TTS, speaker, device lookup and live switching
ui/
  index.html            the page
  static/               app.js, app.css, avatar.js (3D tutor), selfview.js, vendored KaTeX and three.js
scripts/
  download_models.py    cache all models for offline use
  focus_debug.py        live focus overlay (stage 1)
  focus_check.py        spoken guided focus check with PASS/FAIL report
  text_chat.py          terminal text tutor (stage 2)
  voice_chat.py         voice tutor without the camera (stage 3)
  hotkey_check.py       verifies global hotkeys from the current terminal
  *.command             Terminal.app launchers for the scripts above
materials/              your course notes
logs/                   run logs and check reports
```

## Publishing to GitHub

`.gitignore` already keeps out everything big or private: `.venv/` (~2 GB), `.env` (your API key), downloaded
models, the `.chroma` index, `logs/` and `local_settings.json`. What's left is about 3 MB.

```bash
git init
```

```bash
git add .
```

```bash
git status
```

Check that nothing under `.venv`, `.env`, `models` or `logs` is listed, then commit:

```bash
git commit -m "AI tutor"
```

Create an empty repository on GitHub (without a README), and push to it with the commands GitHub shows, for example:

```bash
git branch -M main
```

```bash
git remote add origin https://github.com/sneha822/ai-tutor.git
```

```bash
git push -u origin main
```

If `.env` is ever committed by mistake, delete that key in the Groq console and create a new one: removing the file
afterwards doesn't remove it from git history.

## Known limitations

- End-to-end latency (you stop speaking → tutor starts) measured a median of ~1.5 s on earphones. Most turns
  over target were Groq taking 1–2 s to start its reply.
- The tutor speaks English only (Kokoro's English voice); non-Latin text in replies is skipped.
- Conversation memory lasts for one session only.
