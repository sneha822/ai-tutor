# AI Tutor

A real-time voice AI that runs on one laptop. Pick a subject to study with a tutor, or go off topic with a friend,
a supportive listener or a normal chat. You talk through your mic, it answers out loud with its feelings showing on an
animated 3D face, and in tutor mode it watches your webcam for focus: look at your phone or walk away and it pulls you
back, a little firmer each time it keeps happening. Video frames are processed in memory and immediately discarded;
only a few words like "smiling, looking at you" ever reach the AI.

AI text (replies, the agent, reading notes) comes from NVIDIA NIM when `NVIDIA_API_KEY` is set, with Groq as the
backup; speech recognition and the voice come from Groq, falling back to local models (faster-whisper, Piper) when
Groq can't be reached. Speech detection, retrieval and focus detection always run locally.

**AI providers** (`LLM_PROVIDER` in `config.py`): with `"nvidia"`, each reply goes to `NVIDIA_LLM_MODEL` first. If
NIM fails to start, or stays silent for `NVIDIA_STALL_TIMEOUT_S` before its first words, the reply moves to Groq and
NIM is skipped for `LLM_PROVIDER_RETRY_S`. Measured in Sept 2026 on the free endpoints: nemotron-3.5-lightning
usually starts in 0.4–1.1 s but stalls now and then (Groq then answers, ~3 s later than usual); many other NIM models
timed out. Note photos are read by `NVIDIA_VISION_MODEL` (llama-3.2-11b-vision, 5–16 s a page; Groq's image model is
the backup). `"groq"` uses Groq for everything.

**Setting this up for the first time?** Follow **[SETUP.md](SETUP.md)**: step-by-step, plain-language instructions for
Mac, Windows and Linux.

## Requirements

- macOS on Apple Silicon (this is what it was built and tested on)
- **Python 3.12** (mediapipe 1.0.x crashes on macOS, so 0.10.35 is pinned)
- Webcam, microphone (earphones with a mic recommended), a [Groq](https://console.groq.com) API key

## Setup

Quick version for macOS. For Windows, Linux or a slower walkthrough, see [SETUP.md](SETUP.md).

```bash
brew install python@3.12
```

```bash
"$(brew --prefix)/bin/python3.12" -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Put your key in `.env` (`GROQ_API_KEY=...`). Never commit `.env`.

Then download every model once, while online (~600 MB: Face Landmarker, MiniLM embeddings, spaCy English,
Piper voice, faster-whisper base, Silero VAD). After this the app loads models with Hugging Face forced offline,
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

Look at the screen for the first 2 seconds (focus calibration), wait for `READY`, fill in the welcome form on the page
(your name, and a subject or a way to talk), and start talking. Nothing you say is answered before that.

| Action | Global hotkey | Terminal (type + Enter) |
|---|---|---|
| Interrupt the tutor | Ctrl+Option+I | `i` |
| Force a focus intervention now | Ctrl+Option+F | `f` |
| Suppress / re-enable automatic interventions | Ctrl+Option+S | `s` |
| Recalibrate "looking at the screen" | Ctrl+Option+C | `c` |
| Microphone on / off | Ctrl+Option+M | `m` |
| Camera on / off | Ctrl+Option+V | `v` |
| Start a general tutor session without the page | | `g` |
| Quit | | `q` |

You can also interrupt just by talking over the tutor. Logs go to the console and `logs/tutor.log`, including
the focus score every 2 s, every state transition, and a per-turn `LATENCY` / `E2E` line.

### The page

`main.py` also opens a local page at **http://127.0.0.1:8765** (`UI_PORT` in `config.py`; set
`UI_OPEN_BROWSER = False` to stop it opening automatically). It's a dark, full-screen view that shows:

- a **welcome form** first: your name, then a subject (or type your own) or an off-topic mode, and whether to use the
  camera. The AI greets you by name and says what you're doing today. Switch anytime from the chip at the top right,
- a **3D avatar** (three.js, built from primitives, no model files) whose mouth moves with the loudness of the audio
  actually playing, which leans in while you talk, looks up while thinking and follows your mouse. Its **emotion**
  changes its eyebrows, eyes, mouth, head tilt, motion and colour, with a "?" when confused and a "!" when surprised.
  If WebGL isn't available, a glowing orb that pulses with the voice replaces it,
- a **mood chip** with the AI's current emotion, and each reply labelled with the emotion it was said in,
- a live **focus card** in tutor mode (score, threshold marker, focused / drifting / distracted and why, and how
  many distractions in the last 10 minutes); in the other modes it shows what the camera notices instead,
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

## Modes, emotions and the camera

| Mode | How it talks | Camera |
|---|---|---|
| **Tutor** (a subject) | Teaches step by step (about 100 words a reply) from your notes. Happy to joke or chat for a moment, then steers back. | **On:** tracks focus. Nudges escalate within 10 minutes: 1st light, 2nd annoyed, 3rd and later properly angry (stern but never insulting). Also checks in if you look puzzled right after an explanation, or keep yawning. **Off:** no focus tracking or nudges. |
| **Friend** | Casual and short, with real reactions: excited, sad, playful, annoyed if you're rude. | Optional. |
| **Therapist** | A calm, supportive listener. Says it's an AI and not a real therapist, never gets angry, and points to emergency services or a crisis line if you mention being in danger. | Optional. |
| **Normal chat** | Talks about anything. | Optional. |

**Emotions.** Every reply starts with a hidden tag such as `[happy]`, chosen by the AI to match the conversation
(neutral, happy, excited, proud, playful, caring, thoughtful, confused, surprised, sad, annoyed, angry). The tag is
never spoken or shown; a mid-reply change switches the face when the voice reaches that sentence. After it stops
talking, the face relaxes to neutral within `EMOTION_HOLD_S`.

**What the camera tells the AI.** While the camera is on, whatever you say is sent with a few words about what the
camera sees right now: smiling, frowning, yawning, looking away or down, eyes closed, out of view, someone else with
you. They come from MediaPipe face landmarks and blendshapes, compared with your own neutral face from calibration,
and the frame is dropped. With the camera on the AI is a little warmer and more attentive; with it off, calmer and
more neutral.

## Your notes (PDFs and photos)

Open **Notes** in the bottom bar, or drop files anywhere on the page. PDFs, photos of handwritten pages and
screenshots all work.

- **Reading, once:** pages with real PDF text are used as they are; photos and scanned pages are read by the Groq
  image model (`NOTES_VISION_MODEL`). The text is split into numbered sections (headings, else pages), indexed
  locally, and a small model (`NOTES_SUMMARY_MODEL`) writes a title, subject, topics, summary and suggested questions.
  Cards show a scanning animation while this happens.
- **The AI opens them itself.** It always knows your notes shelf (titles, subjects, when you added them) and has
  tools to open a note, read a section and search all notes. Say *"I just uploaded my AI engineering notes, check my
  notes section and explain them"*: it says "let me look at your notes", finds the newest match and teaches it
  section by section. **Explain this** on a card does the same without speaking.
- **While it teaches**, a holographic card beside the avatar shows the page, the section and progress, and the
  note's card glows. The section in use stays in the AI's prompt, so follow-up questions need no new lookup.
- **Cards expand** into the note's sections (each with *Explain from here*), its web links (click to open them)
  and suggested questions (click to ask).
- Files and everything read from them stay in `./notes` (not in git). Delete a note from its card.

## Agent mode: thinking, steps and links

- **It works in steps.** With notes or links involved, the AI reasons more deeply (`LLM_REASONING_EFFORT_AGENT`) and
  can chain up to `LLM_MAX_TOOL_ROUNDS` tool calls: open a note, list its links, open a page, open a repository
  found there, then answer. A short spoken line ("One sec, looking at github.com") fills the wait.
- **Links it may open:** links found in your notes (clickable PDF links and written addresses, e.g. the GitHub on a
  resume), links you type or say, and links on pages it already opened. Public http(s) sites only: this computer,
  the local network and unusual ports are refused (`tutor/web.py`). GitHub profiles and repositories are read
  through GitHub's public API and README. Sites that block automated reading (LinkedIn) are reported as such.
- **You see its thinking.** Above each reply, a *Thinking* block streams the model's reasoning (rough working, it
  can contain mistakes) and a timeline of every step with timings; it folds into "Thought for 2.4s · 3 steps" when
  the reply starts, and a click reopens it. The holographic card switches to the website being read.
- **Type instead of talking** in the box under the conversation, e.g. to paste a link.
- Terminal: `scripts/text_chat.py` has `/add <file>` and `/notes`.

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
| `STRIKE_WINDOW_S` | 600 | Distractions this close together make the tutor firmer each time |
| `SMILE_DELTA` / `FROWN_DELTA` | 0.30 / 0.30 | How far above your neutral face a smile / frown must be to count |
| `YAWN_JAW_OPEN` / `YAWN_MIN_S` | 0.50 / 1.0 | Jaw opening, held this long, that counts as a yawn |
| `PUZZLED_CHECKIN_S` | 6 | Frowning this long after an explanation makes the tutor offer to explain it again |

The focus log line also prints the smoothed `smile`, `brow` and `jaw` scores and the words sent to the AI, so you can
set the expression thresholds between your neutral face and a real smile, frown or yawn.

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

- **No network / Groq STT down:** the local faster-whisper backup starts if Groq hasn't answered within
  `STT_LOCAL_HEDGE_S` (0.6 s) or fails, its transcript is used, and Groq is skipped for 30 s.
- **Windows webcam:** Windows lets one program use the camera, so the page's self-view shows small frames the app
  sends over localhost (`/camera.mjpg`) instead of opening the camera itself. The browser does the same anywhere it
  finds the camera busy.
- **Groq rate limit (429):** that turn is retried on `LLM_FALLBACK_MODEL` (gpt-oss-20b), which has its own quota.
- **Groq voice unavailable** (terms not accepted, rate limit, no network): Piper speaks on this computer until Groq
  works again. A rate limit waits exactly as long as Groq asks for; anything else retries after
  `TTS_OFFLINE_RETRY_S`. Piper is loaded at startup (`TTS_LOCAL_PRELOAD`), so there's no silent gap.

## The voice

`TTS_ENGINE = "groq"` (default) uses Groq's Orpheus voice (`TTS_GROQ_VOICE`: autumn, diana, hannah, austin, daniel,
troy). It needs a one-time terms acceptance at
<https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english>. Sentences are requested
`TTS_PARALLEL` at a time and played in order; measured ~0.75 s for a short sentence, on any laptop. Groq's free tier
caps each voice model at ~3600 tokens a day (roughly 40-60 sentences), after which Piper takes over until it resets.

`"local"` uses Piper only (`TTS_VOICE`, an ONNX voice in `models/piper`): fully offline, no API key, ~70-90 ms a
sentence on laptop CPU, and it loads in 0.4 s using ~90 MB of RAM. Other voices come from
[piper-voices](https://huggingface.co/rhasspy/piper-voices); `scripts/download_models.py` fetches the configured one.
To choose per computer without editing `config.py`, add
`"tts_engine": "local"` (or `"groq"`) to `local_settings.json`.
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
  prompts.py            system prompt per mode, emotion rules and hidden app messages
  session.py            the session from the welcome form (mode, name, subject)
  emotions.py           emotion tags: the list, and the streaming parser that strips them
  notes.py              your uploaded notes: reading, sections, links, search, and the AI's tools
  web.py                safe web page reader for links (public sites only; GitHub via its API)
  perf.py               CPU thread limits
  speech_text.py        sentence chunking and LaTeX-to-speech
  hotkeys.py            global hotkeys (physical key matching, works with Option)
  state.py              state shared between the loops
  ui_server.py          FastAPI + WebSocket server for the page (background thread)
  audio/                mic + VAD, STT, TTS, speaker, device lookup and live switching
ui/
  index.html            the page
  static/               app.js, app.css, avatar.js (3D face), emotions.js, selfview.js, vendored KaTeX and three.js
scripts/
  download_models.py    cache all models for offline use
  focus_debug.py        live focus overlay (stage 1)
  focus_check.py        spoken guided focus check with PASS/FAIL report
  text_chat.py          terminal text chat in any mode, e.g. --mode friend --name Sam (stage 2)
  voice_chat.py         voice tutor without the camera (stage 3)
  hotkey_check.py       verifies global hotkeys from the current terminal
  *.command             Terminal.app launchers for the scripts above
materials/              your course notes
notes/                  notes uploaded in the page (created on first upload, not in git)
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

## Performance on slower laptops (and Windows)

- CPU thread pools are capped (`tutor/perf.py`), so the voice, speech recognition and face tracking don't fight
  over cores right when a reply is being spoken. Speech detection uses the single-threaded ONNX Silero model.
- The camera is read on its own thread keeping only the newest frame (no lag from driver buffering), at
  `CAMERA_WIDTH`×`CAMERA_HEIGHT`, and tracking drops to `FOCUS_FPS_BUSY` while the AI is thinking or talking.
- The 3D page renders calm moments at 30 fps and steps down its resolution, then its glow, if the GPU can't keep up.
- The focus log shows face-tracking time per frame (`infer=`), and every turn logs a `LATENCY` breakdown.

## Known limitations

- End-to-end latency (you stop speaking → tutor starts) measured a median of ~1.5 s on earphones. Most turns
  over target were Groq taking 1–2 s to start its reply.
- The tutor speaks English only (an English Piper voice); non-Latin text in replies is skipped.
- Conversation memory lasts for one session only.
