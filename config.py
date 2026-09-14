"""
All tunables live here. Edit, save, restart. No other file needs touching to retune the demo.
"""

# =====================================================================
# FOCUS DETECTION  — retune these at the venue
# =====================================================================
FOCUS_THRESHOLD = 50          # score (0-100) below this counts as "unfocused"
DISTRACTED_SECONDS = 10.0     # score must stay below threshold this long before we flag distraction
ABSENT_SECONDS = 10.0         # no face for this long = walked away (flags immediately)
RECOVERY_HYSTERESIS = 10      # once distracted, score must climb above THRESHOLD + this to clear

# Head pose (degrees, relative to the calibrated "looking at screen" pose)
YAW_OK_DEG = 20               # left/right turn tolerated before penalty starts
YAW_MAX_DEG = 40              # full penalty at this turn
PITCH_DOWN_OK_DEG = 10        # looking down tolerated (measured: normal laptop posture drifts to -7..-12 deg)
PITCH_DOWN_MAX_DEG = 20       # full penalty (measured phone glances: -14 to -26 deg; score crosses 50 at -15)
PITCH_UP_OK_DEG = 20
PITCH_UP_MAX_DEG = 40
PITCH_SIGN = 1                # verified: looking down gives negative pitch. Flip to -1 only if that changes.

# Eyes
EAR_CLOSED = 0.12             # eye aspect ratio below this = eyes closed (measured: open 0.21-0.30, closed ~0.03)
EYES_CLOSED_GRACE_S = 0.8     # ignore closures shorter than this (blinks)
EYES_CLOSED_MAX_S = 2.5       # full penalty after eyes closed this long

# Smoothing / calibration
SCORE_SMOOTHING_S = 1.5       # EMA time constant for the score; higher = steadier, slower
FACE_LOST_GRACE_S = 1.0       # brief detection dropouts don't count as absence
CALIBRATION_SECONDS = 2.0     # at startup, average this much head pose as "looking at screen"
CALIBRATION_MAX_OFFSET_DEG = 30  # reject calibration if neutral pose is wilder than this

# Camera
CAMERA_INDEX = 0
FOCUS_FPS = 15
FOCUS_FRAME_WIDTH = 640       # frames are downscaled to this width before inference
FOCUS_LOG_INTERVAL_S = 2.0    # how often the score is printed to the console

# =====================================================================
# LLM (Groq) — only layer that needs network
# =====================================================================
LLM_MODEL = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile was retired from Groq; measured ~0.55s to first sentence
LLM_FALLBACK_MODEL = "openai/gpt-oss-20b"  # used for a turn when LLM_MODEL is rate limited (free tier: 8k tokens/min per model). "" to disable
LLM_REASONING_EFFORT = "low"        # gpt-oss only; higher = slower first token
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 600                # includes hidden reasoning tokens
LLM_TIMEOUT_S = 10.0
LLM_KEEPALIVE_S = 300.0             # pooled connection lifetime between turns (measured: no clear latency effect; harmless)
HISTORY_TURNS = 10                  # most recent user/tutor exchanges sent verbatim (no summarization)

# =====================================================================
# RAG (local)
# =====================================================================
MATERIALS_DIR = "materials"         # .md / .txt / .pdf, re-indexed automatically when files change
CHROMA_DIR = ".chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_CHARS = 800
CHUNK_OVERLAP = 150
RAG_TOP_K = 4
RAG_MAX_DISTANCE = 0.75             # cosine distance; chunks less similar than this are not injected

# =====================================================================
# VOICE (stage 3)
# =====================================================================
# Microphone and speaker. Easiest: pick them in the page (the arrow next to the mic button). That choice is saved in
# LOCAL_SETTINGS_FILE and wins over these two on the next start. Matched by (partial) name so plugging something in
# can't silently switch devices. None = system default. List devices with:  .venv/bin/python -m sounddevice
INPUT_DEVICE = None
OUTPUT_DEVICE = None
LOCAL_SETTINGS_FILE = "local_settings.json"   # per-computer choices saved by the page; never committed to git

# Speech detection (Silero VAD, 16 kHz, 32 ms frames)
VAD_THRESHOLD = 0.5            # speech probability for a frame to count as speech
MIN_SPEECH_RMS = 0.02          # loudness gate: quieter speech (background voices) can't start a turn. Each utterance's
                               # loudness is logged; set this between the room's voices and yours (live barge-ins: 0.03-0.30)
END_SILENCE_MS = 300           # silence that ends an utterance; every ms counts toward latency (450 measured ~1.77s median E2E). Raise if it cuts you off mid-sentence
MIN_UTTERANCE_MS = 250         # ignore blips shorter than this
MAX_UTTERANCE_S = 20
PRE_ROLL_MS = 300              # audio kept from before speech onset so the first syllable isn't clipped

# Mic gating while the tutor speaks (soft gate). Earphones = no echo, so this is permissive.
# On laptop/external speakers: BARGE_IN_VAD_THRESHOLD ~0.85, BARGE_IN_MIN_MS ~400, BARGE_IN_MIN_RMS ~0.03, ECHO_TAIL_MS ~200.
BARGE_IN_ENABLED = True        # False = hard gate: mic ignored while the tutor speaks; interrupt by hotkey only
BARGE_IN_VAD_THRESHOLD = 0.7   # speech probability needed to count toward an interruption
BARGE_IN_MIN_MS = 250          # sustained speech needed to interrupt
BARGE_IN_MIN_RMS = 0.02        # minimum loudness (float RMS) to count toward an interruption (live barge-ins: 0.03-0.30)
ECHO_TAIL_MS = 0               # keep the gate closed this long after playback stops
BARGE_IN_LOOKBACK_MS = 800     # audio before the trigger kept as the start of the new utterance (speakers: ~250, or echo gets transcribed)

# Speech-to-text: Groq primary, local faster-whisper always computed in parallel as the fallback
STT_GROQ_MODEL = "whisper-large-v3-turbo"   # measured ~280-400ms for a 14s clip
STT_GROQ_TIMEOUT_S = 1.5       # slower than this -> use the local transcript
STT_LOCAL_MODEL = "base"       # CPU int8; measured ~310ms for a 14s clip
STT_OFFLINE_RETRY_S = 30       # after a Groq STT failure, stay local-only this long

# Text-to-speech (Kokoro-82M, CPU; measured ~0.1x real time)
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0

# Global hotkeys (pynput syntax). macOS needs Accessibility + Input Monitoring permission for your terminal.
HOTKEY_INTERRUPT = "<ctrl>+<alt>+i"

# =====================================================================
# FOCUS INTERVENTIONS (stage 4)
# =====================================================================
INTERVENTIONS_ENABLED = True       # automatic interventions on at startup (HOTKEY_TOGGLE_SUPPRESS flips this live)
INTERVENTION_COOLDOWN_S = 45       # minimum gap between spoken interventions
FORCE_INTERVENTION_REASON = "looking down at their phone for about 30s"   # used when forcing while detection sees nothing
HOTKEY_FORCE_INTERVENTION = "<ctrl>+<alt>+f"   # guarantee the demo moment: intervene right now
HOTKEY_TOGGLE_SUPPRESS = "<ctrl>+<alt>+s"      # block/unblock automatic interventions (e.g. bad venue lighting)
HOTKEY_RECALIBRATE = "<ctrl>+<alt>+c"          # look at the screen, press, hold still 2s
HOTKEY_TOGGLE_MIC = "<ctrl>+<alt>+m"           # mic off = device released, nothing is heard or transcribed
HOTKEY_TOGGLE_CAMERA = "<ctrl>+<alt>+v"        # camera off = webcam released, focus tracking and auto nudges pause

# =====================================================================
# UI (stage 5)
# =====================================================================
UI_HOST = "127.0.0.1"          # localhost only; the page is never exposed to the network
UI_PORT = 8765
UI_OPEN_BROWSER = True         # open the page in the default browser at startup
UI_FOCUS_HZ = 5                # focus/status updates pushed to the page per second
UI_AUDIO_HZ = 30               # voice loudness updates per second (drives the avatar's mouth)
