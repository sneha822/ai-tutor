"""
Speed and camera check for this computer. Close the tutor first (it uses the camera), then run:

    Windows:     .venv\\Scripts\\python scripts\\diagnose.py
    Mac / Linux: .venv/bin/python scripts/diagnose.py

It takes about a minute and writes logs/diagnose.txt. Share that file: it holds timings, versions and device
names only, nothing you said and no images.
"""
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GLOG_minloglevel", "2")        # quiet the face tracker's C++ chatter
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import config  # noqa: E402

REPORT = ROOT / "logs" / "diagnose.txt"
lines: list[str] = []
facts: dict = {}


def out(text: str = "") -> None:
    print(text, flush=True)
    lines.append(text)


def run(title: str, fn) -> None:
    out()
    out(f"== {title}")
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as e:
        out(f"  FAILED: {type(e).__name__}: {e}")
        out("  " + traceback.format_exc().strip().splitlines()[-2].strip())
    out(f"  ({time.perf_counter() - t0:.1f}s)")


def shell(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return ""


def ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# ------------------------------------------------------------------ system

def system() -> None:
    out(f"  OS: {platform.platform()} ({platform.machine()})")
    out(f"  Python: {sys.version.split()[0]} at {sys.executable}")
    if sys.platform == "win32":
        ps = ["powershell", "-NoProfile", "-Command"]
        out(f"  CPU: {shell(ps + ['(Get-CimInstance Win32_Processor).Name'])}")
        out(f"  GPU: {' | '.join(shell(ps + ['(Get-CimInstance Win32_VideoController).Name']).splitlines())}")
        out(f"  Power plan: {shell(['powercfg', '/getactivescheme'])}")
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("free", ctypes.c_ulonglong)] + [(f"x{i}", ctypes.c_ulonglong) for i in range(5)]

        class Power(ctypes.Structure):
            _fields_ = [("ac", ctypes.c_byte), ("flag", ctypes.c_byte), ("percent", ctypes.c_byte),
                        ("saver", ctypes.c_byte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]

        mem = Memory()
        mem.length = ctypes.sizeof(Memory)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        out(f"  RAM: {mem.total / 2**30:.1f} GB total, {mem.free / 2**30:.1f} GB free ({mem.load}% used)")
        power = Power()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
        facts["on_battery"] = power.ac == 0
        out(f"  Plugged in: {'yes' if power.ac == 1 else 'NO (on battery)' if power.ac == 0 else 'unknown'}, "
            f"battery saver: {'ON' if power.saver == 1 else 'off'}")
        facts["battery_saver"] = power.saver == 1
    elif sys.platform == "darwin":
        out(f"  CPU: {shell(['sysctl', '-n', 'machdep.cpu.brand_string'])}")
        out(f"  RAM: {int(shell(['sysctl', '-n', 'hw.memsize']) or 0) / 2**30:.1f} GB")
        out(f"  Power: {shell(['pmset', '-g', 'batt']).splitlines()[0] if shell(['pmset', '-g', 'batt']) else '?'}")
    else:
        cpu = next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                    if l.startswith('model name')), '?')
        out(f"  CPU: {cpu}")
    out(f"  Logical cores: {os.cpu_count()}")
    import cv2
    import mediapipe
    import onnxruntime
    import torch
    from tutor import perf
    out(f"  torch {torch.__version__} (default threads {torch.get_num_threads()}), onnxruntime {onnxruntime.__version__} "
        f"{onnxruntime.get_available_providers()}, opencv {cv2.__version__}, mediapipe {mediapipe.__version__}, "
        f"numpy {np.__version__}")
    perf.limit_threads()
    out(f"  App thread limits: voice {perf.TORCH_THREADS}, offline STT {perf.WHISPER_THREADS}")


# ------------------------------------------------------------------ camera

def camera() -> None:
    import cv2
    from tutor import focus
    names = {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MediaFoundation", cv2.CAP_AVFOUNDATION: "AVFoundation",
             cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "any"}
    best = None
    for index in (config.CAMERA_INDEX, config.CAMERA_INDEX + 1):
        for backend in focus.CAMERA_BACKENDS:
            for sized in (True, False):
                label = f"camera {index} {names.get(backend, backend)}{' 640x480' if sized else ' own size'}"
                t0 = time.perf_counter()
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    out(f"  {label}: does not open ({ms(t0):.0f}ms)")
                    cap.release()
                    continue
                opened = ms(t0)
                if sized:
                    if backend == cv2.CAP_DSHOW:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                first = None
                frame = None
                deadline = time.perf_counter() + 3
                while time.perf_counter() < deadline:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        first = ms(t0)
                        break
                if first is None:
                    out(f"  {label}: opens in {opened:.0f}ms but sends NO frames")
                    cap.release()
                    continue
                count, t1 = 0, time.perf_counter()
                while time.perf_counter() - t1 < 2:
                    ok, frame = cap.read()
                    count += bool(ok)
                fps = count / (time.perf_counter() - t1)
                brightness = float(frame.mean()) if frame is not None else 0
                out(f"  {label}: opens {opened:.0f}ms, first frame {first:.0f}ms, {fps:.1f} fps, "
                    f"{frame.shape[1]}x{frame.shape[0]}, brightness {brightness:.0f}/255"
                    + ("  <- looks black (privacy shutter or infrared camera?)" if brightness < 8 else ""))
                if brightness >= 8 and (best is None or fps > best[0]):
                    best = (fps, label)
                    facts["frame"] = frame.copy()
                cap.release()
    out(f"  Best: {best[1]} at {best[0]:.1f} fps" if best else "  NO CAMERA METHOD SENT USABLE FRAMES")
    facts["camera_fps"] = best[0] if best else 0


# ------------------------------------------------------------------ models

def face() -> None:
    from tutor.focus import FocusDetector
    t0 = time.perf_counter()
    det = FocusDetector()
    out(f"  Face tracker loads in {ms(t0):.0f}ms")
    frame = facts.get("frame")
    source = "a real camera frame" if frame is not None else "a blank frame (no camera frame available)"
    frame = frame if frame is not None else np.zeros((480, 640, 3), np.uint8)
    times = []
    base = time.monotonic()
    for i in range(40):
        t1 = time.perf_counter()
        snap = det.process(frame, base + i / 15).snapshot
        times.append(ms(t1))
    det._landmarker.close()
    avg = sum(times[5:]) / len(times[5:])
    facts["face_ms"] = avg
    out(f"  Per frame on {source}: {avg:.0f}ms average (face found: {snap.face_present}); "
        f"that allows about {1000 / avg:.0f} fps")


def voice() -> None:
    from tutor.audio.tts import TTS
    t0 = time.perf_counter()
    tts = TTS()
    out(f"  Voice (Kokoro) loads in {ms(t0):.0f}ms")
    for text in ("Sure, let me explain that.",
                 "The chain rule helps us differentiate a function that sits inside another function.",
                 "Think of it in layers: differentiate the outer layer while leaving the inside alone, then multiply "
                 "by the derivative of the inside, and you get the answer."):
        t1 = time.perf_counter()
        audio = tts.synthesize(text)
        took = ms(t1)
        seconds = len(audio) / 24000
        out(f"  {len(text):3d} chars -> {seconds:.1f}s of speech in {took:.0f}ms ({took / 1000 / seconds:.2f}x real time)")
        facts.setdefault("tts_first_ms", took)
        facts["speech"] = audio
    facts["tts_rtf"] = took / 1000 / seconds


def stt() -> None:
    audio24 = facts.get("speech")
    if audio24 is None:
        out("  skipped (no speech sample from the voice test)")
        return
    audio = np.interp(np.arange(0, len(audio24), 1.5), np.arange(len(audio24)), audio24).astype(np.float32)
    from tutor.audio.stt import Transcriber
    t0 = time.perf_counter()
    stt_ = Transcriber()
    out(f"  Speech recognition loads in {ms(t0):.0f}ms")
    t1 = time.perf_counter()
    text = stt_._local_stt(audio)
    out(f"  Offline (faster-whisper {config.STT_LOCAL_MODEL}): {ms(t1):.0f}ms for {len(audio) / 16000:.1f}s: {text[:60]!r}")
    t1 = time.perf_counter()
    try:
        text = stt_._groq_stt(audio)
        out(f"  Groq: {ms(t1):.0f}ms: {text[:60]!r}")
        facts["groq_stt_ms"] = ms(t1)
    except Exception as e:
        out(f"  Groq FAILED: {type(e).__name__}: {e}")


def vad() -> None:
    import torch
    from silero_vad import load_silero_vad
    model = load_silero_vad(onnx=True)
    frame = torch.from_numpy(np.random.default_rng(0).normal(0, 0.05, 512).astype(np.float32))
    for _ in range(20):
        model(frame, 16000)
    t0 = time.perf_counter()
    for _ in range(200):
        model(frame, 16000)
    out(f"  Speech detection: {ms(t0) / 200:.2f}ms per 32ms audio frame")


def groq() -> None:
    from tutor.llm import LLMClient
    client = LLMClient()
    for attempt in range(2):
        t0 = time.perf_counter()
        first = None
        for _ in client.stream([{"role": "user", "content": "Say hello in five words."}]):
            first = first or ms(t0)
        out(f"  Reply {attempt + 1}: first words after {first or 0:.0f}ms, done in {ms(t0):.0f}ms")
        facts["llm_first_ms"] = first or 0
        time.sleep(1)


def audio_devices() -> None:
    import sounddevice as sd
    apis = sd.query_hostapis()
    default_api = apis[sd.default.hostapi]["name"]
    out(f"  Audio system: {default_api}; default input: {sd.query_devices(kind='input')['name']}; "
        f"default output: {sd.query_devices(kind='output')['name']}")
    try:
        with sd.OutputStream(samplerate=24000, channels=1, dtype="float32", blocksize=480, latency="low") as s:
            out(f"  Output at 24 kHz: OK (latency {s.latency * 1000:.0f}ms)")
    except Exception as e:
        out(f"  Output at 24 kHz FAILED ({e}); the app falls back to the device's own rate")
    from tutor import local_settings
    out(f"  Saved choices: {local_settings.load()}")


def summary() -> None:
    hints = []
    if facts.get("on_battery") or facts.get("battery_saver"):
        hints.append("The laptop is on battery / battery saver: plug it in and use the Best performance power mode.")
    if facts.get("tts_first_ms", 0) > 700:
        hints.append(f"The voice is slow on this CPU (first sentence {facts['tts_first_ms']:.0f}ms): replies will start late.")
    if facts.get("face_ms", 0) > 40:
        hints.append(f"Face tracking is heavy here ({facts['face_ms']:.0f}ms a frame): lower FOCUS_FPS in config.py.")
    if facts.get("camera_fps", 99) < 10:
        hints.append("The camera delivers few frames: check the camera method lines above.")
    if facts.get("llm_first_ms", 0) > 1500:
        hints.append("Groq answers slowly from this network.")
    out()
    out("== Summary")
    for hint in hints or ["No obvious bottleneck found. Please also share logs/tutor.log from a session."]:
        out(f"  - {hint}")


if __name__ == "__main__":
    out(f"AI Tutor diagnostics, {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out("Close the tutor app before running this (it needs the camera).")
    for title, fn in (("System", system), ("Camera", camera), ("Face tracking", face), ("Voice", voice),
                      ("Speech recognition", stt), ("Speech detection", vad), ("Groq replies", groq),
                      ("Audio devices", audio_devices)):
        run(title, fn)
    summary()
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved to {REPORT}. Please share that file.")
