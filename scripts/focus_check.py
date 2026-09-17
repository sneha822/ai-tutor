"""
Guided focus-detector check. Speaks instructions aloud (macOS `say`), records per-phase numbers,
prints PASS/FAIL and writes the report to logs/focus_check.txt. Re-run at the venue after changing
lighting or config values.

    .venv/bin/python scripts/focus_check.py      (or double-click scripts/run_focus_check.command)

Only numbers are recorded; no frames are saved. Each frame is also run through a tracking-free
(IMAGE mode) landmarker so face-loss can be diagnosed: tracker bug vs. dark camera vs. no face in view.
"""
import logging
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_tasks  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

import config  # noqa: E402
from tutor.focus import FocusDetector, ensure_model  # noqa: E402

REPORT = ROOT / "logs" / "focus_check.txt"
SETTLE_S = 3.0  # ignore the start of each phase (speech + moving into position)
CAMERA_WAIT_S = 45  # time to click "Allow" on the macOS camera prompt

PHASES = [  # (key, spoken instruction, seconds)
    ("baseline", "Look at the laptop screen normally.", 7),
    ("down", "Look down at your lap, like checking your phone. Hold it.", 18),
    ("recover_down", "Look back up at the laptop screen.", 6),
    ("left", "Turn your head to the left. Hold it.", 8),
    ("recover_left", "Look back at the laptop screen.", 5),
    ("right", "Turn your head to the right. Hold it.", 8),
    ("recover_right", "Look back at the laptop screen.", 5),
    ("blink", "Look at the laptop screen and blink normally.", 7),
    ("closed", "Close your eyes. Keep them closed.", 8),
    ("recover_closed", "Open your eyes.", 5),
    ("absent", "Move out of the camera view, or cover the camera. Stay away.", 17),
    ("return", "Come back and look at the laptop screen.", 6),
]


def out(line=""):
    print(line, flush=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def say(text, wait=False):
    out(f">>> {text}")
    try:
        p = subprocess.Popen(["say", "-r", "190", text])
        if wait:
            p.wait()
    except Exception:
        pass


def med(xs):
    return round(statistics.median(xs), 3) if xs else None


def summarize(rows):
    settled = [r for r in rows if r["t"] >= SETTLE_S] or rows
    face = [r for r in settled if r["face"]]
    pitches = sorted(r["pitch"] for r in face)
    abs_yaws = sorted(abs(r["yaw"]) for r in face)
    n = max(1, len(settled))
    return {
        "pitch_p10": round(pitches[len(pitches) // 10], 1) if pitches else None,
        "abs_yaw_p90": round(abs_yaws[(len(abs_yaws) * 9) // 10], 1) if abs_yaws else None,
        "face_pct": round(100 * len(face) / n),
        "img_face_pct": round(100 * sum(r["img_face"] for r in settled) / n),
        "brightness": round(med([r["bright"] for r in settled]) or 0),
        "yaw": med([r["yaw"] for r in face]),
        "pitch": med([r["pitch"] for r in face]),
        "ear": med([r["ear"] for r in face]),
        "ear_min": round(min(r["ear"] for r in face), 3) if face else None,
        "min_score": round(min(r["score"] for r in settled)) if settled else None,
        "end_score": round(rows[-1]["score"]) if rows else None,
        "states": Counter(r["state"] for r in settled),
        "reasons": Counter(r["reason"] for r in settled if r["reason"]),
        "end_state": rows[-1]["state"] if rows else None,
        "distracted_reason": next((r["dreason"] for r in reversed(rows) if r["state"] == "DISTRACTED"), None),
    }


def top_reason(s):
    return s["reasons"].most_common(1)[0][0] if s["reasons"] else None


def diagnose(s):
    if s["face_pct"] >= 50:
        return ""
    if s["img_face_pct"] >= 50:
        return " | DIAG: tracker lost a face that fresh detection still sees (VIDEO-mode tracking bug)"
    if s["brightness"] < 25:
        return " | DIAG: camera frame dark or covered"
    return " | DIAG: no detectable face in view (off-camera, or head too far down/turned)"


def evaluate(key, s):
    """Return (ok, note)."""
    if s["end_state"] is None:
        return False, "no frames captured"
    if key == "baseline":
        ok = s["face_pct"] >= 90 and s["yaw"] is not None and abs(s["yaw"]) < 10 and abs(s["pitch"]) < 10
        return ok, f"face {s['face_pct']}%, open-eye EAR {s['ear']}"
    if key == "down":
        if s["pitch"] is not None and s["pitch"] > config.PITCH_UP_OK_DEG:
            return False, "pitch went UP when looking down -> set PITCH_SIGN = -1"
        ok = "DISTRACTED" in s["states"] and s["distracted_reason"] == "looking_down"
        return ok, f"distracted reason={s['distracted_reason']}"
    if key in ("left", "right"):
        ok = s["yaw"] is not None and abs(s["yaw"]) > config.YAW_OK_DEG and top_reason(s) == "looking_away"
        return ok, f"top reason={top_reason(s)}"
    if key == "blink":
        ok = not s["reasons"].get("eyes_closed") and s["end_state"] == "FOCUSED"
        return ok, f"blinks (EAR min {s['ear_min']}) must never count as eyes closed"
    if key == "closed":
        ok = (s["ear"] is not None and s["ear"] < config.EAR_CLOSED and top_reason(s) == "eyes_closed"
              and s["min_score"] < config.FOCUS_THRESHOLD)
        return ok, f"top reason={top_reason(s)}"
    if key == "absent":
        ok = s["face_pct"] <= 10 and "DISTRACTED" in s["states"] and s["distracted_reason"] == "absent"
        return ok, f"distracted reason={s['distracted_reason']}"
    ok = s["end_state"] == "FOCUSED"
    return ok, f"ended {s['end_state']}"


def open_camera_with_retry(det):
    deadline = time.monotonic() + CAMERA_WAIT_S
    prompted = False
    while time.monotonic() < deadline:
        if det.open_camera():
            for _ in range(10):
                frame = det.read_frame()
                if frame is not None and frame.mean() > 5:  # permission-pending cameras return black frames
                    return True
                time.sleep(0.2)
            det.close_camera()
        if not prompted:
            say("Please allow camera access in the pop up.")
            prompted = True
        time.sleep(2)
    return False


def main():
    logging.basicConfig(level=logging.WARNING)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(f"focus check {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    det = FocusDetector()
    fresh = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3))
    if not open_camera_with_retry(det):
        out("RESULT: camera unavailable (permission denied or no camera)")
        say("I could not open the camera.")
        return

    say("Starting the focus check in five seconds. Sit in front of the laptop camera.", wait=True)
    time.sleep(3)
    results = []
    for key, text, secs in PHASES:
        if key == "baseline":
            det.recalibrate()
        say(text)
        rows, t0 = [], time.monotonic()
        while (now := time.monotonic()) - t0 < secs:
            frame = det.read_frame()
            if frame is None:
                continue
            small = cv2.resize(frame, (config.FOCUS_FRAME_WIDTH,
                                       int(frame.shape[0] * config.FOCUS_FRAME_WIDTH / frame.shape[1])))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            img_face = bool(fresh.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks)
            bright = float(small.mean())
            s = det.process(frame, now).snapshot
            del frame, small, rgb
            rows.append({"t": now - t0, "score": s.score, "state": s.state, "face": s.face_present,
                         "img_face": img_face, "bright": bright,
                         "yaw": s.yaw, "pitch": s.pitch, "ear": s.ear, "reason": s.reason,
                         "dreason": s.distraction_reason})
        s = summarize(rows)
        ok, note = evaluate(key, s)
        results.append(ok)
        out(f"[{'PASS' if ok else 'FAIL'}] {key:<15} yaw={s['yaw']} (|p90| {s['abs_yaw_p90']}) "
            f"pitch={s['pitch']} (p10 {s['pitch_p10']}) ear={s['ear']} (min {s['ear_min']}) "
            f"face={s['face_pct']}% fresh-detect={s['img_face_pct']}% bright={s['brightness']} "
            f"score min/end={s['min_score']}/{s['end_score']} "
            f"states={dict(s['states'])} reasons={dict(s['reasons'])} | {note}{diagnose(s)}")
    det.stop()
    fresh.close()
    out(f"RESULT: {sum(results)}/{len(results)} phases passed")
    say("Focus check complete.", wait=True)


if __name__ == "__main__":
    main()
