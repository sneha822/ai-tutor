"""
Stage 1 check: live focus score with a debug overlay.

    python scripts/focus_debug.py

Keys (in the video window):  q = quit   c = recalibrate (look at the screen first)
The window is local-only; frames are never saved.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

import config  # noqa: E402
from tutor.focus import FocusDetector  # noqa: E402

STATE_COLORS = {"FOCUSED": (80, 200, 80), "UNFOCUSED": (0, 190, 255), "DISTRACTED": (60, 60, 230)}


def draw(frame, fr):
    s = fr.snapshot
    h, w = frame.shape[:2]
    color = STATE_COLORS.get(s.state, (200, 200, 200))

    for p in fr.eye_points:
        cv2.circle(frame, p, 2, (255, 255, 0), -1)
    if fr.nose and fr.forward:
        tip = (int(fr.nose[0] + fr.forward[0] * 120), int(fr.nose[1] + fr.forward[1] * 120))
        cv2.arrowedLine(frame, fr.nose, tip, (255, 0, 255), 3, tipLength=0.25)

    # score bar
    bar_w = int((w - 40) * s.score / 100)
    thr_x = 20 + int((w - 40) * config.FOCUS_THRESHOLD / 100)
    cv2.rectangle(frame, (20, h - 40), (w - 20, h - 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 15), color, -1)
    cv2.line(frame, (thr_x, h - 46), (thr_x, h - 9), (255, 255, 255), 2)

    def fmt(v, spec):
        return "--" if v is None else format(v, spec)

    pitch_dir = ""
    if s.pitch is not None:
        pitch_dir = "DOWN" if s.pitch < -config.PITCH_DOWN_OK_DEG else ("UP" if s.pitch > config.PITCH_UP_OK_DEG else "")
    lines = [
        f"FOCUS {s.score:5.1f}  {s.state}",
        f"yaw {fmt(s.yaw, '+.1f')}   pitch {fmt(s.pitch, '+.1f')} {pitch_dir}",
        f"EAR {fmt(s.ear, '.3f')}  {'CLOSED' if s.ear is not None and s.ear < config.EAR_CLOSED else ''}",
        f"face {'yes' if s.face_present else 'NO'}   reason {s.reason or '-'}",
        f"unfocused {s.unfocused_seconds:.1f}s / {config.DISTRACTED_SECONDS:.0f}s   "
        f"{'calibrated' if s.calibrated else 'calibrating...'}   {s.fps:.0f} fps",
    ]
    if s.distracted:
        lines.append(f"DISTRACTED: {s.describe()}")
    for i, text in enumerate(lines):
        y = 30 + i * 28
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color if i == 0 else (255, 255, 255), 2, cv2.LINE_AA)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(name)-6s %(message)s",
                        datefmt="%H:%M:%S")
    det = FocusDetector()
    if not det.open_camera():
        sys.exit(1)
    period = 1.0 / config.FOCUS_FPS
    try:
        while True:
            t0 = time.monotonic()
            frame = det.read_frame()
            if frame is None:
                print("camera read failed")
                break
            frame = cv2.flip(frame, 1)  # mirror so movement feels natural
            if frame.shape[1] > config.FOCUS_FRAME_WIDTH:
                scale = config.FOCUS_FRAME_WIDTH / frame.shape[1]
                frame = cv2.resize(frame, (config.FOCUS_FRAME_WIDTH, int(frame.shape[0] * scale)))
            fr = det.process(frame, t0)
            draw(frame, fr)
            cv2.imshow("focus debug (q quit, c recalibrate)", frame)
            key = cv2.waitKey(max(1, int((period - (time.monotonic() - t0)) * 1000))) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                det.recalibrate()
    finally:
        det.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
