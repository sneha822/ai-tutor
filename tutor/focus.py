"""
Webcam focus detector: MediaPipe Face Landmarker -> rolling focus score (0-100).

Privacy: each frame is processed in memory and dropped. Nothing is saved or transmitted.

Signals:
  * head pose (yaw / pitch) from MediaPipe's facial transformation matrix
  * eye aspect ratio (EAR) from eye landmarks -> sustained closure
  * face presence. The detector can't see a face tilted steeply down (phone in lap) or turned past
    ~50 deg, so a face lost right after such a pose is attributed to that pose, not to walking away.

The camera can be turned off at runtime (set_enabled): the webcam is released and tracking pauses.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import config

log = logging.getLogger("focus")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# EAR landmarks, ordered (p1 outer corner, p2 top, p3 top, p4 inner corner, p5 bottom, p6 bottom)
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP = 1
LOST_POSE_WINDOW_S = 1.5  # head-pose history checked when the face disappears

REASON_TEXT = {
    "looking_down": "looking down (possibly at their phone)",
    "looking_away": "looking away from the screen",
    "eyes_closed": "eyes closed / drowsy",
    "absent": "not in front of the camera",
}


@dataclass
class FocusSnapshot:
    score: float = 100.0
    state: str = "FOCUSED"            # FOCUSED | UNFOCUSED | DISTRACTED | NO_CAMERA | CAMERA_OFF
    camera_ok: bool = False
    face_present: bool = False
    yaw: float | None = None          # degrees, calibrated
    pitch: float | None = None        # degrees, calibrated, + = up
    ear: float | None = None
    reason: str | None = None         # what's wrong on this frame
    distracted: bool = False
    distraction_reason: str | None = None  # dominant reason over the whole unfocused window
    unfocused_seconds: float = 0.0
    calibrated: bool = False
    fps: float = 0.0

    def describe(self) -> str:
        """Human-readable summary for the LLM prompt."""
        what = REASON_TEXT.get(self.distraction_reason or "", "not paying attention")
        return f"{what} for {self.unfocused_seconds:.0f}s"


@dataclass
class FrameResult:
    snapshot: FocusSnapshot
    eye_points: list[tuple[int, int]] = field(default_factory=list)
    nose: tuple[int, int] | None = None
    forward: tuple[float, float] | None = None  # (x, y) of face-forward vector, image space


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Face Landmarker model -> %s", MODEL_PATH)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _ear(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


class FocusDetector:
    def __init__(self, camera_index: int = config.CAMERA_INDEX):
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._snap = FocusSnapshot()
        self._stop = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        options = vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

        self._t0 = time.monotonic()
        self._last_ts_ms = -1
        self._fps = 0.0
        self._last_log_t = 0.0
        self._recent_pose: deque[tuple[float, float, float]] = deque(maxlen=90)  # (t, yaw, pitch), calibrated
        self._reason_time: dict[str, float] = {}
        self._reset_tracking()

        self._calib_offset = (0.0, 0.0)   # (yaw, pitch)
        self._calib_samples: list[tuple[float, float]] = []
        self._calib_start: float | None = None
        self._calibrated = False

    # ------------------------------------------------------------------ public

    def snapshot(self) -> FocusSnapshot:
        with self._lock:
            return FocusSnapshot(**self._snap.__dict__)

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def set_enabled(self, on: bool) -> None:
        """Camera off releases the webcam and pauses tracking; back on reopens it and recalibrates."""
        if on == self._enabled.is_set():
            return
        if on:
            self.recalibrate()
            self._enabled.set()
            log.info("Camera turned ON")
        else:
            self._enabled.clear()
            self._set_camera_off()
            log.info("Camera turned OFF: webcam released, focus tracking paused")

    def recalibrate(self) -> None:
        """Treat the next CALIBRATION_SECONDS of head pose as 'looking at the screen'."""
        with self._lock:
            self._calib_samples = []
            self._calib_start = None
            self._calibrated = False
        log.info("Recalibration requested — look at the screen")

    def start(self) -> None:
        """Run the capture loop on a background thread (used by the full app)."""
        self._thread = threading.Thread(target=self._run, name="focus", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.close_camera()
        self._landmarker.close()

    def open_camera(self) -> bool:
        self.close_camera()
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            log.error("Could not open camera %d (check macOS camera permission for your terminal)",
                      self.camera_index)
            self._set_no_camera()
            return False
        self._cap = cap
        self._reset_tracking()   # a fresh start: no stale "absent" time carried over from before
        log.info("Camera %d opened", self.camera_index)
        return True

    def close_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_frame(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def process(self, frame_bgr: np.ndarray, now: float | None = None) -> FrameResult:
        """Run inference on one frame and update the rolling score. The frame is not retained."""
        now = time.monotonic() if now is None else now
        h, w = frame_bgr.shape[:2]
        if w > config.FOCUS_FRAME_WIDTH:
            scale = config.FOCUS_FRAME_WIDTH / w
            frame_bgr = cv2.resize(frame_bgr, (config.FOCUS_FRAME_WIDTH, int(h * scale)))
            h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        ts_ms = max(int((now - self._t0) * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        del rgb

        dt = 0.0 if self._last_t is None else min(now - self._last_t, 1.0)
        self._last_t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)

        fr = FrameResult(snapshot=FocusSnapshot())
        face_present = bool(result.face_landmarks)
        yaw = pitch = ear = None
        penalties: dict[str, float] = {}

        if face_present:
            self._last_face_t = now
            lms = result.face_landmarks[0]
            pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)
            ear = (_ear(pts, LEFT_EYE) + _ear(pts, RIGHT_EYE)) / 2.0

            raw_yaw, raw_pitch, fwd = self._head_pose(result)
            self._update_calibration(now, raw_yaw, raw_pitch)
            yaw = raw_yaw - self._calib_offset[0]
            pitch = raw_pitch - self._calib_offset[1]
            self._recent_pose.append((now, yaw, pitch))

            penalties["looking_down"] = _ramp(-pitch, config.PITCH_DOWN_OK_DEG, config.PITCH_DOWN_MAX_DEG)
            penalties["looking_away"] = max(
                _ramp(abs(yaw), config.YAW_OK_DEG, config.YAW_MAX_DEG),
                _ramp(pitch, config.PITCH_UP_OK_DEG, config.PITCH_UP_MAX_DEG),
            )
            if ear < config.EAR_CLOSED:
                self._eyes_closed_since = self._eyes_closed_since or now
                closed_for = now - self._eyes_closed_since
                penalties["eyes_closed"] = _ramp(closed_for, config.EYES_CLOSED_GRACE_S, config.EYES_CLOSED_MAX_S)
            else:
                self._eyes_closed_since = None

            fr.eye_points = [tuple(map(int, pts[i])) for i in LEFT_EYE + RIGHT_EYE]
            fr.nose = tuple(map(int, pts[NOSE_TIP]))
            fr.forward = fwd
        else:
            self._eyes_closed_since = None
            if now - self._last_face_t > config.FACE_LOST_GRACE_S:
                penalties[self._lost_face_reason()] = 1.0

        # Dominant penalty on this frame. Dict order breaks ties: looking_down beats eyes_closed,
        # since looking down also lowers EAR.
        reason, penalty = None, 0.0
        for k, v in penalties.items():
            if v > penalty:
                reason, penalty = k, v
        target = 100.0 * (1.0 - penalty)
        alpha = 1.0 - math.exp(-dt / config.SCORE_SMOOTHING_S) if dt > 0 else 0.0
        self._score += alpha * (target - self._score)

        # Sustained-window state machine
        threshold = config.FOCUS_THRESHOLD
        if self._state == "DISTRACTED":
            threshold += config.RECOVERY_HYSTERESIS
        if self._score < threshold:
            self._unfocused_since = self._unfocused_since or now
            if reason:
                self._reason_time[reason] = self._reason_time.get(reason, 0.0) + dt
        else:
            self._unfocused_since = None
            self._reason_time.clear()

        unfocused_for = now - self._unfocused_since if self._unfocused_since else 0.0
        absent_for = now - self._last_face_t
        long_absent = not face_present and absent_for >= config.ABSENT_SECONDS
        distracted = unfocused_for >= config.DISTRACTED_SECONDS or long_absent
        dominant = max(self._reason_time, key=self._reason_time.get) if self._reason_time else reason
        if long_absent:
            dominant = reason or "absent"
            unfocused_for = max(unfocused_for, absent_for)

        new_state = "DISTRACTED" if distracted else ("UNFOCUSED" if self._unfocused_since else "FOCUSED")
        if new_state != self._state:
            log.info("Focus state %s -> %s  (score=%.0f, reason=%s, %.1fs)",
                     self._state, new_state, self._score, dominant, unfocused_for)
            self._state = new_state

        snap = FocusSnapshot(
            score=round(self._score, 1),
            state=new_state,
            camera_ok=True,
            face_present=face_present,
            yaw=None if yaw is None else round(yaw, 1),
            pitch=None if pitch is None else round(pitch, 1),
            ear=None if ear is None else round(ear, 3),
            reason=reason,
            distracted=distracted,
            distraction_reason=dominant if new_state != "FOCUSED" else None,
            unfocused_seconds=round(unfocused_for, 1),
            calibrated=self._calibrated,
            fps=round(self._fps, 1),
        )
        with self._lock:
            self._snap = snap
        fr.snapshot = snap

        if now - self._last_log_t >= config.FOCUS_LOG_INTERVAL_S:
            self._last_log_t = now
            log.info("focus=%3.0f %-10s face=%s yaw=%s pitch=%s ear=%s reason=%s fps=%.0f",
                     snap.score, snap.state, "Y" if face_present else "N",
                     snap.yaw, snap.pitch, snap.ear, snap.reason, snap.fps)
        return fr

    # ----------------------------------------------------------------- internal

    def _reset_tracking(self) -> None:
        self._last_t: float | None = None
        self._score = 100.0
        self._last_face_t = time.monotonic()
        self._recent_pose.clear()
        self._eyes_closed_since: float | None = None
        self._unfocused_since: float | None = None
        self._reason_time.clear()
        self._state = "FOCUSED"

    def _lost_face_reason(self) -> str:
        """Why the face vanished, judged from the head pose just before it did."""
        recent = [(y, p) for t, y, p in self._recent_pose if self._last_face_t - t <= LOST_POSE_WINDOW_S]
        if recent and min(p for _, p in recent) < -config.PITCH_DOWN_OK_DEG:
            return "looking_down"
        if recent and max(abs(y) for y, _ in recent) > config.YAW_OK_DEG:
            return "looking_away"
        return "absent"

    def _head_pose(self, result) -> tuple[float, float, tuple[float, float]]:
        """Yaw/pitch (degrees) from the face-forward axis of the transformation matrix.

        MediaPipe's metric space is right-handed, +Y up, +Z toward the camera. The canonical face
        model looks down +Z, so column 2 of the rotation is where the face is pointing.
        """
        m = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
        f = m[:3, 2]
        f = f / (np.linalg.norm(f) + 1e-9)
        yaw = math.degrees(math.atan2(f[0], f[2]))
        pitch = config.PITCH_SIGN * math.degrees(math.asin(float(np.clip(f[1], -1.0, 1.0))))
        return yaw, pitch, (float(f[0]), float(-f[1] * config.PITCH_SIGN))

    def _update_calibration(self, now: float, yaw: float, pitch: float) -> None:
        if self._calibrated:
            return
        if self._calib_start is None:
            self._calib_start = now
        self._calib_samples.append((yaw, pitch))
        if now - self._calib_start < config.CALIBRATION_SECONDS:
            return
        y0, p0 = np.median(np.array(self._calib_samples), axis=0)
        if abs(y0) > config.CALIBRATION_MAX_OFFSET_DEG or abs(p0) > config.CALIBRATION_MAX_OFFSET_DEG:
            log.warning("Calibration rejected (yaw=%.1f pitch=%.1f looks off-screen); using 0,0. "
                        "Look at the screen and recalibrate.", y0, p0)
            y0, p0 = 0.0, 0.0
        else:
            log.info("Calibrated neutral pose: yaw=%.1f pitch=%.1f", y0, p0)
        self._calib_offset = (float(y0), float(p0))
        self._calibrated = True

    def _set_no_camera(self) -> None:
        with self._lock:
            # Hold score at 100 so a camera failure never fires a false intervention.
            self._snap = FocusSnapshot(state="NO_CAMERA", camera_ok=False)

    def _set_camera_off(self) -> None:
        with self._lock:
            self._snap = FocusSnapshot(state="CAMERA_OFF", camera_ok=False)

    def _run(self) -> None:
        period = 1.0 / config.FOCUS_FPS
        while not self._stop.is_set():
            if not self._enabled.is_set():
                if self._cap is not None:
                    self.close_camera()
                    log.info("Webcam released")
                self._set_camera_off()
                self._stop.wait(0.2)
                continue
            if self._cap is None and not self.open_camera():
                self._stop.wait(2.0)
                continue
            t0 = time.monotonic()
            frame = self.read_frame()
            if frame is None:
                log.error("Camera read failed; reopening")
                self.close_camera()
                self._set_no_camera()
                self._stop.wait(1.0)
                continue
            try:
                self.process(frame, t0)
            except Exception:
                log.exception("Focus processing error (continuing)")
            del frame
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))
