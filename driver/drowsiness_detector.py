"""
=============================================================================
Driver Drowsiness Detection System  (MediaPipe Tasks API – v0.10.x+)
=============================================================================
Monitors driver eye closure in real time via webcam. If eyes remain closed
for more than N seconds, an alert sound is played and a phone call is
automatically placed to the owner via the Twilio API.

Setup Instructions:
-------------------
1. Install dependencies:
       pip install opencv-python "mediapipe>=0.10.0" twilio python-dotenv scipy numpy

2. Create a .env file in the same directory with your Twilio credentials:
       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=your_auth_token
       TWILIO_FROM_NUMBER=+1XXXXXXXXXX      # Your Twilio number
       TWILIO_TO_NUMBER=+1XXXXXXXXXX        # Owner's phone number

3. The face landmark model (~6 MB) is downloaded automatically on first run
   and cached as  face_landmarker.task  in the project folder.

4. Run the script:
       python drowsiness_detector.py            # live mode (real Twilio call)
       python drowsiness_detector.py --test     # simulation – no real call

Controls:
---------
  Q  – Quit the application
  R  – Manually reset the drowsiness timer
=============================================================================
"""

import os
import sys
import time
import argparse
import logging
import threading
import urllib.request
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial import distance as dist
from dotenv import load_dotenv

# ── Mediapipe  (Tasks API – 0.10.x) ─────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
except ImportError as _e:
    print(f"[ERROR] mediapipe import failed: {_e}")
    print("        Run:  pip install 'mediapipe>=0.10.0'")
    sys.exit(1)

# ── Twilio ───────────────────────────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    from twilio.base.exceptions import TwilioRestException
except ImportError:
    print("[ERROR] twilio is not installed. Run: pip install twilio")
    sys.exit(1)

# ── .env ─────────────────────────────────────────────────────────────────────
load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION  – tweak these values to tune detection sensitivity
# =============================================================================

class Config:
    # ── Eye Aspect Ratio ──────────────────────────────────────────────────────
    # EAR below this value → eye treated as closed  (typical open ≈ 0.25–0.35)
    EAR_THRESHOLD: float = 0.20

    # Consecutive frames EAR must be low before the drowsiness timer starts.
    # Filters out normal blinks (~150 ms ≈ 4–5 frames at 30 fps).
    EAR_CONSEC_FRAMES: int = 4

    # ── Drowsiness timing ─────────────────────────────────────────────────────
    # Seconds eyes must remain closed to trigger the alert
    DROWSY_SECONDS: float = 2.5

    # ── Twilio call ───────────────────────────────────────────────────────────
    # Minimum gap (seconds) between successive calls for the SAME event
    CALL_COOLDOWN_SECONDS: float = 30.0

    # Text-to-speech message played to the owner
    TWIML_MESSAGE: str = (
        "Warning! The driver appears to be drowsy. "
        "Please check on them immediately."
    )

    # ── Webcam ────────────────────────────────────────────────────────────────
    CAMERA_INDEX: int = 0       # 0 = default webcam; change for external cam
    FRAME_WIDTH:  int = 640
    FRAME_HEIGHT: int = 480

    # ── Display ───────────────────────────────────────────────────────────────
    SHOW_LANDMARKS: bool = True    # Draw the 12 eye landmark dots
    SHOW_EAR_VALUE: bool = True    # Overlay live EAR readout

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = "face_landmarker.task"
    MODEL_URL:  str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    # ── Twilio credentials (loaded from .env) ─────────────────────────────────
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN:  str = os.getenv("TWILIO_AUTH_TOKEN",  "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    TWILIO_TO_NUMBER:   str = os.getenv("TWILIO_TO_NUMBER",   "")


# =============================================================================
# MediaPipe Face Landmarker – eye landmark indices
# =============================================================================
# 6 points used for the EAR formula:
#   EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
#
#   P2  P3
#  P1    P4   ← horizontal eye corners
#   P6  P5
#
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # left eye  (from wearer's POV)
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]  # right eye


# =============================================================================
# Model Download Helper
# =============================================================================

def ensure_model(cfg: Config) -> str:
    """
    Return the absolute path to the face landmarker model file.
    Downloads it from Google's CDN if not already present (~6 MB, one-time).
    """
    model_path = Path(cfg.MODEL_PATH).resolve()
    if model_path.exists():
        log.info("Model found: %s", model_path)
        return str(model_path)

    log.info("Downloading face landmarker model (~6 MB) …")
    log.info("URL: %s", cfg.MODEL_URL)

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            pct = min(block_count * block_size / total_size * 100, 100)
            sys.stdout.write(f"\r  Progress: {pct:5.1f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(cfg.MODEL_URL, str(model_path), reporthook=_progress)
        sys.stdout.write("\n")
        log.info("Model saved to: %s", model_path)
    except Exception as exc:
        log.error("Failed to download model: %s", exc)
        log.error("Download it manually from:\n  %s\nand place it as '%s'",
                  cfg.MODEL_URL, cfg.MODEL_PATH)
        sys.exit(1)

    return str(model_path)


# =============================================================================
# Helper Functions
# =============================================================================

def compute_ear(
    landmarks: list,          # list[NormalizedLandmark] for one face
    eye_indices: list,        # 6 indices [P1..P6]
    image_w: int,
    image_h: int,
) -> float:
    """
    Compute Eye Aspect Ratio (EAR).

    Returns a float in roughly [0.0, 0.45]:
        ~0.30 = eyes wide open
        ~0.20 = threshold (eyes half-closed)
        ~0.00 = fully closed
    """
    pts = np.array(
        [(landmarks[i].x * image_w, landmarks[i].y * image_h) for i in eye_indices],
        dtype=np.float64,
    )

    vertical_1 = dist.euclidean(pts[1], pts[5])   # P2–P6
    vertical_2 = dist.euclidean(pts[2], pts[4])   # P3–P5
    horizontal = dist.euclidean(pts[0], pts[3])   # P1–P4

    if horizontal < 1e-6:
        return 1.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def play_alert_beep() -> None:
    """
    Non-blocking audible alert.
    Uses winsound on Windows; terminal bell elsewhere.
    """
    def _beep():
        try:
            import winsound
            for _ in range(4):
                winsound.Beep(1000, 300)
                time.sleep(0.08)
        except ImportError:
            sys.stdout.write("\a\a\a\a")
            sys.stdout.flush()

    threading.Thread(target=_beep, daemon=True).start()


# =============================================================================
# Twilio Call Manager
# =============================================================================

class TwilioCallManager:
    """
    Wraps Twilio API. All network I/O is offloaded to a daemon thread so
    the video loop is never stalled.
    """

    def __init__(self, cfg: Config, test_mode: bool = False):
        self._cfg = cfg
        self._test_mode = test_mode
        self._last_call_time: float = 0.0
        self._calling: bool = False

        if not test_mode:
            self._validate_credentials()
            self._client = TwilioClient(cfg.TWILIO_ACCOUNT_SID, cfg.TWILIO_AUTH_TOKEN)
        else:
            self._client = None
            log.info("TEST MODE – Twilio calls will be simulated, not placed.")

    # ── Credentials check ────────────────────────────────────────────────────
    def _validate_credentials(self) -> None:
        missing = [
            k for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
                        "TWILIO_FROM_NUMBER",  "TWILIO_TO_NUMBER")
            if not getattr(self._cfg, k)
        ]
        if missing:
            raise ValueError(
                f"Missing Twilio credentials in .env: {', '.join(missing)}\n"
                "Add them to a .env file or run with --test."
            )

    # ── Public API ───────────────────────────────────────────────────────────
    def can_call(self) -> bool:
        elapsed = time.time() - self._last_call_time
        return (not self._calling) and (elapsed >= self._cfg.CALL_COOLDOWN_SECONDS)

    def trigger_call(self) -> None:
        """Fire a call; silently drops if within the cooldown window."""
        if not self.can_call():
            rem = self.cooldown_remaining
            log.info("Call suppressed – cooldown active (%.0fs left).", rem)
            return

        self._calling = True
        self._last_call_time = time.time()
        threading.Thread(target=self._place_call, daemon=True).start()

    @property
    def cooldown_remaining(self) -> float:
        return max(0.0, self._cfg.CALL_COOLDOWN_SECONDS - (time.time() - self._last_call_time))

    # ── Internal call (runs in thread) ───────────────────────────────────────
    def _place_call(self) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            if self._test_mode:
                log.info(
                    "[SIMULATED CALL] %s  from=%s  to=%s",
                    ts, self._cfg.TWILIO_FROM_NUMBER, self._cfg.TWILIO_TO_NUMBER,
                )
                log.info("[SIMULATED MSG ] %s", self._cfg.TWIML_MESSAGE)
                time.sleep(2)
            else:
                twiml = (
                    f"<Response>"
                    f"<Say voice='alice'>{self._cfg.TWIML_MESSAGE}</Say>"
                    f"</Response>"
                )
                call = self._client.calls.create(
                    to=self._cfg.TWILIO_TO_NUMBER,
                    from_=self._cfg.TWILIO_FROM_NUMBER,
                    twiml=twiml,
                )
                log.info("Twilio call placed at %s – SID: %s", ts, call.sid)
        except TwilioRestException as exc:
            log.error("Twilio API error: %s", exc)
        except Exception as exc:
            log.error("Unexpected call error: %s", exc)
        finally:
            self._calling = False


# =============================================================================
# Drowsiness State Machine
# =============================================================================

class DrowsinessMonitor:
    """
    Simple state machine:
      AWAKE → EYES_CLOSING (timer runs) → DROWSY (alert fires once per event)
    Eyes opening resets back to AWAKE.
    """

    def __init__(self, cfg: Config):
        self._cfg = cfg
        self.consec_closed_frames: int = 0
        self.eyes_closed_since:    float = 0.0
        self.is_drowsy:            bool  = False
        self.alert_triggered:      bool  = False

    def update(self, avg_ear: float) -> bool:
        """
        Feed the latest average EAR.
        Returns True on the *first* frame the drowsy threshold is crossed.
        """
        if avg_ear < self._cfg.EAR_THRESHOLD:
            self.consec_closed_frames += 1

            # Start the timer exactly when the debounce count is reached
            if self.consec_closed_frames == self._cfg.EAR_CONSEC_FRAMES:
                self.eyes_closed_since = time.time()

            if self.consec_closed_frames >= self._cfg.EAR_CONSEC_FRAMES:
                duration = time.time() - self.eyes_closed_since
                self.is_drowsy = duration >= self._cfg.DROWSY_SECONDS

                if self.is_drowsy and not self.alert_triggered:
                    self.alert_triggered = True
                    return True   # ← alert fires exactly once per event
        else:
            # Eyes opened → full reset
            self.consec_closed_frames = 0
            self.is_drowsy            = False
            self.alert_triggered      = False

        return False

    def reset(self) -> None:
        """Manually reset (R key)."""
        self.consec_closed_frames = 0
        self.eyes_closed_since    = 0.0
        self.is_drowsy            = False
        self.alert_triggered      = False
        log.info("Drowsiness timer manually reset.")

    @property
    def closed_seconds(self) -> float:
        if self.consec_closed_frames >= self._cfg.EAR_CONSEC_FRAMES:
            return time.time() - self.eyes_closed_since
        return 0.0


# =============================================================================
# HUD / On-Screen Display
# =============================================================================

def draw_hud(
    frame:        np.ndarray,
    avg_ear:      float,
    monitor:      DrowsinessMonitor,
    call_manager: TwilioCallManager,
    cfg:          Config,
    test_mode:    bool,
) -> None:
    """Render status overlay onto the frame (in-place)."""
    h, w = frame.shape[:2]

    # ── Eye status label ─────────────────────────────────────────────────────
    if monitor.is_drowsy:
        label, color = "!! DROWSY ALERT !!", (0, 0, 255)
    elif monitor.consec_closed_frames >= cfg.EAR_CONSEC_FRAMES:
        label = f"Eyes Closed: {monitor.closed_seconds:.1f}s / {cfg.DROWSY_SECONDS:.1f}s"
        color = (0, 140, 255)         # orange
    else:
        label, color = "Eyes Open", (0, 210, 0)   # green

    cv2.putText(frame, label, (10, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

    # ── EAR readout ──────────────────────────────────────────────────────────
    if cfg.SHOW_EAR_VALUE:
        ear_label = f"EAR: {avg_ear:.3f}  (thr: {cfg.EAR_THRESHOLD:.2f})"
        cv2.putText(frame, ear_label, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Progress bar – eye-closed duration ───────────────────────────────────
    if monitor.consec_closed_frames >= cfg.EAR_CONSEC_FRAMES:
        ratio = min(monitor.closed_seconds / cfg.DROWSY_SECONDS, 1.0)
        bar_w = int((w - 20) * ratio)
        bar_color = (0, 0, 255) if ratio >= 1.0 else (0, 140, 255)
        cv2.rectangle(frame, (10, 75), (w - 10, 90), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 75), (10 + bar_w, 90), bar_color, -1)
        cv2.putText(frame, "Drowsy Timer", (13, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Call cooldown ─────────────────────────────────────────────────────────
    cooldown = call_manager.cooldown_remaining
    if cooldown > 0:
        cd_text = f"Call cooldown: {cooldown:.0f}s"
        cv2.putText(frame, cd_text, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 200, 255), 1, cv2.LINE_AA)

    # ── Test mode badge ───────────────────────────────────────────────────────
    if test_mode:
        cv2.rectangle(frame, (w - 150, 5), (w - 5, 28), (50, 50, 0), -1)
        cv2.putText(frame, "TEST MODE", (w - 145, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # ── Keybind bar ───────────────────────────────────────────────────────────
    cv2.putText(frame, "Q: Quit   R: Reset timer", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    # ── Full-width alert banner ───────────────────────────────────────────────
    if monitor.is_drowsy:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 52), (w, h - 28), (0, 0, 180), -1)
        alpha = 0.55 + 0.20 * abs(time.time() % 1.0 - 0.5)   # subtle pulse
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, "  ! CALLING OWNER NOW !",
                    (w // 2 - 140, h - 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)


def draw_eye_contour(
    frame:       np.ndarray,
    landmarks:   list,
    eye_indices: list,
    color:       tuple,
    image_w:     int,
    image_h:     int,
) -> None:
    """Draw the 6 EAR sample points and connect them as a contour."""
    pts = [
        (int(landmarks[i].x * image_w), int(landmarks[i].y * image_h))
        for i in eye_indices
    ]
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1, cv2.LINE_AA)
    # Draw outline: top arc P2-P3, corners P1 P4, bottom arc P5-P6
    outline = [pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[0]]
    cv2.polylines(frame, [np.array(outline, np.int32)], False, color, 1, cv2.LINE_AA)


# =============================================================================
# Main Application
# =============================================================================

def run(test_mode: bool = False) -> None:
    """
    Entry point: opens webcam, runs the detection loop, handles key input,
    and releases all resources on exit.
    """
    cfg = Config()

    # ── 1. Ensure model file ─────────────────────────────────────────────────
    model_path = ensure_model(cfg)

    # ── 2. Twilio call manager ───────────────────────────────────────────────
    try:
        call_manager = TwilioCallManager(cfg, test_mode=test_mode)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # ── 3. Drowsiness monitor ────────────────────────────────────────────────
    monitor = DrowsinessMonitor(cfg)

    # ── 4. MediaPipe Face Landmarker (Tasks API) ─────────────────────────────
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    fl_options   = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(fl_options)

    # ── 5. Webcam ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    if not cap.isOpened():
        log.error("Cannot open camera index %d.", cfg.CAMERA_INDEX)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    log.info("Drowsiness detection running.  Q = quit   R = reset.")
    if test_mode:
        log.info("TEST MODE – no real Twilio calls will be placed.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Dropped frame – retrying …")
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)          # mirror for natural selfie view
            h, w  = frame.shape[:2]

            # ── MediaPipe inference ──────────────────────────────────────────
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result    = landmarker.detect(mp_image)

            avg_ear = 1.0    # safe default when no face is found

            if result.face_landmarks:
                # face_landmarks[0] is a flat list of 478 NormalizedLandmark objects
                lm = result.face_landmarks[0]

                left_ear  = compute_ear(lm, LEFT_EYE_IDX,  w, h)
                right_ear = compute_ear(lm, RIGHT_EYE_IDX, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0

                if cfg.SHOW_LANDMARKS:
                    eye_color = (0, 0, 255) if avg_ear < cfg.EAR_THRESHOLD else (0, 220, 0)
                    draw_eye_contour(frame, lm, LEFT_EYE_IDX,  eye_color, w, h)
                    draw_eye_contour(frame, lm, RIGHT_EYE_IDX, eye_color, w, h)
            else:
                # No face detected – show a notice
                cv2.putText(frame, "No face detected", (10, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 80, 255), 2)

            # ── State machine update ─────────────────────────────────────────
            alert_fired = monitor.update(avg_ear)

            if alert_fired:
                log.warning(
                    "DROWSY ALERT – eyes closed %.1fs. Triggering beep + call.",
                    monitor.closed_seconds,
                )
                play_alert_beep()
                call_manager.trigger_call()

            # ── Render HUD ───────────────────────────────────────────────────
            draw_hud(frame, avg_ear, monitor, call_manager, cfg, test_mode)
            cv2.imshow("Driver Drowsiness Detection", frame)

            # ── Key input ────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or Esc
                log.info("Quit.")
                break
            elif key in (ord("r"), ord("R")):
                monitor.reset()

    except KeyboardInterrupt:
        log.info("Interrupted by keyboard.")
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        log.info("All resources released. Goodbye.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Driver Drowsiness Detection with Twilio Alert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--test", action="store_true", default=False,
        help="Simulate mode – detects drowsiness but does NOT place a real call.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(test_mode=args.test)
