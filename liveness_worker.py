import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

os.dup2(2, 1)
sys.stdout = sys.stderr
# Ask Windows not to throttle this process.
if os.name == "nt":

    try:
        import ctypes

        class PowerThrottlingState(ctypes.Structure):
            _fields_ = [
                ("Version", ctypes.c_ulong),
                ("ControlMask", ctypes.c_ulong),
                ("StateMask", ctypes.c_ulong),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        kernel32.GetCurrentProcess.restype = ctypes.c_void_p

        kernel32.SetProcessInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]

        kernel32.SetProcessInformation.restype = ctypes.c_int

        state = PowerThrottlingState(1, 1, 0)

        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(),
            4,  # ProcessPowerThrottling
            ctypes.byref(state),
            ctypes.sizeof(state),
        )

        print("Power throttling opt-out:", bool(ok), flush=True)

    except Exception as e:
        print("Power throttling opt-out failed:", e, flush=True)


t_start = time.time()

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


BASE_DIR = Path(__file__).resolve().parent

FRAME_PATH = BASE_DIR / "liveness_frame.jpg"
STATE_PATH = BASE_DIR / "liveness_state.json"
SELFIE_PATH = BASE_DIR / "selfie.png"
MODEL_PATH = BASE_DIR / "face_landmarker.task"


# The face model, created once

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
)

landmarker = vision.FaceLandmarker.create_from_options(options)

print(
    f"TIMING: liveness models loaded in {time.time() - t_start:.1f}s",
    flush=True
)


#  landmarks and rules

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
FACE_EDGE_A, FACE_EDGE_B = 234, 454

EAR_THRESH = 0.21
EAR_FRAMES = 2

TURN_LEFT_MAX = 0.6
TURN_RIGHT_MIN = 1.6
HOLD_FRAMES = 5

CHALLENGES = ["blink", "turn_left", "turn_right"]
TIME_LIMIT = 8
LOOK_STRAIGHT_SECONDS = 5

#  only take the selfie when the whole face is inside the picture and looking straight the picture is taken after 5 seconds
REQUIRE_FULL_FACE = True

# straight looking centred face yaw around 1
STRAIGHT_MIN, STRAIGHT_MAX = 0.7, 1.4

WARMUP_FRAMES = 10

MAX_SESSION_SECONDS = 90

INSTRUCTIONS = {
    "blink": "blink",
    "turn_left": "turn your head left",
    "turn_right": "turn your head right",
}


def get_points(result, w, h):

    lm = result.face_landmarks[0]

    return np.array([[p.x * w, p.y * h] for p in lm])


def eye_aspect_ratio(pts, idx):

    p1, p2, p3, p4, p5, p6 = pts[idx]

    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)

    return vertical / horizontal


def get_ear(pts):

    return (
        eye_aspect_ratio(pts, LEFT_EYE)
        + eye_aspect_ratio(pts, RIGHT_EYE)
    ) / 2


class BlinkDetector:

    def __init__(self):
        self.closed = 0

    def update(self, ear):

        if ear < EAR_THRESH:
            self.closed += 1
            return False

        blinked = self.closed >= EAR_FRAMES
        self.closed = 0

        return blinked


def yaw_ratio(pts):

    xa, xb = pts[FACE_EDGE_A][0], pts[FACE_EDGE_B][0]
    left_edge, right_edge = min(xa, xb), max(xa, xb)
    nose = pts[NOSE_TIP][0]

    return (nose - left_edge) / (right_edge - nose)


def new_challenges():

    order = CHALLENGES.copy()
    random.shuffle(order)

    return {"order": order, "idx": 0, "start": time.time(), "hold": 0}


def check_challenge(name, pts, blink_det, state):

    if name == "blink":
        return blink_det.update(get_ear(pts))

    r = yaw_ratio(pts)

    ok = (
        (r < TURN_LEFT_MAX)
        if name == "turn_left"
        else (r > TURN_RIGHT_MIN)
    )

    state["hold"] = state["hold"] + 1 if ok else 0

    return state["hold"] >= HOLD_FRAMES


def face_is_fully_visible(pts, w, h):
    """True if the whole face is inside the picture and looks straight."""

    inside = (
        pts[:, 0].min() >= 0
        and pts[:, 0].max() <= w
        and pts[:, 1].min() >= 0
        and pts[:, 1].max() <= h
    )

    return bool(inside and STRAIGHT_MIN <= yaw_ratio(pts) <= STRAIGHT_MAX)


# files for the page

def write_atomic(path, data):

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_bytes(data)

    for _ in range(10):

        try:
            os.replace(tmp, path)
            return

        except PermissionError:
            time.sleep(0.01)


def write_state(session, status, msg="", left=None,
                frame_id=0, label=None, passed=False):

    data = {
        "session": session,
        "status": status,
        "msg": msg,
        "left": left,
        "frame_id": frame_id,
        "label": label,
        "passed": passed,
    }

    write_atomic(
        STATE_PATH,
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    )


clock_start = time.time()
last_ts = 0


def next_timestamp():

    global last_ts

    ts = int((time.time() - clock_start) * 1000)

    last_ts = max(ts, last_ts + 1)

    return last_ts


def run_session(session):

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    label = None
    selfie = None
    passed = False
    msg = ""
    frame_id = 0

    try:

        if not cap.isOpened():

            write_state(
                session, "done",
                msg="The camera could not be opened.",
                label="CAMERA ERROR"
            )

            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        for _ in range(WARMUP_FRAMES):
            cap.read()

        challenges = new_challenges()
        blink_det = BlinkDetector()

        straight_start = None
        fails = 0
        began = time.time()

        while label is None:

            if time.time() - began > MAX_SESSION_SECONDS:

                label = "TIMEOUT (FAIL)"
                msg = "The check took too long."
                break

            ret, frame = cap.read()

            if not ret:

                fails += 1

                if fails > 100:
                    label = "CAMERA ERROR"
                    msg = "The camera stopped giving pictures."
                    break

                continue

            fails = 0

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_img,
                next_timestamp()
            )

            faces = result.face_landmarks
            left = None

            if straight_start is not None:

                # take the pic only when the whole face is visible and straight

                wait_left = LOOK_STRAIGHT_SECONDS - (
                    time.time() - straight_start
                )

                msg = (
                    "Look straight at the camera: "
                    f"{max(int(wait_left) + 1, 1)}"
                )

                if wait_left <= 0:

                    good = False

                    if len(faces) == 1:

                        pts = get_points(result, w, h)

                        good = (
                            not REQUIRE_FULL_FACE
                            or face_is_fully_visible(pts, w, h)
                        )

                    if good:
                        selfie = frame.copy()
                        label = "REAL (PASS)"

                    else:
                        msg = (
                            "Look straight at the camera, "
                            "with your whole face visible"
                        )

            else:

                current = challenges["order"][challenges["idx"]]

                if len(faces) == 1:

                    pts = get_points(result, w, h)

                    if check_challenge(
                        current, pts, blink_det, challenges
                    ):

                        challenges["idx"] += 1
                        challenges["start"] = time.time()
                        challenges["hold"] = 0

                        if challenges["idx"] == len(CHALLENGES):
                            straight_start = time.time()

                        else:
                            current = challenges["order"][
                                challenges["idx"]
                            ]

                    if straight_start is None:

                        msg = (
                            f"Please {INSTRUCTIONS[current]} "
                            f"({challenges['idx']}/{len(CHALLENGES)} done)"
                        )

                    else:
                        msg = "Great! Look straight at the camera"

                elif len(faces) == 0:
                    msg = "No face detected"

                else:
                    msg = "Only one person please"

                left = max(
                    TIME_LIMIT - int(time.time() - challenges["start"]),
                    0
                )

                if (
                    straight_start is None
                    and TIME_LIMIT - int(time.time() - challenges["start"]) < 0
                ):
                    label = "FAKE (FAIL)"
                    msg = "Time is up."

            # Send the picture and the status to the page

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            if ok:
                frame_id += 1
                write_atomic(FRAME_PATH, buffer.tobytes())

            if label is None:

                write_state(
                    session, "running",
                    msg=msg, left=left, frame_id=frame_id
                )

        if label is not None and "PASS" in label and selfie is not None:

            # The picture on screen is mirrored, save the right way round
            passed = bool(
                cv2.imwrite(str(SELFIE_PATH), cv2.flip(selfie, 1))
            )

        write_state(
            session, "done",
            msg=msg, frame_id=frame_id, label=label, passed=passed
        )

    except Exception:

        print(traceback.format_exc(), flush=True)

        write_state(
            session, "done",
            msg="The liveness check failed.",
            label="ERROR"
        )

    finally:
        cap.release()


#  main loop

write_state(0, "idle")

for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    try:
        request = json.loads(line)

        if request.get("cmd") == "start":

            t = time.time()

            run_session(request["session"])

            print(
                f"TIMING: liveness session lasted {time.time() - t:.1f}s",
                flush=True
            )

    except Exception:
        print(traceback.format_exc(), flush=True)