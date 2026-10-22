import hashlib
import json
import os
import sys
import time
import traceback

proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr


# On my version, Windows 11, a process running in the background can be put in efficiency mode which makes it much slower.
# Ask Windows not to throttle this process
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


t0 = time.time()

import numpy as np
from deepface import DeepFace

print(
    f"TIMING: DeepFace imported in {time.time() - t0:.1f}s",
    flush=True
)


MODEL = "ArcFace"
DETECTOR = "retinaface"
COMPARE_WITH_OLD = False

# Same threshold DeepFace itself uses for ArcFace with cosine distance, since it worked well with different cases
try:
    from deepface.modules import verification as deepface_verification

    THRESHOLD = float(
        deepface_verification.find_threshold(MODEL, "cosine")
    )

except Exception:
    THRESHOLD = 0.68

print(f"Threshold used: {THRESHOLD}", flush=True)


class FaceError(Exception):
    """No usable face in one of the two pictures."""

    def __init__(self, which, message):
        super().__init__(message)
        self.which = which


def send(message):

    proto.write(json.dumps(message, ensure_ascii=False) + "\n")
    proto.flush()


def file_key(path):
    #Identify a picture by its content

    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def embeddings(path, which):
    #Face signature found in pic

    try:
        reps = DeepFace.represent(
            img_path=path,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )

    except Exception as e:

        msg = str(e)

        if e.__cause__ is not None:
            msg += " | " + str(e.__cause__)

        raise FaceError(which, msg)

    return [np.array(r["embedding"], dtype=float) for r in reps]


def cosine_distance(a, b):

    return 1.0 - float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


# Face signature of the last prepared ID photo: key with embedding
prepared = {}


def id_embeddings(id_pic):

    key = file_key(id_pic)

    if key not in prepared:

        embs = embeddings(id_pic, "id")

        prepared.clear()
        prepared[key] = embs

    return prepared[key]


def prepare(id_pic):

    t = time.time()

    try:
        id_embeddings(id_pic)

        print(
            f"TIMING: ID photo prepared in {time.time() - t:.1f}s",
            flush=True
        )

    except FaceError as e:

        # The verif request reports error
        print("ID photo could not be prepared:", e, flush=True)


def verify_fast(id_pic, selfie):

    t = time.time()

    try:
        id_embs = id_embeddings(id_pic)
        selfie_embs = embeddings(selfie, "selfie")

    except FaceError as e:

        print(
            f"TIMING: verification failed ({e.which}) "
            f"after {time.time() - t:.1f}s",
            flush=True
        )

        return {"ok": False, "error": str(e), "which": e.which}

    # best match among all the faces found
    distance = min(
        cosine_distance(a, b)
        for a in id_embs
        for b in selfie_embs
    )

    print(
        f"TIMING: verification (selfie only) {time.time() - t:.1f}s",
        flush=True
    )

    return {
        "ok": True,
        "verified": bool(distance <= THRESHOLD),
        "distance": float(distance),
        "threshold": float(THRESHOLD),
    }


def compare_with_old(id_pic, selfie, new_result):
    """Run the old full DeepFace comparison and log both results."""

    try:
        t = time.time()

        obj = DeepFace.verify(
            id_pic,
            selfie,
            model_name=MODEL,
            detector_backend=DETECTOR,
        )

        print(
            f"COMPARE: new distance {new_result['distance']:.4f} "
            f"(verified {new_result['verified']}) | "
            f"old distance {float(obj['distance']):.4f} "
            f"(verified {bool(obj['verified'])}) | "
            f"old method took {time.time() - t:.1f}s",
            flush=True
        )

    except Exception as e:
        print("COMPARE failed:", e, flush=True)


try:
    t1 = time.time()

    blank = np.full((300, 300, 3), 127, dtype=np.uint8)

    DeepFace.verify(
        blank,
        blank,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )

    DeepFace.represent(
        img_path=blank,
        model_name=MODEL,
        detector_backend=DETECTOR,
        enforce_detection=False,
    )

    print(
        f"TIMING: models warmed up in {time.time() - t1:.1f}s",
        flush=True
    )

except Exception as e:
    print("Warm-up failed (first request will be slower):", e, flush=True)


send({"ready": True})


for line in sys.stdin:

    line = line.strip()

    if not line:
        continue

    cmd = "verify"

    try:
        request = json.loads(line)

        cmd = request.get("cmd", "verify")

        if cmd == "prepare":

            # No answer is sent so the app doesnt wait 
            prepare(request["id"])
            continue

        result = verify_fast(request["id"], request["selfie"])

        send({"result": result})

        if COMPARE_WITH_OLD and result.get("ok"):
            compare_with_old(request["id"], request["selfie"], result)

    except Exception:

        tb = traceback.format_exc()

        if cmd == "prepare":
            print("Prepare failed:", tb, flush=True)

        else:
            send({"error": tb})