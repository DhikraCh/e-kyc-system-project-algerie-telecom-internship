import json
import os
import sys
import traceback

# Keep a private copy of the real stdout for our answers, then point
# stdout at stderr so library messages (even C-level ones) can never
# get mixed into the answers.
proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
os.dup2(2, 1)
sys.stdout = sys.stderr


# On Windows 11, a process running in the background can be put in
# "efficiency mode", which can make the OCR about 3x slower.
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

        # Version 1, control "execution speed", state 0 = never throttle
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


from ocr import run_ocr, get_models


def send(message):

    proto.write(json.dumps(message, ensure_ascii=False) + "\n")
    proto.flush()


get_models()  # load the models once, at startup

send({"ready": True})


for line in sys.stdin:

    path = line.strip()

    if not path:
        continue

    try:
        send({"result": run_ocr(path)})

    except Exception:
        send({"error": traceback.format_exc()})
