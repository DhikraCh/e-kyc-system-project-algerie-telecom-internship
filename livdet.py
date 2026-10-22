# %%
import urllib.request

#url = ("https://storage.googleapis.com/mediapipe-models/"
 #      "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
#urllib.request.urlretrieve(url, "face_landmarker.task")
# %% [markdown]
# ## Creating the landmarker 

# %%
import cv2, time, random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=2,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

# %% [markdown]
# ## Landmarks indices

# %%
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
FACE_EDGE_A, FACE_EDGE_B = 234, 454    

def get_points(result, w, h):
    lm = result.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm])

# %% [markdown]
# ## EAR 

# %%
def eye_aspect_ratio(pts, idx):
    p1, p2, p3, p4, p5, p6 = pts[idx]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    return vertical / horizontal

def get_ear(pts):
    return (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2

# %% [markdown]
# ## Blink Detector 

# %%
EAR_THRESH = 0.21     # to experiment with
EAR_FRAMES = 2        

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

# %% [markdown]
# ## Head turn

# %%
def yaw_ratio(pts):
    xa, xb = pts[FACE_EDGE_A][0], pts[FACE_EDGE_B][0]
    left_edge, right_edge = min(xa, xb), max(xa, xb)   #defining left and right 
    nose = pts[NOSE_TIP][0]
    return (nose - left_edge) / (right_edge - nose)

TURN_LEFT_MAX  = 0.6    
TURN_RIGHT_MIN = 1.6    
HOLD_FRAMES    = 5      #to experiment with

# %%
CHALLENGES = ["blink", "turn_left", "turn_right"]
TIME_LIMIT = 8     

def new_session():
    order = CHALLENGES.copy()
    random.shuffle(order)        
    return {"order": order, "idx": 0, "start": time.time(), "hold": 0}

def check_challenge(name, pts, blink_det, state):
    if name == "blink":
        return blink_det.update(get_ear(pts))

    r = yaw_ratio(pts)
    ok = (r < TURN_LEFT_MAX) if name == "turn_left" else (r > TURN_RIGHT_MIN)
    state["hold"] = state["hold"] + 1 if ok else 0
    return state["hold"] >= HOLD_FRAMES

# %% [markdown]
# ## loop

# %%
import os
SAVE_DIR  = "."
SAVE_NAME = "selfie.png"
LOOK_STRAIGHT_SECONDS = 5                    # NEW: how long to wait before taking the picture

landmarker = vision.FaceLandmarker.create_from_options(options)     #landmarker of the loop
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Camera opened:", cap.isOpened())

for i in range(40):
    ret, frame = cap.read()
    if ret and i % 10 == 0:
        print("warm-up frame", i, "brightness:", round(float(frame.mean()), 1))
    time.sleep(0.03)

t0 = time.time()
fails = 0

state = new_session()
blink_det = BlinkDetector()
label, selfie = None, None
straight_start = None                        # NEW: None = challenges still running

while True:
    ret, frame = cap.read()
    if not ret:
        fails += 1
        if fails > 100:
            print("Camera gave no frame 100 times in a row"); break
        continue
    fails = 0

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, int((time.time() - t0) * 1000))

    faces = result.face_landmarks
    info = ""

    if straight_start is not None:                                   # NEW: look-straight phase
        wait_left = LOOK_STRAIGHT_SECONDS - (time.time() - straight_start)
        msg = f"Look straight at the camera: {max(int(wait_left) + 1, 1)}"
        if wait_left <= 0:
            selfie = frame.copy()                                    # CHANGED: picture taken here
            label = "REAL (PASS)"
    else:                                                            # CHANGED: challenges now live here
        current = state["order"][state["idx"]]
        if len(faces) == 1:
            pts = get_points(result, w, h)
            info = f"EAR {get_ear(pts):.2f}  YAW {yaw_ratio(pts):.2f}"

            if check_challenge(current, pts, blink_det, state):
                state["idx"] += 1
                state["start"] = time.time()
                state["hold"] = 0
                if state["idx"] == len(CHALLENGES):
                    straight_start = time.time()                     # CHANGED: start the 5 s wait
                else:
                    current = state["order"][state["idx"]]
            msg = (f"Please: {current}   ({state['idx']}/{len(CHALLENGES)} done)"
                   if straight_start is None else "Great! Look straight at the camera")
        elif len(faces) == 0:
            msg = "No face detected"
        else:
            msg = "Only one person please"

        left = TIME_LIMIT - int(time.time() - state["start"])
        if label is None and straight_start is None and left < 0:    # CHANGED: timeout only in challenges
            label = "FAKE (FAIL)"
        if straight_start is None:
            cv2.putText(frame, f"time left: {max(left, 0)}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if label is not None:
        color = (0, 255, 0) if "PASS" in label else (0, 0, 255)
        cv2.putText(frame, label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Tuning", frame)
        cv2.waitKey(2000)
        break

    cv2.imshow("Tuning", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()
landmarker.close()
print("Result:", label)

if selfie is not None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    selfie = cv2.flip(selfie, 1)
    ok = cv2.imwrite(save_path, selfie)
    print("Saved:" if ok else "SAVE FAILED:", save_path)
else:
    print("No picture saved (test failed or stopped).")

# %%



