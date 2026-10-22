

# %%
import cv2, threading, time
from retinaface import RetinaFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # don't queue old frames

SCALE = 0.5                               # detect on a half-size image
latest_frame, latest_boxes = None, []
lock = threading.Lock()
running = True

def detector_loop():
    global latest_boxes
    while running:
        with lock:
            f = None if latest_frame is None else latest_frame.copy()
        if f is None:
            time.sleep(0.01)
            continue
        small = cv2.resize(f, None, fx=SCALE, fy=SCALE)
        faces = RetinaFace.detect_faces(small)
        boxes = []
        if isinstance(faces, dict):
            for face in faces.values():
                boxes.append([int(v / SCALE) for v in face["facial_area"]])
        latest_boxes = boxes

t = threading.Thread(target=detector_loop, daemon=True)
t.start()

try:
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        with lock:
            latest_frame = frame

        for x1, y1, x2, y2 in latest_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    running = False
    t.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()


