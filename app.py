import os
import json
import subprocess
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

from squarecam import process_image


# Paths are anchored to the folder where app.py lives,so it works from any location
BASE_DIR = Path(__file__).resolve().parent

# OCR (separate environment)
OCR_PYTHON = BASE_DIR / ".venv_ocr" / "Scripts" / "python.exe"
OCR_WORKER = BASE_DIR / "ocr_worker.py"
OCR_LOG = BASE_DIR / "ocr_worker.log"

ML_PYTHON = Path(
    r"C:\Users\dhikra Ch\AppData\Local\Programs\Python\Python313\python.exe"
)
LIVENESS_SCRIPT = BASE_DIR / "livdet.py"
FACE_RUNNER = BASE_DIR / "face_runner.py"
FACE_WORKER = BASE_DIR / "face_worker.py"
FACE_LOG = BASE_DIR / "face_worker.log"


INTEGRATED_LIVENESS = True   # True Liveness detection shown INSIDE the page 
LIVENESS_WORKER = BASE_DIR / "liveness_worker.py"
LIVENESS_LOG = BASE_DIR / "liveness_worker.log"
LIVENESS_FRAME = BASE_DIR / "liveness_frame.jpg"
LIVENESS_STATE = BASE_DIR / "liveness_state.json"

STOP_IF_OCR_REJECTED = True    #true stop the process if id card rejected , true lets the user continue anyway

STEPS = [
    "ID Card",
    "OCR & Information",
    "Liveness Detection",
    "Face Verification",
    "Result",
]


def clean_env():
    #Environment for subprocesses, without variables from Streamlit's

    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    return env


# OCR

@st.cache_resource
def start_ocr_worker():
    #Start the OCR process , so every card after the first one is processed in seconds

    log = open(OCR_LOG, "w", encoding="utf-8")

    # variable not needed by ocr, can slow it 
    env = clean_env()
    env.pop("KMP_DUPLICATE_LIB_OK", None)

    proc = subprocess.Popen(
        [str(OCR_PYTHON), str(OCR_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
        cwd=BASE_DIR,
    )

    return proc, threading.Lock()


def get_worker():
    #Return the running OCR process  or start new one if dead

    proc, lock = start_ocr_worker()

    if proc.poll() is not None:
        start_ocr_worker.clear()
        proc, lock = start_ocr_worker()

    return proc, lock


def log_tail(n=3000, path=OCR_LOG):

    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n:]

    except OSError:
        return ""


def read_message(proc, log_path=OCR_LOG):
    #Read lines from the worker until a result or error arrives

    while True:

        line = proc.stdout.readline()

        if line == "":
            raise RuntimeError(
                "The background process stopped unexpectedly.\n\n"
                "Last lines of its log:\n" + log_tail(path=log_path)
            )

        try:
            msg = json.loads(line)

        except json.JSONDecodeError:
            continue

        if isinstance(msg, dict) and ("result" in msg or "error" in msg):
            return msg


def run_ocr(image_path: Path) -> dict:

    proc, lock = get_worker()

    with lock:

        try:
            proc.stdin.write(str(image_path) + "\n")
            proc.stdin.flush()

        except OSError:
            raise RuntimeError(
                "The OCR process is not running. Try again.\n\n"
                "Last lines of ocr_worker.log:\n" + log_tail()
            )

        msg = read_message(proc)

    if "error" in msg:
        raise RuntimeError(msg["error"])

    return msg["result"]


#  liveness + face

def crop_id_face(id_card_image):
    """Crop the portrait from the cropped ID card (same as main.py)."""

    height, width = id_card_image.shape[:2]

    # Portrait is approximately 28% of ID width and 57% of ID height, positioned at bottom-left.
    face_width = int(width * 0.28)
    face_height = int(height * 0.57)

    # the small margins from the left and bottom
    x = int(width * 0.03)
    y = height - face_height - int(height * 0.09)

    return id_card_image[
        y:y + face_height,
        x:x + face_width
    ]


@st.cache_resource
def start_liveness_worker():
    #Start the liveness process once in the background (mediapipe)

    log = open(LIVENESS_LOG, "w", encoding="utf-8")

    return subprocess.Popen(
        [str(ML_PYTHON), str(LIVENESS_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=clean_env(),
        cwd=BASE_DIR,
    )


def get_liveness_worker():
    #Return the running liveness process or restart if dead

    proc = start_liveness_worker()

    if proc.poll() is not None:
        start_liveness_worker.clear()
        proc = start_liveness_worker()

    return proc


def run_liveness_integrated(frame_box, msg_box) -> dict:
    #Run the liveness check and show the live camera in the page.

    proc = get_liveness_worker()

    selfie_path = BASE_DIR / "selfie.png"

    for old in (selfie_path, LIVENESS_STATE, LIVENESS_FRAME):

        try:
            old.unlink()

        except OSError:
            pass

    session = int(time.time() * 1000)
    clicked = time.time()

    try:
        proc.stdin.write(
            json.dumps({"cmd": "start", "session": session}) + "\n"
        )
        proc.stdin.flush()

    except OSError:
        raise RuntimeError(
            "The liveness process is not running. Try again.\n\n"
            "Last lines of liveness_worker.log:\n"
            + log_tail(path=LIVENESS_LOG)
        )

    last_frame = -1
    last_msg = None
    first_frame_seconds = None
    state = None

    while True:

        if proc.poll() is not None:
            raise RuntimeError(
                "The liveness process stopped unexpectedly.\n\n"
                "Last lines of liveness_worker.log:\n"
                + log_tail(path=LIVENESS_LOG)
            )

        if time.time() - clicked > 150:

            return {
                "passed": False,
                "label": None,
                "selfie": None,
                "note": "The liveness check took too long and was stopped.",
            }

        try:
            state = json.loads(
                LIVENESS_STATE.read_text(encoding="utf-8")
            )

        except (OSError, ValueError):
            time.sleep(0.05)
            continue

        if state.get("session") != session:
            time.sleep(0.05)
            continue

        if state["frame_id"] != last_frame:

            try:
                frame_box.image(LIVENESS_FRAME.read_bytes())
                last_frame = state["frame_id"]

                if first_frame_seconds is None:
                    first_frame_seconds = time.time() - clicked

            except OSError:
                pass

        text = f"### {state['msg']}"

        if state.get("left") is not None:
            text += f"\n\nTime left: {state['left']} s"

        if text != last_msg:
            msg_box.markdown(text)
            last_msg = text

        if state["status"] == "done":
            break

        time.sleep(0.04)

    passed = bool(state.get("passed")) and selfie_path.exists()

    if passed:
        note = ""

    else:
        note = state.get("msg") or "The liveness check did not pass."

    return {
        "passed": passed,
        "label": state.get("label"),
        "selfie": str(selfie_path) if passed else None,
        "note": note,
        "first_frame_seconds": first_frame_seconds,
    }


def run_liveness() -> dict:
    #run livdet.py : second option

    selfie_path = BASE_DIR / "selfie.png"

    if selfie_path.exists():
        selfie_path.unlink()

    try:
        result = subprocess.run(
            [str(ML_PYTHON), str(LIVENESS_SCRIPT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=BASE_DIR,
            env=clean_env(),
            timeout=180,
        )

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "label": None,
            "selfie": None,
            "note": "The liveness check took too long and was stopped.",
        }

    if result.returncode != 0:
        raise RuntimeError(
            "livdet.py failed.\n\n" + result.stderr[-3000:]
        )

    label = None

    for line in result.stdout.splitlines():
        if line.startswith("Result:"):
            label = line.split(":", 1)[1].strip()

    passed = (
        label is not None
        and "PASS" in label
        and selfie_path.exists()
    )

    if passed:
        note = ""

    elif label is None or label == "None":
        note = "The check was closed before it finished."

    else:
        note = f"Result: {label}"

    return {
        "passed": passed,
        "label": label,
        "selfie": str(selfie_path) if passed else None,
        "note": note,
    }


@st.cache_resource
def start_face_worker():
    #start face verif process once and keep it

    log = open(FACE_LOG, "w", encoding="utf-8")

    proc = subprocess.Popen(
        [str(ML_PYTHON), str(FACE_WORKER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=clean_env(),
        cwd=BASE_DIR,
    )

    return proc, threading.Lock()


def get_face_worker():
    #Return the running face process or start new one if died

    proc, lock = start_face_worker()

    if proc.poll() is not None:
        start_face_worker.clear()
        proc, lock = start_face_worker()

    return proc, lock


def prepare_id_face(id_pic_path: Path):
    #prepare the id pic in advance to gain time

    try:
        proc, lock = get_face_worker()

        request = json.dumps(
            {"cmd": "prepare", "id": str(id_pic_path)}
        )

        with lock:
            proc.stdin.write(request + "\n")
            proc.stdin.flush()

    except Exception:
        pass


def run_face_verification(id_pic_path: Path, selfie_path: str) -> dict:
    #compare id photo with the selfie

    proc, lock = get_face_worker()

    request = json.dumps(
        {"id": str(id_pic_path), "selfie": str(selfie_path)}
    )

    with lock:

        try:
            proc.stdin.write(request + "\n")
            proc.stdin.flush()

        except OSError:
            raise RuntimeError(
                "The face verification process is not running. "
                "Try again.\n\nLast lines of face_worker.log:\n"
                + log_tail(path=FACE_LOG)
            )

        msg = read_message(proc, FACE_LOG)

    if "error" in msg:
        raise RuntimeError(msg["error"])

    return msg["result"]



#styling

STYLE = """
<style>
.stApp {
    background: linear-gradient(180deg, #F3F9FF 0%, #DCEBFB 100%);
}
h1, h2, h3, h4 { color: #0B2545 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B2545 0%, #13427A 100%);
}
[data-testid="stSidebar"] * { color: #EAF4FF !important; }
.stButton > button {
    background: #2F7DE1;
    color: white;
    border: 0;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    box-shadow: 0 6px 16px rgba(47, 125, 225, 0.30);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover {
    background: #1D5FBF;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(47, 125, 225, 0.45);
}
.stButton > button:focus:not(:active) { color: white; }
[data-testid="stAlert"] {
    border-radius: 14px;
    border: 1px solid rgba(11, 37, 69, 0.08);
}
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stFileUploader"] section,
[data-testid="stCameraInput"] {
    border-radius: 14px;
}
.home { text-align: center; padding: 2.5rem 0 1rem; }
.home .icon { font-size: 3.2rem; }
.home .title {
    font-size: 2.8rem; font-weight: 800; color: #0B2545;
    margin: 0.3rem 0 0.6rem;
}
.home .text {
    color: #46607F; font-size: 1.1rem;
    max-width: 34rem; margin: 0 auto 1.8rem;
}
</style>
"""

# main page

st.set_page_config(
    page_title="eKYC System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown(STYLE, unsafe_allow_html=True)

# Start loading the OCR models beforehand while user is uploading the pic
start_ocr_worker()
start_face_worker()

if INTEGRATED_LIVENESS:
    start_liveness_worker()


if "verification_started" not in st.session_state:
    st.session_state.verification_started = False

if "current_step" not in st.session_state:
    st.session_state.current_step = "ID Card"

if "cropped_id" not in st.session_state:
    st.session_state.cropped_id = None


# Home page

if not st.session_state.verification_started:

    st.markdown(
        '<div class="home">'
        '<div class="icon">🛡️</div>'
        '<div class="title">eKYC System</div>'
        '<div class="text">Welcome to our eKYC system. Verify your '
        'identity in a few short steps: ID card, liveness check and '
        'face match.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    _, centre, _ = st.columns([1, 1, 1])

    with centre:

        if st.button("Start Verification", use_container_width=True):
            st.session_state.verification_started = True
            st.rerun()


# Verif pages

else:

    # Sidebar

    st.sidebar.title("Verification")

    for number, step in enumerate(STEPS, start=1):

        if step == st.session_state.current_step:
            st.sidebar.markdown(f"**{number}. {step}**")

        else:
            st.sidebar.markdown(f"{number}. {step}")


    #  ID card retrieval

    if st.session_state.current_step == "ID Card":

        st.header("ID Card")

        st.write(
            "Provide your ID card to begin verification."
        )

        st.info(
            "Please place your ID card on a clear, dark background "
            "and make sure the whole card is visible."
        )


        # Choose input method

        input_method = st.radio(
            "How would you like to provide your ID card?",
            ["Take a picture", "Upload a picture"]
        )

        image_path = None


        # Camera

        if input_method == "Take a picture":

            captured_image = st.camera_input(
                "Take a picture of your ID card"
            )

            if captured_image is not None:

                image_path = BASE_DIR / "id_input.jpg"

                with open(image_path, "wb") as file:
                    file.write(captured_image.getbuffer())


        # Upload

        else:

            uploaded_image = st.file_uploader(
                "Upload your ID card",
                type=["jpg", "jpeg", "png"]
            )

            if uploaded_image is not None:

                image_path = BASE_DIR / "id_input.jpg"

                with open(image_path, "wb") as file:
                    file.write(uploaded_image.getbuffer())


        # image processing

        if image_path is not None:

            cropped_id = process_image(
                str(image_path)
            )

            if cropped_id is None:

                st.error(
                    "ID card could not be detected."
                )

                st.write(
                    "Please make sure the entire ID card is visible."
                )

            else:

                st.success(
                    "ID card detected successfully!"
                )

                st.subheader("Cropped ID Card")

                st.image(
                    cropped_id,
                    channels="BGR"
                )

                st.session_state.cropped_id = cropped_id


                if st.button("Continue to OCR →"):

                    # Forget results from any previous card
                    for old in ("ocr_results", "liveness", "face_result"):
                        st.session_state.pop(old, None)


                    st.session_state.current_step = (
                        "OCR & Information"
                    )

                    st.rerun()


    #  OCR running

    elif st.session_state.current_step == "OCR & Information":

        st.header("OCR & Information")

        st.write(
            "OCR is in progress. "
            "Please hold on while we process your information."
        )


        if st.session_state.cropped_id is not None:

            # Run OCR only once, then keep the result in sessionstate
            if "ocr_results" not in st.session_state:

                ocr_path = BASE_DIR / "id_card.jpg"

                # Save cropped ID temporarily

                cv2.imwrite(
                    str(ocr_path),
                    st.session_state.cropped_id
                )

                with st.spinner("Running OCR..."):

                    started = time.time()

                    try:
                        st.session_state.ocr_results = run_ocr(ocr_path)

                        st.session_state.ocr_seconds = (
                            time.time() - started
                        )

                    except Exception as e:
                        st.error("OCR failed.")
                        st.code(str(e))
                        st.stop()

                # OCR is done, face verif starts in the background to gain time
                id_pic_early = BASE_DIR / "id_pic.jpg"

                cv2.imwrite(
                    str(id_pic_early),
                    crop_id_face(st.session_state.cropped_id)
                )

                prepare_id_face(id_pic_early)

            ocr_results = st.session_state.ocr_results


            st.success(
                "OCR completed successfully!"
            )

            if "ocr_seconds" in st.session_state:

                st.caption(
                    f"OCR time: {st.session_state.ocr_seconds:.1f} s"
                )


            # Extracted information

            st.subheader(
                "Extracted Information"
            )

            for key, field in ocr_results["fields"].items():

                st.write(
                    f"**{field['label']}**: "
                    f"{field['value']}"
                )


            # Field validation

            st.subheader(
                "Field Validation"
            )

            for key, field in ocr_results["fields"].items():

                if field["status"] == "OK":

                    st.success(
                        f"{field['label']}: OK"
                    )

                elif field["status"] == "REVIEW":

                    st.warning(
                        f"{field['label']}: REVIEW"
                    )

                else:

                    st.error(
                        f"{field['label']}: {field['status']}"
                    )


            # Cross-field checks

            st.subheader(
                "Cross-field Checks"
            )

            st.write(
                ocr_results["cross_field"]
            )

            # Each individual check

            for check in ocr_results.get("checks", []):

                if check["status"] == "PASS":

                    st.success(
                        f"{check['name']}: PASS"
                    )

                elif check["status"] == "WARN":

                    st.warning(
                        f"{check['name']}: WARN"
                    )

                elif check["status"] == "FAIL":

                    st.error(
                        f"{check['name']}: FAIL"
                    )

                else:

                    st.info(
                        f"{check['name']}: {check['status']}"
                    )


            # Final decision

            st.subheader(
                "Final Decision"
            )

            st.write(
                ocr_results["final_status"]
            )


            st.divider()

            if (
                STOP_IF_OCR_REJECTED
                and ocr_results["final_status"] == "REJECTED"
            ):

                st.error(
                    "The ID card was rejected. "
                    "Verification cannot continue."
                )

            elif st.button("Continue to Liveness Detection →"):

                st.session_state.current_step = "Liveness Detection"
                st.rerun()


    # Liveness detection

    elif st.session_state.current_step == "Liveness Detection":

        st.header("Liveness Detection")

        liveness = st.session_state.get("liveness")

        if liveness is None:

            if INTEGRATED_LIVENESS:

                st.write(
                    "Your camera will start in this page. Follow the "
                    "instructions shown above the video: blink, turn "
                    "your head left and right, then look straight at "
                    "the camera, with your whole face visible, for a "
                    "few seconds."
                )

            else:

                st.write(
                    "A camera window will open on your computer. "
                    "Follow the instructions written on it: blink, turn "
                    "your head left and right, then look straight at the "
                    "camera for a few seconds. Keep the window open until "
                    "it closes by itself."
                )

            if st.button("Start liveness check"):

                if INTEGRATED_LIVENESS:

                    msg_box = st.empty()
                    frame_box = st.empty()

                    msg_box.info("Starting the camera...")

                    try:
                        st.session_state.liveness = (
                            run_liveness_integrated(frame_box, msg_box)
                        )

                    except Exception as e:
                        st.error("The liveness check could not run.")
                        st.code(str(e))
                        st.stop()

                else:

                    with st.spinner(
                        "Liveness check in progress. "
                        "Look at the camera window..."
                    ):

                        try:
                            st.session_state.liveness = run_liveness()

                        except Exception as e:
                            st.error("The liveness check could not run.")
                            st.code(str(e))
                            st.stop()

                st.rerun()

        elif liveness["passed"]:

            st.success("Liveness check passed!")

            if liveness.get("first_frame_seconds") is not None:

                st.caption(
                    "Camera started in "
                    f"{liveness['first_frame_seconds']:.1f} s"
                )

            st.image(
                liveness["selfie"],
                caption="Captured selfie"
            )

            if st.button("Continue to Face Verification →"):

                st.session_state.current_step = "Face Verification"
                st.rerun()

        else:

            st.error("Liveness check failed.")

            if liveness.get("note"):
                st.write(liveness["note"])

            if st.button("Try again"):

                st.session_state.pop("liveness", None)
                st.rerun()


    #  Face verification

    elif st.session_state.current_step == "Face Verification":

        st.header("Face Verification")

        liveness = st.session_state.get("liveness")

        if not liveness or not liveness["passed"]:

            st.error("Liveness detection must be passed first.")

            if st.button("← Back to Liveness Detection"):

                st.session_state.current_step = "Liveness Detection"
                st.rerun()

            st.stop()


        id_pic_path = BASE_DIR / "id_pic.jpg"

        if "face_result" not in st.session_state:

            cv2.imwrite(
                str(id_pic_path),
                crop_id_face(st.session_state.cropped_id)
            )

            with st.spinner(
                "Comparing the ID photo with your selfie..."
            ):

                try:
                    st.session_state.face_result = run_face_verification(
                        id_pic_path,
                        liveness["selfie"]
                    )

                except Exception as e:
                    st.error("Face verification could not run.")
                    st.code(str(e))
                    st.stop()

        face = st.session_state.face_result


        col1, col2 = st.columns(2)

        col1.image(str(id_pic_path), caption="Photo from the ID card")
        col2.image(liveness["selfie"], caption="Selfie")


        if not face.get("ok"):

            if face.get("which") == "selfie":

                st.error(
                    "No face could be detected in the selfie. "
                    "Part of your face may be cut off, or the picture "
                    "may be too dark. Please retake it with your whole "
                    "face visible."
                )

                if st.button("Retake selfie"):

                    st.session_state.pop("liveness", None)
                    st.session_state.pop("face_result", None)
                    st.session_state.current_step = "Liveness Detection"
                    st.rerun()

            elif face.get("which") == "id":

                st.error(
                    "No face could be detected in the photo cropped "
                    "from the ID card. Please retake the ID card "
                    "picture so the portrait is fully visible."
                )

                if st.button("Retake ID card picture"):

                    for old in ("ocr_results", "liveness", "face_result"):
                        st.session_state.pop(old, None)

                    st.session_state.current_step = "ID Card"
                    st.rerun()

            else:

                st.error("Face verification could not be completed.")

                if st.button("Try again"):

                    st.session_state.pop("face_result", None)
                    st.rerun()

            with st.expander("Technical details"):
                st.code(face.get("error", "Unknown error"))

        else:

            if face["verified"]:
                st.success("The faces match.")

            else:
                st.error("The faces do not match.")

            st.write(
                f"Distance: {face['distance']:.3f} "
                f"(threshold: {face['threshold']:.3f}, "
                "lower distance = more similar)"
            )

            if st.button("Continue to Result →"):

                st.session_state.current_step = "Result"
                st.rerun()


    # displaying the Result

    elif st.session_state.current_step == "Result":

        st.header("Result")

        ocr_results = st.session_state.get("ocr_results")
        liveness = st.session_state.get("liveness")
        face = st.session_state.get("face_result")

        ocr_status = (
            ocr_results["final_status"] if ocr_results else "NOT DONE"
        )
        liveness_ok = bool(liveness and liveness["passed"])
        face_ok = bool(face and face.get("ok") and face.get("verified"))


        st.subheader("Summary")

        # OCR

        if ocr_status == "ACCEPTED":
            st.success(f"ID card (OCR): {ocr_status}")

        elif ocr_status == "NEEDS REVIEW":
            st.warning(f"ID card (OCR): {ocr_status}")

        else:
            st.error(f"ID card (OCR): {ocr_status}")

        # Liveness

        if liveness_ok:
            st.success("Liveness detection: PASSED")

        else:
            st.error("Liveness detection: FAILED")

        # Face

        if face_ok:
            st.success("Face verification: MATCH")

        else:
            st.error("Face verification: NO MATCH")


        st.subheader("Final Decision")

        if ocr_status == "REJECTED" or not liveness_ok or not face_ok:

            st.error("REJECTED")

        elif ocr_status == "ACCEPTED":

            st.success("VERIFIED")

        else:

            st.warning("NEEDS MANUAL REVIEW")


        if st.button("Start over"):

            st.session_state.clear()
            st.rerun()