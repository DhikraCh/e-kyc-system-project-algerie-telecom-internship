# eKYC System

An electronic Know Your Customer (eKYC) prototype developed during an internship at Algérie Télécom. It verifies a person's identity in one Streamlit application by combining four steps:

1. **ID card capture**: take a photo or upload one; the card is detected and cropped.
2. **OCR and validation**: Arabic and English text is read with PaddleOCR, the fields are extracted, and cross-field checks (dates, card number, national ID number) give a decision: `ACCEPTED`, `NEEDS REVIEW` or `REJECTED`.
3. **Liveness detection**: random challenges (blink, turn the head left, turn the head right) with MediaPipe, shown live in the page, then a selfie is taken.
4. **Face verification**: the selfie is compared with the portrait on the ID card using DeepFace (ArcFace + RetinaFace). The three results give the final decision: `VERIFIED`, `NEEDS MANUAL REVIEW` or `REJECTED`.

> NB: This is a prototype, not a certified identity verification product. See [Limitations](#limitations).

## How it works

The Streamlit page (`app.py`) does not run any model itself. It starts three long-running worker processes when the page opens. Each worker loads its model once and keeps it in memory, so each verification only pays the processing time.

```
Browser  <->  Streamlit app (app.py)
                 |-- OCR worker       (ocr_worker.py + ocr.py)     own virtual environment
                 |-- Face worker      (face_worker.py)             DeepFace, ArcFace, RetinaFace
                 |-- Liveness worker  (liveness_worker.py)         MediaPipe + camera
```

- The OCR runs in its **own virtual environment** (`.venv_ocr`) because PaddleOCR needs package versions that conflict with the other libraries.
- Each worker writes its messages and timings to its own log file (`ocr_worker.log`, `face_worker.log`, `liveness_worker.log`).

Measured on a laptop (CPU only, 12 logical cores): OCR about 3 s (might go to 5 s), face comparison about 2 s.

## Project files

| File | Role |
|---|---|
| `app.py` | Streamlit application: pages, steps, session state, starts the workers. |
| `squarecam.py` | Detects the ID card in the picture and crops it. |
| `ocr.py` | OCR logic: field extraction, validation checks, OCR decision. |
| `ocr_worker.py` | Runs the OCR as a background process. |
| `face_worker.py` | Runs the face verification as a background process. |
| `liveness_worker.py` | Runs the liveness check as a background process. |
| `livdet.py` | Old liveness script with its own camera window; used only as a fallback. |
| `face_landmarker.task` | MediaPipe face landmark model used by the liveness check. |

## Requirements

- Windows 10/11 (developed and tested on Windows 11; the camera code uses DirectShow and some speed settings are Windows-specific)
- Python 3
- Internet access on the first run: PaddleOCR and DeepFace download their models once

## Installation

Two environments are needed: the OCR's and the main one. In PowerShell, from the project folder:

```powershell
# 1. OCR environment (must be named .venv_ocr, same directory as app.py)
python -m venv .venv_ocr
.venv_ocr\Scripts\python.exe -m pip install -r requirements_ocr.txt

# 2. Main environment (Streamlit, face and liveness libraries)
python -m pip install -r requirements.txt
```

Then open `app.py` and set `ML_PYTHON` (near the top) to the full path of the Python that has the main requirements installed. The face and liveness workers are started with it. Example:

```python
ML_PYTHON = Path(r"C:\Users\<your_user>\AppData\Local\Programs\Python\Python313\python.exe")
```

## Run

```powershell
streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501) and click **Start Verification**.

## Settings

At the top of `app.py`:

| Setting | Default | Meaning |
|---|---|---|
| `STOP_IF_OCR_REJECTED` | `False` | `True` stops the process when the ID card is rejected. Use `True` for a final product, `False` while testing. |
| `INTEGRATED_LIVENESS` | `True` | `True` shows the camera inside the page. `False` uses the old separate camera window (`livdet.py`). |

In `face_worker.py`, set `COMPARE_WITH_OLD = False` to skip the extra comparison that was only used to check the faster method.

## Privacy

All processing is local: no image is sent to an external service. While running, the app saves the ID card, the portrait and the selfie as image files in the project folder, and the `.gitignore` keeps them (and the logs) out of the repository.

## Limitations

- Tuned for one Arabic/latin written Algerian ID card layout.
- Liveness is based on blink and head-turn challenges; a video replay (if happens to agree with the random instructions) or 3D mask may still pass. A passive anti-spoofing model would improve it.
- One camera frame is used for the face comparison.
- No database, audit log or encrypted storage; images are kept as local files.
- One user at a time.
- Cases marked `NEEDS MANUAL REVIEW` have no review interface implemented yet.
- Tested on Windows with a CPU only, on a small number of cards.

## Technologies

Python, Streamlit, OpenCV, PaddleOCR (PP-OCRv5), MediaPipe Face Landmarker, DeepFace (ArcFace), RetinaFace.
