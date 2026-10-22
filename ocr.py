import os
import re
import time
import numpy as np
import cv2
from datetime import datetime, date

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from paddleocr import PaddleOCR, TextDetection, TextRecognition
from rapidfuzz import fuzz


FIELDS = {
    "national_id":       "رقم التعريف الوطني",
    "surname":           "اللقب",
    "first_name":        "الاسم",
    "birth_date":        "تاريخ الميلاد",
    "birth_place":       "مكان الميلاد",
    "gender":            "الجنس",
    "issuing_authority": "سلطة الإصدار",
    "issue_date":        "تاريخ الإصدار",
    "expiry_date":       "تاريخ الانتهاء",
}

# Arabic-Indic digits -> normal digits (the Arabic OCR may output them)
AR_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789"
)


def norm(s):

    for a in "أإآ":
        s = s.replace(a, "ا")

    return s.replace(" ", "").replace("ـ", "")


def match_label(text, threshold=80):

    t = norm(text)

    best_score, best_key, best_n = 0, None, 0

    for key, label in FIELDS.items():

        nl = norm(label)

        for n in (len(nl) - 1, len(nl), len(nl) + 1):

            score = fuzz.ratio(t[:n], nl)

            if score > best_score:
                best_score, best_key, best_n = score, key, n

    if best_score < threshold:
        return None, 0, text

    count, i = 0, 0

    while i < len(text) and count < best_n:

        if text[i] != " ":
            count += 1

        i += 1

    return best_key, best_score, text[i:].strip()


def valid_date(s):

    try:
        datetime.strptime(s, "%Y.%m.%d")
        return True

    except (ValueError, TypeError):
        return False


def looks_real(s, n):

    return (
        bool(s)
        and s.isdigit()
        and len(s) == n
        and len(set(s)) > 1
    )


def check_gender(text):

    if not text:
        return text, False

    options = {
        "ذكر": "ذكر",
        "انثى": "أنثى"
    }

    t = norm(text)

    best = max(
        options,
        key=lambda o: fuzz.ratio(t, o)
    )

    if fuzz.ratio(t, best) >= 60:
        return options[best], True

    return text, False


def find_blood_group(lines):

    for t in lines:

        m = re.search(
            r'(AB|A|B|O)\s*([+-])',
            t
        )

        if m:
            return m.group(1) + m.group(2), True

    return None, False


def check_text(v):

    return v, bool(
        re.fullmatch(
            r'[\u0600-\u06FF ]+',
            v or ''
        )
    )


_MODELS = {}


def get_models():
    """Load the two OCR models once and keep them in memory."""

    if not _MODELS:

        t_load = time.time()

        common = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
            text_det_thresh=0.2,
            text_det_box_thresh=0.4,
            text_detection_model_name="PP-OCRv5_mobile_det",
        )

        _MODELS["ar"] = PaddleOCR(
            text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
            use_textline_orientation=True,
            **common,
        )

        _MODELS["en"] = PaddleOCR(
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            **common
        )

        # Standalone English recognizer, used on the text boxes found
        # by the Arabic pass (fast path)
        _MODELS["rec_en"] = TextRecognition(
            model_name="en_PP-OCRv5_mobile_rec"
        )

        # Standalone box detector + Arabic recognizer (fastest path).
        # If they cannot be created, run_ocr uses the slower methods.
        try:
            _MODELS["det"] = TextDetection(
                model_name="PP-OCRv5_mobile_det",
                enable_mkldnn=False,
            )

            _MODELS["rec_ar"] = TextRecognition(
                model_name="arabic_PP-OCRv5_mobile_rec"
            )

        except Exception as e:
            print("Fastest path models not available:", e, flush=True)

        print(
            f"TIMING: models loaded in {time.time() - t_load:.1f}s",
            flush=True
        )

    return _MODELS["ar"], _MODELS["en"]


def get_rec_en():

    get_models()

    return _MODELS["rec_en"]


def fast_pages(path, img):
    """Fastest path: find the text boxes ONCE, then read all of them
    with the standalone Arabic and English recognizers.

    Returns (page_ar, page_en), both shaped like normal OCR pages."""

    get_models()

    det = _MODELS.get("det")
    rec_ar = _MODELS.get("rec_ar")

    if det is None or rec_ar is None:
        raise RuntimeError("fastest path models are not available")

    h, w = img.shape[:2]

    result = list(det.predict(input=path))[0]

    crops, polys = [], []

    for poly in result["dt_polys"]:

        p = np.array(poly)

        x1 = max(int(p[:, 0].min()) - 4, 0)
        x2 = min(int(p[:, 0].max()) + 4, w)
        y1 = max(int(p[:, 1].min()) - 4, 0)
        y2 = min(int(p[:, 1].max()) + 4, h)

        if x2 - x1 > 5 and y2 - y1 > 5:

            crops.append(img[y1:y2, x1:x2])
            polys.append(poly)

    if not crops:
        raise ValueError("no text boxes found")

    def read(recognizer):

        results = list(
            recognizer.predict(
                input=crops,
                batch_size=len(crops)
            )
        )

        return {
            "rec_texts": [r["rec_text"] for r in results],
            "rec_scores": [float(r["rec_score"]) for r in results],
            "rec_polys": polys,
        }

    return read(rec_ar), read(_MODELS["rec_en"])


_DATE_RE = re.compile(r'(\d{4})[./\-,](\d{2})[./\-,](\d{2})')
_CARD_RE = re.compile(r'(?<!\d)\d{8,10}(?!\d)')
_NID_RE = re.compile(r'(?<!\d)\d{15,20}(?!\d)')


def english_on_boxes(page_ar, img):
    """Run only the English recognizer on the text boxes that the
    Arabic pass already found (no second box search = much faster).

    Returns a dict shaped like a normal OCR page."""

    h, w = img.shape[:2]

    crops, polys = [], []

    for poly in page_ar["rec_polys"]:

        p = np.array(poly)

        x1 = max(int(p[:, 0].min()) - 4, 0)
        x2 = min(int(p[:, 0].max()) + 4, w)
        y1 = max(int(p[:, 1].min()) - 4, 0)
        y2 = min(int(p[:, 1].max()) + 4, h)

        if x2 - x1 > 5 and y2 - y1 > 5:

            crops.append(img[y1:y2, x1:x2])
            polys.append(poly)

    if not crops:
        raise ValueError("no text boxes found")

    results = list(
        get_rec_en().predict(
            input=crops,
            batch_size=len(crops)
        )
    )

    return {
        "rec_texts": [r["rec_text"] for r in results],
        "rec_scores": [float(r["rec_score"]) for r in results],
        "rec_polys": polys,
    }


def digits_complete(texts):
    """True if exactly 3 dates, a card number and a national ID
    were found in the texts."""

    dates = set()
    has_card = False
    has_nid = False

    for t in texts:

        t = t.translate(AR_DIGITS).replace(' ', '')

        for y, m, d in _DATE_RE.findall(t):
            dates.add((y, m, d))

        if _CARD_RE.search(t):
            has_card = True

        if _NID_RE.search(t):
            has_nid = True

    return len(dates) == 3 and has_card and has_nid


def run_ocr(image_path, slow=False):

    ocr_ar, ocr_en = get_models()

    cv_img = cv2.imread(image_path)

    if cv_img is None:
        raise ValueError("Could not read the cropped ID image.")

    cv_img = cv2.copyMakeBorder(
        cv_img,
        60,
        60,
        60,
        60,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    # Resize the card to a fixed width (about 1000 px):
    # not too big and not too small for the text detector
    h, w = cv_img.shape[:2]
    scale = 1000 / w

    cv_img = cv2.resize(
        cv_img,
        None,
        fx=scale,
        fy=scale,
        interpolation=(
            cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        )
    )

    cv2.imwrite(
        "card_padded.png",
        cv_img
    )

    t_a = time.time()

    page_ar = None
    page_en = None
    mode = "fastest"

    # Fastest path: find the text boxes once, then read them with the
    # standalone Arabic and English recognizers.

    if not slow:

        try:
            page_ar, page_en = fast_pages("card_padded.png", cv_img)

            if not digits_complete(
                list(page_ar["rec_texts"]) + page_en["rec_texts"]
            ):
                page_ar, page_en = None, None

        except Exception as e:
            print("Fastest path failed:", e, flush=True)
            page_ar, page_en = None, None

    if page_ar is None:

        # Previous method: all-in-one Arabic pass, then the English
        # recognizer on its boxes.

        mode = "medium"

        page_ar = ocr_ar.predict(
            "card_padded.png"
        )[0]

        try:
            page_en = english_on_boxes(page_ar, cv_img)

            if not digits_complete(
                list(page_ar["rec_texts"]) + page_en["rec_texts"]
            ):
                page_en = None

        except Exception as e:
            print("Fast English path failed:", e, flush=True)
            page_en = None

        if page_en is None:

            # Last resort: the full English pass

            mode = "fallback"

            page_en = ocr_en.predict(
                "card_padded.png"
            )[0]

    t_c = time.time()

    print(
        f"TIMING: total OCR {t_c - t_a:.1f}s ({mode})",
        flush=True
    )


    def items(page, min_score=0.7):

        out = []

        for t, s, b in zip(
            page['rec_texts'],
            page['rec_scores'],
            page['rec_polys']
        ):

            b = np.array(b)

            if s >= min_score and t.strip():

                out.append({
                    'text': t,
                    'score': float(s),
                    'y1': b[:, 1].min(),
                    'y2': b[:, 1].max(),
                    'x1': b[:, 0].min(),
                    'x2': b[:, 0].max()
                })

        return sorted(
            out,
            key=lambda d: (d['y1'] + d['y2']) / 2
        )


    ar = items(page_ar)




    date_re = re.compile(
        r'(\d{4})[./\-,](\d{2})[./\-,](\d{2})'
    )

    card_re = re.compile(
        r'(?<!\d)\d{8,10}(?!\d)'
    )

    nid_re = re.compile(
        r'(?<!\d)\d{15,20}(?!\d)'
    )


    dates = []
    card_numbers = []
    national_ids = []


    # Look for dates and numbers in BOTH OCR results
    # (English first, then Arabic as a backup)

    sources = (
        items(page_en, min_score=0.3)
        + items(page_ar, min_score=0.3)
    )

    for d in sources:

        t = d['text'].translate(AR_DIGITS).replace(' ', '')

        dates += [
            f"{y}.{m}.{dd}"
            for y, m, dd in date_re.findall(t)
        ]

        card_numbers += card_re.findall(t)

        national_ids += nid_re.findall(t)


    dates = sorted(set(dates))

    card_numbers = list(
        dict.fromkeys(card_numbers)
    )

    national_ids = list(
        dict.fromkeys(national_ids)
    )


    digits = {}

    if card_numbers:
        digits['card_number'] = card_numbers[0]

    if national_ids:
        digits['national_id'] = national_ids[0]

    if len(dates) == 3:

        digits['birth_date'] = dates[0]
        digits['issue_date'] = dates[1]
        digits['expiry_date'] = dates[2]

    else:

        print(
            "WARNING: expected 3 dates, found",
            dates
        )


    if len(card_numbers) > 1 or len(national_ids) > 1:

        print(
            "WARNING: several candidates:",
            card_numbers,
            national_ids
        )


    fields = {}
    others = []
    index = {}


    for i, d in enumerate(ar):

        key, match, value = match_label(
            d['text']
        )

        if key is None:

            others.append(
                d['text']
            )

        else:

            index[key] = i

            fields[key] = {
                "label": FIELDS[key],
                "value": value,
                "ocr_score": round(
                    d['score'],
                    2
                ),
                "match": round(match)
            }


    # If a label was found with no value on its line, the value was
    # probably detected as a separate line on the same row, to the
    # LEFT of the label (Arabic is written right to left)

    label_idx = set(index.values())

    for key, idx in index.items():

        if fields[key]["value"]:
            continue

        lab = ar[idx]
        lab_h = lab['y2'] - lab['y1']
        lab_cy = (lab['y1'] + lab['y2']) / 2

        best, best_gap = None, None

        for jdx, cand in enumerate(ar):

            if jdx in label_idx:
                continue

            cand_cy = (cand['y1'] + cand['y2']) / 2

            # must be on the same row
            if abs(cand_cy - lab_cy) > lab_h:
                continue

            # must end to the left of the label
            if cand['x2'] > lab['x1'] + 20:
                continue

            gap = max(0, lab['x1'] - cand['x2'])

            if best is None or gap < best_gap:
                best, best_gap = cand, gap

        if best is not None:

            fields[key]["value"] = best['text'].strip()
            fields[key]["ocr_score"] = round(best['score'], 2)


    if (
        'surname' not in fields
        and 'first_name' in index
    ):

        j = index['first_name'] - 1

        if j >= 0 and j not in index.values():

            parts = ar[j]['text'].split(
                ' ',
                1
            )

            if len(parts) == 2:

                fields['surname'] = {
                    "label": FIELDS['surname'],
                    "value": parts[1],
                    "ocr_score": round(
                        ar[j]['score'],
                        2
                    ),
                    "match": 0
                }

                others = [
                    t
                    for t in others
                    if t != ar[j]['text']
                ]


    # The detector sometimes merges the gender label and its value
    # (and even other labels) into one line. In that case, look for
    # the gender label inside the line and take what follows it.

    if not fields.get('gender', {}).get('value'):

        gender_label = norm(FIELDS['gender'])

        for d in ar:

            t = norm(d['text'])

            pos = t.find(gender_label)

            if pos >= 0 and len(t) > pos + len(gender_label):

                fields['gender'] = {
                    "label": FIELDS['gender'],
                    "value": t[pos + len(gender_label):],
                    "ocr_score": round(d['score'], 2),
                    "match": 0
                }

                break


    results = {}


    nid = digits.get(
        'national_id'
    )

    results['national_id'] = (
        nid,
        looks_real(nid, 18)
    )


    card = digits.get(
        'card_number'
    )

    results['card_number'] = (
        card,
        looks_real(card, 9)
    )


    for k in (
        'birth_date',
        'issue_date',
        'expiry_date'
    ):

        v = digits.get(k)

        results[k] = (
            v,
            valid_date(v)
        )


    results['gender'] = check_gender(
        fields.get(
            'gender',
            {}
        ).get(
            'value',
            ''
        )
    )


    results['blood_group'] = find_blood_group(
        [
            d['text']
            for d in ar
        ]
    )


    for k in (
        'surname',
        'first_name',
        'birth_place'
    ):

        results[k] = check_text(
            fields.get(
                k,
                {}
            ).get(
                'value',
                ''
            )
        )


    def to_date(key):

        v, ok = results.get(
            key,
            (None, False)
        )

        if not ok:
            return None

        y, m, d = map(
            int,
            v.split('.')
        )

        return date(y, m, d)


    birth = to_date('birth_date')
    issue = to_date('issue_date')
    expiry = to_date('expiry_date')

    nid = (
        results['national_id'][0]
        if results['national_id'][1]
        else None
    )

    today = date.today()

    checks = []


    def add(
        name,
        test,
        needs,
        level="FAIL"
    ):

        if any(
            n is None
            for n in needs
        ):

            checks.append(
                (name, "SKIPPED")
            )

        else:

            checks.append(
                (
                    name,
                    "PASS"
                    if test()
                    else level
                )
            )


    add(
        "birth date is before issue date",
        lambda: birth < issue,
        (birth, issue)
    )

    add(
        "issue date is before expiry date",
        lambda: issue < expiry,
        (issue, expiry)
    )

    add(
        "card is valid about 10 years",
        lambda:
            9.5 <=
            (expiry - issue).days / 365.25
            <= 10.5,
        (issue, expiry),
        "WARN"
    )

    add(
        "card is expired by today",
        lambda: expiry >= today,
        (expiry,)
    )

    add(
        "national ID digits 4-5 match birth year",
        lambda:
            nid[3:5]
            == f"{birth.year % 100:02d}",
        (nid, birth),
        "WARN"
    )


    statuses = [
        s
        for _, s in checks
    ]


    if "FAIL" in statuses:

        overall = "REFUSE"

    elif "WARN" in statuses:

        overall = "ACCEPT WITH WARNINGS"

    elif all(
        s == "SKIPPED"
        for s in statuses
    ):

        overall = (
            "NOT CHECKED "
            "(dates or ID missing)"
        )

    else:

        overall = "ACCEPT"


    MIN_SCORE = 0.80

    # Lines from both OCR models, used to find the confidence
    # score of a digit value

    en_lines = (
        items(page_en, min_score=0.0)
        + items(page_ar, min_score=0.0)
    )


    def score_in_english(value):

        if not value:
            return None

        for d in en_lines:

            t = (
                d['text']
                .translate(AR_DIGITS)
                .replace(' ', '')
                .replace('/', '.')
                .replace('-', '.')
                .replace(',', '.')
            )

            if value in t:

                return round(
                    d['score'],
                    2
                )

        return None


    def score_blood():

        for d in ar:

            if re.search(
                r'(AB|A|B|O)\s*([+-])',
                d['text']
            ):

                return round(
                    d['score'],
                    2
                )

        return None


    LABELS_OUT = dict(FIELDS)

    LABELS_OUT["card_number"] = "رقم البطاقة"
    LABELS_OUT["blood_group"] = "فصيلة الدم"


    DIGIT_KEYS = (
        "national_id",
        "card_number",
        "birth_date",
        "issue_date",
        "expiry_date"
    )


    ORDER = [
        "card_number",
        "national_id",
        "surname",
        "first_name",
        "gender",
        "birth_date",
        "birth_place",
        "issue_date",
        "expiry_date",
        "blood_group"
    ]


    record = {}


    for key in ORDER:

        value, valid = results[key]

        if key in DIGIT_KEYS:

            score = score_in_english(value)

        elif key == "blood_group":

            score = score_blood()

        else:

            score = fields.get(
                key,
                {}
            ).get(
                "ocr_score"
            )


        if not value:

            status = "MISSING"

        elif not valid:

            status = "REVIEW"

        elif (
            score is not None
            and score < MIN_SCORE
        ):

            status = "REVIEW"

        else:

            status = "OK"


        record[key] = {
            "label": LABELS_OUT[key],
            "value": value,
            "ocr_score": score,
            "valid": bool(valid),
            "status": status
        }


    # If a field is missing or doubtful, write what the OCR read
    # into ocr_worker.log so the cause can be found

    if any(r["status"] != "OK" for r in record.values()):

        print("DEBUG some fields are not OK. Lines read by the OCR:")

        print("DEBUG Arabic OCR (score  text):")

        for d in items(page_ar, min_score=0.0):
            print(f"  {d['score']:.2f}  {d['text']}")

        print("DEBUG English OCR (score  text):")

        for d in items(page_en, min_score=0.0):
            print(f"  {d['score']:.2f}  {d['text']}")

        print(
            "DEBUG statuses:",
            {k: r["status"] for k, r in record.items()},
            flush=True
        )


    all_fields_ok = all(
        r["status"] == "OK"
        for r in record.values()
    )


    if overall == "REFUSE":

        final = "REJECTED"

    elif (
        all_fields_ok
        and overall in (
            "ACCEPT",
            "ACCEPT WITH WARNINGS"
        )
    ):

        final = "ACCEPTED"

    else:

        final = "NEEDS REVIEW"


    output = {
        "fields": record,
        "cross_field": overall,
        "checks": [
            {"name": name, "status": status}
            for name, status in checks
        ],
        "final_status": final
    }


    # The fastest path may split the lines differently from the
    # all-in-one pipeline. If a field is MISSING, redo everything
    # with the slower, previously validated method.

    if mode == "fastest" and any(
        r["status"] == "MISSING" for r in record.values()
    ):

        print(
            "A field is missing: retrying with the slower method",
            flush=True
        )

        return run_ocr(image_path, slow=True)

    return output