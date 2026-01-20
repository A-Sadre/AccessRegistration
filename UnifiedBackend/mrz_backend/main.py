import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import re
import easyocr
from datetime import datetime


app = FastAPI(title="Passport Extractor (MRZ + Face)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# EasyOCR reader (English is enough for MRZ)
reader = easyocr.Reader(["en"], gpu=False)

MRZ_ALLOWED = re.compile(r"[^A-Z0-9<]")


def to_b64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def clean_mrz(s: str) -> str:
    s = (s or "").upper().strip()
    s = MRZ_ALLOWED.sub("", s)
    return s


def mrz_check_digit(data: str) -> str:
    values = {
        **{str(i): i for i in range(10)},
        **{chr(ord("A") + i): 10 + i for i in range(26)},
        "<": 0,
    }
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        total += values.get(ch, 0) * weights[i % 3]
    return str(total % 10)


def fmt_date(yyMMdd: str, kind: str) -> str:
    """
    yyMMdd -> YYYY-MM-DD
    kind: "birth" or "expiry"

    Fixes the common issue where expiry like 32 becomes 1932 instead of 2032.
    """
    if len(yyMMdd) != 6 or not yyMMdd.isdigit():
        return ""
    yy = int(yyMMdd[0:2])
    mm = int(yyMMdd[2:4])
    dd = int(yyMMdd[4:6])
    if mm < 1 or mm > 12 or dd < 1 or dd > 31:
        return ""

    now_year = datetime.utcnow().year

    if kind == "expiry":
        # expiry is typically in the future and usually 20xx
        year = 2000 + yy

        # If OCR produced something that becomes "too far" in future, fall back to 19xx
        if year > now_year + 20:
            year = 1900 + yy

        # If still ended up far in the past (e.g., 1932), force 20xx
        if year < now_year - 5:
            year = 2000 + yy

    else:
        # birth: pick a century that yields a reasonable age range (0..120)
        y1900 = 1900 + yy
        y2000 = 2000 + yy
        age1900 = now_year - y1900
        age2000 = now_year - y2000

        if 0 <= age1900 <= 120:
            year = y1900
        elif 0 <= age2000 <= 120:
            year = y2000
        else:
            # fallback pivot
            year = y1900 if yy > (now_year % 100) else y2000

    return f"{year:04d}-{mm:02d}-{dd:02d}"


def parse_td3(line1: str, line2: str):
    """
    TD3 (passport) parsing:
    line1: 44 chars
    line2: 44 chars
    Returns: (parsed_fields_dict, checks_dict)
    """
    # --- names ---
    names_field = line1[5:44]
    parts = names_field.split("<<", 1)
    surname_raw = parts[0] if parts else ""
    given_raw = parts[1] if len(parts) > 1 else ""
    last_name = " ".join([p for p in surname_raw.split("<") if p]).strip()
    first_name = " ".join([p for p in given_raw.split("<") if p]).strip()

    # --- fields ---
    doc_num = line2[0:9]
    doc_cd = line2[9:10]
    nationality = line2[10:13]
    birth = line2[13:19]
    birth_cd = line2[19:20]
    sex = line2[20:21]
    expiry = line2[21:27]
    expiry_cd = line2[27:28]
    optional = line2[28:42]
    final_cd = line2[43:44]

    checks = {
        "doc": (mrz_check_digit(doc_num) == doc_cd),
        "birth": (mrz_check_digit(birth) == birth_cd),
        "expiry": (mrz_check_digit(expiry) == expiry_cd),
        "final": (
            mrz_check_digit(doc_num + doc_cd + birth + birth_cd + expiry + expiry_cd + optional)
            == final_cd
        ),
    }

    parsed = {
        "firstName": first_name,
        "lastName": last_name,
        "documentNumber": doc_num.replace("<", ""),
        "nationality": nationality,
        "dateOfBirth": fmt_date(birth, "birth"),
        "sex": sex.replace("<", ""),
        "expirationDate": fmt_date(expiry, "expiry"),
    }

    return parsed, checks


def ocr_mrz_best(mrz_roi_bgr: np.ndarray):
    variants = []

    variants.append(("base", mrz_roi_bgr))

    gray = cv2.cvtColor(mrz_roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))

    thr = cv2.adaptiveThreshold(
        g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(("thr", cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)))

    inv = 255 - thr
    variants.append(("inv_thr", cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)))

    best = None

    for name, img in variants:
        # EasyOCR returns: [ [bbox, text, conf], ... ]
        results = reader.readtext(img, detail=1, paragraph=False)

        # sort by Y
        lines = []
        for bbox, text, conf in results:
            y = float(bbox[0][1])
            t = clean_mrz(text)
            if "<" in t and len(t) >= 30:
                lines.append((y, t, float(conf)))

        lines.sort(key=lambda x: x[0])
        if len(lines) < 2:
            continue

        l1 = (lines[-2][1][:44]).ljust(44, "<")
        l2 = (lines[-1][1][:44]).ljust(44, "<")

        # ✅ Don't crash on bad candidates
        try:
            parsed, checks = parse_td3(l1, l2)
        except Exception:
            continue

        passed = sum(1 for v in checks.values() if v)
        avg_conf = (lines[-2][2] + lines[-1][2]) / 2.0
        score = avg_conf + passed * 0.8

        cand = {
            "variant": name,
            "lines": [l1, l2],
            "checks": checks,
            "passed": passed,
            "avg_conf": avg_conf,
            "fields": parsed,
            "score": score,
        }

        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


def extract_face(img_bgr: np.ndarray):
    # Uses OpenCV Haar cascade (fast + stable)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return {"ok": False}

    # pick largest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    # add padding
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    face = img_bgr[y1:y2, x1:x2]

    return {
        "ok": True,
        "confidence": None,
        "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
        "b64_jpeg": to_b64_jpeg(face, 92),
    }
    


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Invalid image"}

    # upscale small images
    h, w = img.shape[:2]
    if w < 1100:
        scale = 1100 / w
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    h, w = img.shape[:2]
    mrz_roi = img[int(h * 0.65) : h, 0:w]

    mrz = ocr_mrz_best(mrz_roi)
    face = extract_face(img)

    if not mrz:
        return {"ok": False, "error": "MRZ not found", "face": face}

    return {
        "ok": True,
        "mrz": {
            "lines": mrz["lines"],
            "variant": mrz["variant"],
            "avg_conf": mrz["avg_conf"],
            "checks": mrz["checks"],
            "passed": mrz["passed"],
            "total": 4,
        },
        "fields": mrz["fields"],
        "face": face,
    }
