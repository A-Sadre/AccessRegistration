import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import easyocr
from datetime import datetime
import re

app = FastAPI(title="Passport Extractor (MRZ + Face)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

reader = easyocr.Reader(["en"], gpu=False)

# ---------------------------
# Utils
# ---------------------------

def to_b64_jpeg(img_bgr: np.ndarray, quality: int = 90) -> str:
    if img_bgr is None:
        return ""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def resize_for_ocr(img_bgr: np.ndarray, target_w: int = 1200) -> np.ndarray:
    """
    Huge speedup: OCR cost grows fast with pixels.
    Keep MRZ width around 1000-1400px for great accuracy.
    """
    if img_bgr is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if w <= target_w:
        return img_bgr
    s = target_w / float(w)
    return cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------------------------
# MRZ sanitization
# ---------------------------

FILLER_EQUIV = {
    "<",
    "«", "»", "‹", "›",
    "〈", "〉", "＜", "＞",
    "⟨", "⟩", "〈", "〉",
    "﹤", "﹥",
}

def mrz_sanitize_keep_fillers(s: str) -> str:
    s = (s or "").strip().upper()
    out = []
    for ch in s:
        if ch in FILLER_EQUIV:
            out.append("<")
            continue
        if ch == "K":  # common OCR instead of <
            out.append("<")
            continue
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            out.append(ch)
            continue
        out.append("<")
    return "".join(out)

def normalize_mrz_line(s: str) -> str:
    return mrz_sanitize_keep_fillers(s)

def fix_mrz_numeric(s: str) -> str:
    out = []
    for ch in (s or ""):
        c = ch.upper()
        if c.isdigit() or c == "<":
            out.append(c)
        elif c in ("O", "Q", "D"):
            out.append("0")
        elif c in ("I", "L"):
            out.append("1")
        elif c == "Z":
            out.append("2")
        elif c == "S":
            out.append("5")
        elif c == "G":
            out.append("6")
        elif c == "B":
            out.append("8")
        else:
            out.append("<")
    return "".join(out)

def fix_mrz_alpha3(s: str) -> str:
    s = (s or "").upper()
    map_digit_to_letter = {
        "0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B",
    }
    out = []
    for ch in s:
        if "A" <= ch <= "Z":
            out.append(ch)
        elif ch in map_digit_to_letter:
            out.append(map_digit_to_letter[ch])
        else:
            out.append("<")
    return "".join(out)[:3]


# ---------------------------
# Check digits + date
# ---------------------------

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
    if len(yyMMdd) != 6 or not yyMMdd.isdigit():
        return ""
    yy = int(yyMMdd[0:2])
    mm = int(yyMMdd[2:4])
    dd = int(yyMMdd[4:6])
    if mm < 1 or mm > 12 or dd < 1 or dd > 31:
        return ""

    now_year = datetime.utcnow().year
    if kind == "expiry":
        year = 2000 + yy
        if year > now_year + 20:
            year = 1900 + yy
        if year < now_year - 5:
            year = 2000 + yy
    else:
        y1900 = 1900 + yy
        y2000 = 2000 + yy
        age1900 = now_year - y1900
        age2000 = now_year - y2000

        if 0 <= age1900 <= 120:
            year = y1900
        elif 0 <= age2000 <= 120:
            year = y2000
        else:
            year = y1900 if yy > (now_year % 100) else y2000

    return f"{year:04d}-{mm:02d}-{dd:02d}"


# ---------------------------
# TD3 strict validators
# ---------------------------

MRZ_TD3_LINE1_RE = re.compile(r"^P<[A-Z]{3}[A-Z0-9<]{39}$")
MRZ_TD3_LINE2_RE = re.compile(r"^[A-Z0-9<]{44}$")

def repair_td3_line1(l1: str) -> str:
    l1 = normalize_mrz_line(l1)
    if len(l1) >= 4 and l1[0] == "P" and l1[1] != "<" and l1[1:4].isalpha():
        l1 = "P<" + l1[1:]
    return l1.ljust(44, "<")[:44]

def looks_like_td3_line1(l1: str) -> bool:
    l1 = repair_td3_line1(l1)
    return MRZ_TD3_LINE1_RE.match(l1) is not None

def looks_like_td3_line2(l2: str) -> bool:
    l2 = normalize_mrz_line(l2).ljust(44, "<")[:44]
    if MRZ_TD3_LINE2_RE.match(l2) is None:
        return False

    nat = fix_mrz_alpha3(l2[10:13])
    if len(nat) != 3 or "<" in nat:
        return False

    if l2[20:21] not in ("M", "F", "X", "<"):
        return False

    b = fix_mrz_numeric(l2[13:19]).replace("<", "0")
    e = fix_mrz_numeric(l2[21:27]).replace("<", "0")
    if not (b.isdigit() and e.isdigit()):
        return False

    return True

def parse_td3(line1: str, line2: str):
    line1 = repair_td3_line1(line1)
    line2 = normalize_mrz_line(line2).ljust(44, "<")[:44]

    names_field = line1[5:44]
    parts = names_field.split("<<", 1)
    surname_raw = parts[0] if parts else ""
    given_raw = parts[1] if len(parts) > 1 else ""
    last_name = " ".join([p for p in surname_raw.split("<") if p]).strip()
    first_name = " ".join([p for p in given_raw.split("<") if p]).strip()

    doc_num = line2[0:9]
    doc_cd = fix_mrz_numeric(line2[9:10])

    nationality = fix_mrz_alpha3(line2[10:13])

    birth = fix_mrz_numeric(line2[13:19])
    birth_cd = fix_mrz_numeric(line2[19:20])

    sex = line2[20:21]
    sex = sex if sex in ("M", "F", "X", "<") else "<"
    sex_out = sex.replace("<", "")

    expiry = fix_mrz_numeric(line2[21:27])
    expiry_cd = fix_mrz_numeric(line2[27:28])

    optional = fix_mrz_numeric(line2[28:42])
    optional_cd = fix_mrz_numeric(line2[42:43])

    final_cd = fix_mrz_numeric(line2[43:44])

    checks = {
        "doc": (mrz_check_digit(doc_num) == doc_cd),
        "birth": (mrz_check_digit(birth) == birth_cd),
        "expiry": (mrz_check_digit(expiry) == expiry_cd),
        "optional": (mrz_check_digit(optional) == optional_cd),
        "final": (
            mrz_check_digit(
                doc_num + doc_cd +
                birth + birth_cd +
                expiry + expiry_cd +
                optional + optional_cd
            ) == final_cd
        ),
    }

    parsed = {
        "firstName": first_name,
        "lastName": last_name,
        "documentNumber": doc_num.replace("<", ""),
        "nationality": nationality.replace("<", ""),
        "dateOfBirth": fmt_date(birth, "birth"),
        "sex": sex_out,
        "expirationDate": fmt_date(expiry, "expiry"),
    }

    return parsed, checks


# ---------------------------
# Document warp + ROI (same as yours)
# ---------------------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype="float32"
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def try_warp_document(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    target = 900
    scale = (target / max(h, w)) if max(h, w) > target else 1.0
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    area_img = small.shape[0] * small.shape[1]

    for c in cnts[:10]:
        area = cv2.contourArea(c)
        if area < 0.15 * area_img:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            pts = pts / scale
            return four_point_warp(img_bgr, pts)

    return img_bgr

def deskew_by_min_area_rect(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = 255 - bw
    coords = cv2.findNonZero(bw)
    if coords is None:
        return roi_bgr

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = roi_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        roi_bgr, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def rotate_bgr(img, k):
    if k == 0:
        return img
    if k == 1:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def find_mrz_roi_single(doc_bgr: np.ndarray):
    h, w = doc_bgr.shape[:2]
    y0 = int(h * 0.40)
    search = doc_bgr[y0:h, 0:w]

    gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    rectK = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectK)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    gradX = np.absolute(gradX)
    gradX = (255 * (gradX / (gradX.max() + 1e-6))).astype("uint8")

    gradX = cv2.morphologyEx(
        gradX, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (55, 7)),
    )

    _, bw = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=1)
    bw = cv2.erode(bw, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    area_search = search.shape[0] * search.shape[1]
    best = None

    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.01 * area_search:
            continue
        aspect = cw / float(ch + 1e-6)
        if aspect < 3.0:
            continue
        if ch < 18:
            continue

        y_center = y + ch / 2
        y_bias = y_center / float(search.shape[0])

        score = area * (0.6 + 0.8 * y_bias)
        if best is None or score > best[0]:
            best = (score, (x, y, cw, ch))

    if best is None:
        return None

    _, (x, y, cw, ch) = best

    padX = int(0.03 * w)
    padY = int(0.35 * ch)

    x1 = max(0, x - padX)
    y1 = max(0, y - padY)
    x2 = min(search.shape[1], x + cw + padX)
    y2 = min(search.shape[0], y + ch + padY)

    roi = search[y1:y2, x1:x2]
    return deskew_by_min_area_rect(roi)

def find_mrz_roi(doc_bgr: np.ndarray):
    for k in (0, 1, 2, 3):
        rotated = rotate_bgr(doc_bgr, k)
        roi = find_mrz_roi_single(rotated)
        if roi is not None:
            return roi
    return None


# ---------------------------
# OCR helpers (FAST)
# ---------------------------

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<«»‹›〈〉＜＞⟨⟩〈〉﹤﹥"

def ocr_lines(img_bgr: np.ndarray):
    # ✅ enforce resize (big speedup)
    img_bgr = resize_for_ocr(img_bgr, 1200)
    return reader.readtext(
        img_bgr,
        detail=1,
        paragraph=False,
        allowlist=ALLOWLIST
    )

def ocr_mrz_best(mrz_roi_bgr: np.ndarray):
    if mrz_roi_bgr is None:
        return None

    # ✅ enforce resize early
    mrz_roi_bgr = resize_for_ocr(mrz_roi_bgr, 1400)

    gray = cv2.cvtColor(mrz_roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # keep only 3 variants (enough + faster)
    variants = []
    variants.append(("base", mrz_roi_bgr))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g2 = clahe.apply(gray)
    variants.append(("clahe", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)))

    thr = cv2.adaptiveThreshold(
        g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    inv = 255 - thr
    variants.append(("inv_thr", cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)))

    best = None

    for name, img in variants:
        results = ocr_lines(img)
        if not results:
            continue

        items = []
        for bbox, text, conf in results:
            t = normalize_mrz_line(text)
            if len(t) >= 18:
                y = float(bbox[0][1])
                items.append((y, t, float(conf)))

        if len(items) < 2:
            continue

        items.sort(key=lambda x: x[0])
        tail = items[-min(10, len(items)):]  # fewer pairs = faster

        for i in range(len(tail) - 1):
            for j in range(i + 1, len(tail)):
                l1 = repair_td3_line1(tail[i][1])
                l2 = normalize_mrz_line(tail[j][1]).ljust(44, "<")[:44]
                conf_pair = (tail[i][2] + tail[j][2]) / 2.0

                if not looks_like_td3_line1(l1):
                    continue
                if not looks_like_td3_line2(l2):
                    continue

                try:
                    parsed, checks = parse_td3(l1, l2)
                except Exception:
                    continue

                if not (checks["doc"] and checks["birth"] and checks["expiry"]):
                    continue

                passed = sum(1 for v in checks.values() if v)
                score = conf_pair * 0.6 + passed * 1.2

                cand = {
                    "variant": name,
                    "lines": [l1, l2],
                    "checks": checks,
                    "passed": passed,
                    "avg_conf": conf_pair,
                    "fields": parsed,
                    "score": score,
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand

    return best


def bottom_strip_ocr_fast(doc_bgr: np.ndarray):
    """
    Faster bottom-strip finder:
    - only 2 strips
    - only 2 scales
    - only 2 preprocess variants
    """
    h, w = doc_bgr.shape[:2]

    strips = [
        doc_bgr[int(h * 0.62):h, 0:w],
        doc_bgr[int(h * 0.70):h, 0:w],
    ]
    scales = [1.0, 1.5]

    for strip in strips:
        for sc in scales:
            s = strip
            if sc != 1.0:
                s = cv2.resize(
                    s, (int(s.shape[1] * sc), int(s.shape[0] * sc)),
                    interpolation=cv2.INTER_CUBIC
                )

            s = resize_for_ocr(s, 1400)

            gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            g2 = clahe.apply(gray)

            thr = cv2.adaptiveThreshold(
                g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )
            inv = 255 - thr

            for v in [
                cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR),
            ]:
                cand = ocr_mrz_best(v)
                if cand:
                    return cand, strip

    return None, None


def find_mrz_by_ocr_full(img_bgr: np.ndarray):
    # ⚠️ Keep but use resized doc to reduce cost
    img_bgr = resize_for_ocr(img_bgr, 1400)
    results = ocr_lines(img_bgr)
    if not results:
        return None

    candidates = []
    for bbox, text, conf in results:
        t = normalize_mrz_line(text)
        if len(t) >= 25 and t.count("<") >= 5:
            ys = [p[1] for p in bbox]
            y_center = sum(ys) / len(ys)
            candidates.append((y_center, bbox, t, float(conf)))

    if len(candidates) < 2:
        return None

    candidates.sort(key=lambda x: x[0])
    l1 = candidates[-2]
    l2 = candidates[-1]

    xs, ys = [], []
    for (_, bbox, _, _) in (l1, l2):
        for x, y in bbox:
            xs.append(int(x))
            ys.append(int(y))

    x1 = max(0, min(xs) - 35)
    y1 = max(0, min(ys) - 35)
    x2 = min(img_bgr.shape[1], max(xs) + 35)
    y2 = min(img_bgr.shape[0], max(ys) + 35)
    return img_bgr[y1:y2, x1:x2]


# ---------------------------
# Face (unchanged)
# ---------------------------

def extract_face(img_bgr: np.ndarray):
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

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
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


# ---------------------------
# Endpoint (FASTER ORDER)
# ---------------------------

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "Invalid image"}

    # keep a reasonable base size (not huge)
    h, w = img.shape[:2]
    if w < 1100:
        scale = 1100 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    doc = try_warp_document(img)

    mrz = None
    mrz_roi = None
    mrz_used = ""

    # 1) morphology ROI (fast)
    mrz_used = "morphology"
    mrz_roi = find_mrz_roi(doc)
    if mrz_roi is not None:
        mrz = ocr_mrz_best(mrz_roi)

    # 2) bottom-strip OCR (fast & strong)
    if not mrz:
        mrz_used = "bottom_strip_ocr"
        cand, roi_strip = bottom_strip_ocr_fast(doc)
        if cand:
            mrz = cand
            mrz_roi = roi_strip

    # 3) OPTIONAL: OCR full page (slow) -> keep last
    # If you want maximum speed, comment this block out.
    if not mrz:
        mrz_used = "ocr_full"
        mrz_roi2 = find_mrz_by_ocr_full(doc)
        if mrz_roi2 is not None:
            mrz2 = ocr_mrz_best(mrz_roi2)
            if mrz2:
                mrz = mrz2
                mrz_roi = mrz_roi2

    face = extract_face(doc)

    if not mrz:
        return {
            "ok": False,
            "error": "MRZ not found",
            "mrz_used": mrz_used,
            "debug": {
                "doc_b64": to_b64_jpeg(doc, 85),
                "mrz_roi_b64": to_b64_jpeg(mrz_roi, 90) if mrz_roi is not None else ""
            },
            "face": face
        }

    return {
        "ok": True,
        "mrz_used": mrz_used,
        "mrz": {
            "lines": mrz["lines"],
            "variant": mrz["variant"],
            "avg_conf": mrz["avg_conf"],
            "checks": mrz["checks"],
            "passed": mrz["passed"],
            "total": 5,
        },
        "fields": mrz["fields"],
        "debug": {
            "doc_b64": to_b64_jpeg(doc, 85),
            "mrz_roi_b64": to_b64_jpeg(mrz_roi, 90)
        },
        "face": face,
    }
