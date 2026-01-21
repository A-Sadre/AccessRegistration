# main.py
# InsightFace (ArcFace) Enrollment API + Duplicate Face Protection
# + Silent-Face Anti-Spoofing (Liveness Gate)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import os
import threading
import json  # ✅ NEW (metadata persistence)

from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

from face_backend.antispoof.silentface_engine import SilentFaceLiveness

# -----------------------------
# Config
# -----------------------------
API_ORIGINS = ["*"]

# -----------------------------
# Liveness (Silent-Face)
# -----------------------------
ENABLE_LIVENESS = True

# Main pass threshold (binary decision)
LIVENESS_THRESH = 0.45

# Instant hard block threshold (super confident spoof region)
# Based on your observation (spoofs often 0.00x)
LIVENESS_HARD_BLOCK = 0.10

# models folder inside antispoof/resources/anti_spoof_models
LIVENESS_MODELS_DIR = os.path.join(
    os.path.dirname(__file__),
    "antispoof",
    "resources",
    "anti_spoof_models",
)
LIVENESS_DEVICE_ID = 0  # on CPU it's fine too

DB_DIR = "db"
REF_DIR = "ref_faces"
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

SAVE_REFERENCE_FACE = True
SAVE_VERIFY_UPLOADS = False

DET_MIN_SCORE = 0.65
MIN_FACE_PX = 120
MIN_BLUR = 40.0
MIN_BRIGHT = 55.0
MAX_BRIGHT = 210.0

DUP_FACE_THRESH = 0.6
VERIFY_THRESH = 0.55

MAX_UPLOADS_PER_ENROLL = 3
KEEP_TEMPLATES = 3

DET_SIZE = (480, 480)

PAD_BEFORE_ALIGN = 90
MAX_TEMPLATE_SIM = 0.92

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Face Enrollment API (InsightFace ArcFace + Liveness)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=API_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB: Dict[str, List[np.ndarray]] = {}

# -----------------------------
# ✅ Metadata persistence (name/email by person_id)
# -----------------------------
META_FILE = os.path.join(DB_DIR, "meta.json")
META: Dict[str, dict] = {}

def load_meta():
    global META
    if os.path.isfile(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as f:
                META = json.load(f)
        except Exception:
            META = {}
    else:
        META = {}

def save_meta():
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(META, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

load_meta()

# -----------------------------
# InsightFace init
# -----------------------------
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "recognition"],
)
face_app.prepare(ctx_id=-1, det_size=DET_SIZE)

infer_lock = threading.Lock()

# -----------------------------
# Liveness init
# -----------------------------
# You already discovered: for your model pack, REAL is label=1
LIVENESS_REAL_LABEL = 1
LIVENESS_REAL_LABEL = int(LIVENESS_REAL_LABEL)

liveness_engine = None
liveness_lock = threading.Lock()

if ENABLE_LIVENESS:
    liveness_engine = SilentFaceLiveness(
        models_dir=LIVENESS_MODELS_DIR,
        device_id=LIVENESS_DEVICE_ID,
        real_label=LIVENESS_REAL_LABEL,
    )

# -----------------------------
# Utils
# -----------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-9)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def image_blur(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def image_brightness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))

def read_upload_bgr(upload: UploadFile) -> np.ndarray:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image upload")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image (cannot decode)")
    return img

def save_reference_face(person_id: str, aligned_bgr: np.ndarray) -> str:
    path = os.path.join(REF_DIR, f"{person_id}.jpg")
    ok = cv2.imwrite(path, aligned_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to write reference face image")
    return path

def pad_reflect(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)

def _extract_real_prob_from_dbg(dbg: dict, real_label: int) -> Optional[float]:
    """
    SilentFace typically provides something like:
      dbg["avg_pred"] = [p0, p1, p2]
    We want the REAL probability = avg_pred[real_label]
    """
    if not isinstance(dbg, dict):
        return None
    avg_pred = dbg.get("avg_pred") or dbg.get("pred") or dbg.get("probs")
    if avg_pred is None:
        return None
    try:
        arr = np.array(avg_pred, dtype=np.float32).reshape(-1)
        if arr.size <= real_label:
            return None
        return float(arr[real_label])
    except Exception:
        return None

def run_liveness_or_403(img_bgr: np.ndarray) -> dict:
    if not ENABLE_LIVENESS:
        return {"enabled": False}

    if liveness_engine is None:
        raise HTTPException(status_code=500, detail={"reason": "liveness_engine_not_initialized"})

    with liveness_lock:
        is_live, score, label, bbox, dbg = liveness_engine.predict(img_bgr, live_thresh=LIVENESS_THRESH)

    predicted_label = int(label)

    is_live_decision = (predicted_label == LIVENESS_REAL_LABEL)

    if not is_live_decision:
        raise HTTPException(
            status_code=403,
            detail={
                "reason": "spoof_detected",
                "liveness": {
                    "live": False,
                    "label": predicted_label,
                    "score": round(float(score), 6),
                    "bbox_xywh": bbox,
                },
                "debug": dbg,
            },
        )

    return {
        "enabled": True,
        "live": True,
        "label": predicted_label,
        "score": round(float(score), 6),
        "bbox_xywh": bbox,
        "debug": dbg,
    }

    

# -----------------------------
# DB persistence
# -----------------------------
def db_file(person_id: str) -> str:
    return os.path.join(DB_DIR, f"{person_id}.npz")

def save_person(person_id: str):
    embs = DB.get(person_id, [])
    if not embs:
        return
    mat = np.stack([e.reshape(-1) for e in embs], axis=0).astype(np.float32)
    np.savez_compressed(db_file(person_id), embs=mat)

def load_db():
    DB.clear()
    if not os.path.isdir(DB_DIR):
        return
    for fn in os.listdir(DB_DIR):
        if not fn.endswith(".npz"):
            continue
        pid = fn[:-4]
        try:
            data = np.load(os.path.join(DB_DIR, fn))
            mat = data["embs"]
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            DB[pid] = [l2_normalize(mat[i]) for i in range(mat.shape[0])]
        except Exception:
            continue

load_db()

# -----------------------------
# Core pipeline
# -----------------------------
def extract_embedding_and_debug(img_bgr: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
    padded = pad_reflect(img_bgr, PAD_BEFORE_ALIGN)

    with infer_lock:
        faces = face_app.get(padded)

    if not faces:
        raise HTTPException(status_code=400, detail={"reason": "no_face_detected"})
    if len(faces) != 1:
        raise HTTPException(status_code=400, detail={"reason": "multiple_faces", "faces_found": len(faces)})

    f = faces[0]
    det_score = float(getattr(f, "det_score", 0.0))
    if det_score < DET_MIN_SCORE:
        raise HTTPException(status_code=400, detail={"reason": "low_detection_conf", "det_score": round(det_score, 4)})

    if getattr(f, "bbox", None) is None:
        raise HTTPException(status_code=400, detail={"reason": "no_bbox"})

    bbox = f.bbox.astype(int)
    x1, y1, x2, y2 = bbox.tolist()

    px = PAD_BEFORE_ALIGN
    ox1, oy1, ox2, oy2 = x1 - px, y1 - px, x2 - px, y2 - px

    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if min(bw, bh) < MIN_FACE_PX:
        raise HTTPException(status_code=400, detail={"reason": "face_too_small", "face_px": int(min(bw, bh))})

    kps = getattr(f, "kps", None)
    if kps is None or np.array(kps).shape != (5, 2):
        raise HTTPException(status_code=400, detail={"reason": "no_keypoints"})

    aligned = norm_crop(padded, kps, image_size=112)

    blur = image_blur(aligned)
    bright = image_brightness(aligned)

    if blur < MIN_BLUR:
        raise HTTPException(status_code=400, detail={"reason": "too_blurry", "blur": round(blur, 2)})
    if bright < MIN_BRIGHT:
        raise HTTPException(status_code=400, detail={"reason": "too_dark", "brightness": round(bright, 2)})
    if bright > MAX_BRIGHT:
        raise HTTPException(status_code=400, detail={"reason": "too_bright", "brightness": round(bright, 2)})

    emb = getattr(f, "normed_embedding", None)
    if emb is None:
        emb = getattr(f, "embedding", None)
    if emb is None:
        raise HTTPException(status_code=500, detail={"reason": "no_embedding"})

    emb = l2_normalize(np.array(emb, dtype=np.float32))

    dbg = {
        "faces_found": 1,
        "det_score": round(det_score, 4),
        "face_box": {"x1": int(ox1), "y1": int(oy1), "x2": int(ox2), "y2": int(oy2)},
        "face_px": {"w": int(bw), "h": int(bh)},
        "pad_before_align": int(PAD_BEFORE_ALIGN),
        "blur": round(blur, 2),
        "brightness": round(bright, 2),
        "emb_dim": int(emb.shape[0]),
        "emb_norm": round(float(np.linalg.norm(emb)), 6),
    }
    return emb, dbg, aligned

def find_best_match(probe: np.ndarray, exclude_person_id: Optional[str] = None) -> Tuple[Optional[str], float]:
    best_pid = None
    best_score = -1.0
    for pid, embs in DB.items():
        if exclude_person_id and pid == exclude_person_id:
            continue
        for ref in embs:
            s = cosine_sim(probe, ref)
            if s > best_score:
                best_score = s
                best_pid = pid
    return best_pid, float(best_score)

def pick_diverse_templates(extracted, k: int) -> List[tuple]:
    extracted = sorted(extracted, key=lambda x: x[0], reverse=True)
    chosen = []
    for item in extracted:
        if len(chosen) >= k:
            break
        _, emb, _, _ = item
        if not chosen:
            chosen.append(item)
            continue
        too_similar = any(cosine_sim(emb, c[1]) > MAX_TEMPLATE_SIM for c in chosen)
        if not too_similar:
            chosen.append(item)

    if len(chosen) < k:
        for item in extracted:
            if len(chosen) >= k:
                break
            if item not in chosen:
                chosen.append(item)

    return chosen[:k]

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "db_users": len(DB),
        "det_size": list(DET_SIZE),
        "thresholds": {
            "dup_face": DUP_FACE_THRESH,
            "verify": VERIFY_THRESH,
            "det_min_score": DET_MIN_SCORE,
            "min_face_px": MIN_FACE_PX,
            "min_blur": MIN_BLUR,
            "min_bright": MIN_BRIGHT,
            "max_bright": MAX_BRIGHT,
        },
        "liveness": {
            "enabled": ENABLE_LIVENESS,
            "threshold": float(LIVENESS_THRESH),
            "hard_block": float(LIVENESS_HARD_BLOCK),
            "models_dir": LIVENESS_MODELS_DIR,
            "device_id": LIVENESS_DEVICE_ID,
            "real_label": int(LIVENESS_REAL_LABEL),
        },
    }

@app.post("/debug/liveness")
async def debug_liveness(image: UploadFile = File(...)):
    img = read_upload_bgr(image)

    if not ENABLE_LIVENESS or liveness_engine is None:
        return {"ok": False, "reason": "liveness_disabled"}

    with liveness_lock:
        is_live, score, label, bbox, dbg = liveness_engine.predict(img, live_thresh=LIVENESS_THRESH)

    real_prob = _extract_real_prob_from_dbg(dbg, LIVENESS_REAL_LABEL)
    if real_prob is None:
        real_prob = float(score)

    return {
        "ok": True,
        "is_live_raw": bool(is_live),
        "label_raw": int(label) if isinstance(label, (int, np.integer)) else label,
        "score_raw": float(score),
        "real_label": int(LIVENESS_REAL_LABEL),
        "real_prob": float(real_prob),  # ✅ what we use in decisions
        "bbox_xywh": bbox,
        "debug": dbg,
    }

@app.get("/debug/db")
def debug_db():
    return {"ok": True, "persons": {pid: len(v) for pid, v in DB.items()}}

@app.post("/enroll")
async def enroll(
    person_id: str = Form(...),
    first_name: str = Form(""),
    last_name: str = Form(""),
    email: str = Form(""),
    images: List[UploadFile] = File(...),
):
    person_id = person_id.strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="person_id is required")

    if person_id in DB and len(DB[person_id]) > 0:
        raise HTTPException(status_code=409, detail={"reason": "person_id_exists", "person_id": person_id})

    if not images:
        raise HTTPException(status_code=400, detail={"reason": "no_images"})

    extracted = []

    for up in images[:MAX_UPLOADS_PER_ENROLL]:
        img = read_upload_bgr(up)

        # ✅ Liveness gate first (binary)
        liveness_dbg = run_liveness_or_403(img)

        # Then embedding pipeline
        emb, dbg, aligned = extract_embedding_and_debug(img)

        q = float(dbg.get("blur", 0.0)) + 80.0 * float(dbg.get("det_score", 0.0))
        dbg["liveness"] = liveness_dbg

        extracted.append((q, emb, dbg, aligned))

    if not extracted:
        raise HTTPException(status_code=400, detail={"reason": "no_valid_templates"})

    best = pick_diverse_templates(extracted, KEEP_TEMPLATES)

    # Duplicate face protection (max similarity across selected templates)
    if len(DB) > 0:
        best_owner = None
        best_score = -1.0
        for (_, emb, _, _) in best:
            owner, score = find_best_match(emb)
            if score > best_score:
                best_score = score
                best_owner = owner

        if best_owner is not None and best_score >= DUP_FACE_THRESH:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "face_already_assigned",
                    "assigned_to": best_owner,
                    "score": round(best_score, 6),
                    "threshold": DUP_FACE_THRESH,
                },
            )

    DB.setdefault(person_id, [])
    for (_, emb, _, _) in best:
        DB[person_id].append(emb)
    save_person(person_id)

    # ✅ Save metadata (name/email) keyed by person_id
    first_name = (first_name or "").strip()
    last_name = (last_name or "").strip()
    email = (email or "").strip()
    META[person_id] = {"first_name": first_name, "last_name": last_name, "email": email}
    save_meta()

    ref_path = None
    if SAVE_REFERENCE_FACE:
        best_aligned = best[0][3]
        ref_path = save_reference_face(person_id, best_aligned)

    return {
        "ok": True,
        "person_id": person_id,
        "templates": len(DB[person_id]),
        "duplicate_check": {"enabled": True, "threshold": DUP_FACE_THRESH},
        "liveness": {
            "enabled": ENABLE_LIVENESS,
            "threshold": float(LIVENESS_THRESH),
            "hard_block": float(LIVENESS_HARD_BLOCK),
            "real_label": int(LIVENESS_REAL_LABEL),
        },
        "reference_face_saved": bool(ref_path),
        "reference_face_path": ref_path,
        "debug": {
            "templates_kept": len(best),
            "templates": [b[2] for b in best],
        },
    }

@app.post("/verify")
async def verify(image: UploadFile = File(...)):
    if len(DB) == 0:
        raise HTTPException(status_code=404, detail={"reason": "no_enrolled_users"})

    img = read_upload_bgr(image)

    # ✅ Liveness gate (binary)
    liveness_dbg = run_liveness_or_403(img)

    emb, dbg, _aligned = extract_embedding_and_debug(img)
    dbg["liveness"] = liveness_dbg

    owner, score = find_best_match(emb)
    matched = (owner is not None) and (score >= VERIFY_THRESH)

    # ✅ Provide friendly name (if available)
    display_name = None
    if owner and owner in META:
        fn = (META[owner].get("first_name") or "").strip()
        ln = (META[owner].get("last_name") or "").strip()
        display_name = (fn + " " + ln).strip() or None

    return {
        "ok": True,
        "match": matched,
        "best_person_id": owner,
        "best_name": display_name,  # ✅ NEW (Android can show this instead of passport)
        "score": round(score, 6),
        "threshold": VERIFY_THRESH,
        "debug": dbg,
    }

@app.post("/reset")
def reset():
    DB.clear()

    if os.path.isdir(DB_DIR):
        for fn in os.listdir(DB_DIR):
            if fn.endswith(".npz"):
                try:
                    os.remove(os.path.join(DB_DIR, fn))
                except Exception:
                    pass

    if os.path.isdir(REF_DIR):
        for fn in os.listdir(REF_DIR):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    os.remove(os.path.join(REF_DIR, fn))
                except Exception:
                    pass

    # ✅ Clear metadata too
    META.clear()
    save_meta()

    return {"ok": True, "message": "DB cleared (and reference faces cleared)"}
