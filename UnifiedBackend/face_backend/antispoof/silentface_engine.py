# antispoof/silentface_engine.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from face_backend.antispoof.src.anti_spoof_predict import AntiSpoofPredict
from face_backend.antispoof.src.generate_patches import CropImage
from face_backend.antispoof.src.utility import parse_model_name  # ✅ IMPORTANT


class SilentFaceLiveness:
    """
    Silent-Face Anti-Spoofing wrapper.

    Returns:
      (is_live, score, label, bbox_xywh, debug)

    - label = argmax(avg_pred)
    - score = avg_pred[label]
    - is_live = (label == real_label) and (score >= live_thresh)
    """

    def __init__(self, models_dir: str, device_id: int = 0, real_label: int = 1):
        self.models_dir = os.path.abspath(models_dir)
        self.device_id = int(device_id)

        if real_label is None:
            real_label = 1
        self.real_label = int(real_label)

        if not os.path.isdir(self.models_dir):
            raise RuntimeError(f"SilentFace models_dir not found: {self.models_dir}")

        self.model_test = AntiSpoofPredict(self.device_id)
        self.image_cropper = CropImage()

        self.model_files = self._list_model_files(self.models_dir)
        if not self.model_files:
            raise RuntimeError(f"No model files found in: {self.models_dir}")

    def _list_model_files(self, models_dir: str) -> List[str]:
        out = []
        for fn in os.listdir(models_dir):
            p = os.path.join(models_dir, fn)
            if os.path.isfile(p) and fn.lower().endswith((".pth", ".pt")):
                out.append(p)
        out.sort()
        return out

    def predict(self, bgr: np.ndarray, live_thresh: float = 0.5):
        if bgr is None or not isinstance(bgr, np.ndarray):
            raise ValueError("predict expects a BGR image numpy array")

        # --- detect bbox (xywh) ---
        bbox = self.model_test.get_bbox(bgr)  # [x,y,w,h]
        if bbox is None:
            return False, 0.0, -1, None, {"reason": "no_face_bbox"}

        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            return False, 0.0, -1, [x, y, w, h], {"reason": "invalid_bbox", "bbox_xywh": [x, y, w, h]}

        # --- sum predictions across models ---
        sum_pred: Optional[np.ndarray] = None
        per_model_preds: List[List[float]] = []
        used_files: List[str] = []

        for model_path in self.model_files:
            model_name = os.path.basename(model_path)

            # ✅ FIX: use function parse_model_name (not a method)
            h_input, w_input, model_type, scale = parse_model_name(model_name)

            param = {
                "org_img": bgr,
                "bbox": [x, y, w, h],
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            crop_img = self.image_cropper.crop(**param)
            pred = self.model_test.predict(crop_img, model_path)  # usually (1,3)

            pred = np.asarray(pred, dtype=np.float32).reshape(-1)

            if sum_pred is None:
                sum_pred = pred.copy()
            else:
                if pred.shape[0] != sum_pred.shape[0]:
                    # ignore incompatible model file
                    continue
                sum_pred += pred

            per_model_preds.append(pred.tolist())
            used_files.append(model_name)

        if sum_pred is None or len(per_model_preds) == 0:
            return False, 0.0, -1, [x, y, w, h], {"reason": "no_model_predictions"}

        # average
        used = len(per_model_preds)
        avg_pred = (sum_pred / float(used)).astype(np.float32).reshape(-1)

        real_score = float(avg_pred[self.real_label])
        label = int(np.argmax(avg_pred))
        score = real_score  # score means "real probability"
        is_live = (real_score >= float(live_thresh))

        dbg: Dict[str, Any] = {
            "models_dir": self.models_dir,
            "models_used": used,
            "model_files_used": used_files,
            "per_model_preds": per_model_preds,
            "avg_pred": avg_pred.tolist(),
            "real_label": int(self.real_label),
            "live_thresh": float(live_thresh),
        }

        return is_live, score, label, [x, y, w, h], dbg
