# antispoof/src/anti_spoof_predict.py
import os
import cv2
import math
import numpy as np
import torch

from face_backend.antispoof.src.model_lib.MiniFASNet import (
    MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV1, MiniFASNetV2SE
)
from face_backend.antispoof.src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class Detection:
    def __init__(self):
        # ✅ FIX: build absolute paths from this file location
        here = os.path.dirname(os.path.abspath(__file__))          # .../antispoof/src
        antispoof_root = os.path.dirname(here)                     # .../antispoof
        resources_dir = os.path.join(antispoof_root, "resources")  # .../antispoof/resources

        deploy = os.path.join(resources_dir, "detection_model", "deploy.prototxt")
        caffemodel = os.path.join(resources_dir, "detection_model", "Widerface-RetinaFace.caffemodel")

        if not os.path.isfile(deploy):
            raise FileNotFoundError(f"deploy.prototxt not found: {deploy}")
        if not os.path.isfile(caffemodel):
            raise FileNotFoundError(f"caffemodel not found: {caffemodel}")

        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)

class AntiSpoofPredict(Detection):
    def __init__(self, device_id):
        super(AntiSpoofPredict, self).__init__()
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    def get_bbox(self, img):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(
                img,
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR
            )

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = (
            out[max_conf_index, 3] * width,
            out[max_conf_index, 4] * height,
            out[max_conf_index, 5] * width,
            out[max_conf_index, 6] * height
        )
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox

    def predict(self, img, model_path):
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name(model_name)
        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = next(keys)
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

        self.model.eval()

        img = self.trans(img, w_input, h_input)
        with torch.no_grad():
            img = img.to(self.device)
            result = self.model.forward(img)
            result = torch.softmax(result, dim=1).cpu().numpy()
        return result

    def trans(self, img, w_input, h_input):
        img = cv2.resize(img, (w_input, h_input))
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return img
