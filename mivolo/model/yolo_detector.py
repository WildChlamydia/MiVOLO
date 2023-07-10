import os
from typing import Dict, Union

import cv2
import numpy as np
import PIL
import torch
from mivolo.structures import PersonAndFaceResult
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.results import Results

# because of ultralytics bug it is important to unset CUBLAS_WORKSPACE_CONFIG after the module importing
os.unsetenv("CUBLAS_WORKSPACE_CONFIG")


class Detector:
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.4,
        iou_thresh: float = 0.7,
    ):
        self.yolo = YOLO(weights)
        self.yolo.fuse()

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: Dict[int, str] = self.yolo.model.names

        # init yolo.predictor
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose}
        # self.yolo.predict(**self.detector_kwargs)

    def predict(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        results: Results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)


if __name__ == "__main__":
    model = Detector("/data/dataset/iikrasnova/age_gender/yolov8x_person_face.pt")
    predicted = model.predict("/data/repositories/iikrasnova/agegender/jennifer_lawrence.jpg")
    out_im = predicted.plot()
    cv2.imwrite("/data/repositories/iikrasnova/agegender/jennifer_lawrence_out.jpg", out_im)
