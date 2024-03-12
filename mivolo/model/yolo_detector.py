import os
from typing import Dict, Union

import numpy as np
import PIL
import torch
from mivolo.structures import PersonAndFaceResult
from ultralytics import YOLO
from ultralytics.engine.results import Results

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

    def track(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        results: Results = self.yolo.track(image, persist=True, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)
