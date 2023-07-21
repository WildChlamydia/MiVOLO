from collections import defaultdict
from typing import Dict, Generator, Optional, Tuple

import cv2
import numpy as np
import tqdm
from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector, PersonAndFaceResult


class Predictor:
    def __init__(self, config, verbose: bool = False):
        self.detector = Detector(config.detector_weights, config.device, verbose=verbose)
        self.age_gender_model = MiVOLO(
            config.checkpoint,
            config.device,
            half=True,
            use_persons=config.with_persons,
            disable_faces=config.disable_faces,
            verbose=verbose,
        )
        self.draw = config.draw

    def recognize(self, image: np.ndarray) -> Tuple[PersonAndFaceResult, Optional[np.ndarray]]:
        detected_objects: PersonAndFaceResult = self.detector.predict(image)
        self.age_gender_model.predict(image, detected_objects)

        out_im = None
        if self.draw:
            # plot results on image
            out_im = detected_objects.plot()

        return detected_objects, out_im

    def recognize_video(self, source: str) -> Generator:
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise ValueError(f"Failed to open video source {source}")

        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        detected_objects_history: Dict = defaultdict(list)
        for _ in tqdm.tqdm(range(total_frames)):
            ret, frame = video_capture.read()
            if not ret:
                break

            detected_objects: PersonAndFaceResult = self.detector.track(frame)
            self.age_gender_model.predict(frame, detected_objects)
            tr_persons, tr_faces = detected_objects.get_results_for_tracking()

            # add tr_persons and tr_faces to history
            for guid, data in tr_persons.items():
                detected_objects_history[guid].append(data)
            for guid, data in tr_faces.items():
                detected_objects_history[guid].append(data)

            frame = detected_objects.plot(
                tracked_objects=detected_objects_history,
            )
            yield frame
