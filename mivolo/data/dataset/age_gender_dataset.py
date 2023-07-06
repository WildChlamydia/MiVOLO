import logging
from typing import Any, List, Optional, Set

import cv2
import numpy as np
import torch
from mivolo.data.dataset.reader_age_gender import ReaderAgeGender
from PIL import Image
from torchvision import transforms

_logger = logging.getLogger("AgeGenderDataset")


class AgeGenderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images_path,
        annotations_path,
        name=None,
        split="train",
        load_bytes=False,
        img_mode="RGB",
        transform=None,
        is_training=False,
        seed=1234,
        target_size=224,
        min_age=None,
        max_age=None,
        model_with_persons=False,
        use_persons=False,
        disable_faces=False,
        only_age=False,
    ):
        reader = ReaderAgeGender(
            images_path,
            annotations_path,
            split=split,
            seed=seed,
            target_size=target_size,
            with_persons=use_persons,
            disable_faces=disable_faces,
            only_age=only_age,
        )

        self.name = name
        self.model_with_persons = model_with_persons
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self._consecutive_errors = 0
        self.is_training = is_training
        self.random_flip = 0.0

        # Setting up classes.
        # If min and max classes are passed - use them to have the same preprocessing for validation
        self.max_age: float = None
        self.min_age: float = None
        self.avg_age: float = None
        self.set_ages_min_max(min_age, max_age)

        self.genders = ["M", "F"]
        self.num_classes_gender = len(self.genders)

        self.age_classes: Optional[List[str]] = self.set_age_classes()

        self.num_classes_age = 1 if self.age_classes is None else len(self.age_classes)
        self.num_classes: int = self.num_classes_age + self.num_classes_gender
        self.target_dtype = torch.float32

    def set_age_classes(self) -> Optional[List[str]]:
        return None  # for regression dataset

    def set_ages_min_max(self, min_age: Optional[float], max_age: Optional[float]):

        assert all(age is None for age in [min_age, max_age]) or all(
            age is not None for age in [min_age, max_age]
        ), "Both min and max age must be passed or none of them"

        if max_age is not None and min_age is not None:
            _logger.info(f"Received predefined min_age {min_age} and max_age {max_age}")
            self.max_age = max_age
            self.min_age = min_age
        else:
            # collect statistics from loaded dataset
            all_ages_set: Set[int] = set()
            for img_path, image_samples in self.reader._ann.items():
                for image_sample_info in image_samples:
                    if image_sample_info.age == "-1":
                        continue
                    age = round(float(image_sample_info.age))
                    all_ages_set.add(age)

            self.max_age = max(all_ages_set)
            self.min_age = min(all_ages_set)

        self.avg_age = (self.max_age + self.min_age) / 2.0

    def _norm_age(self, age):
        return (age - self.avg_age) / (self.max_age - self.min_age)

    def parse_gender(self, _gender: str) -> float:
        if _gender != "-1":
            gender = float(0 if _gender == "M" or _gender == "0" else 1)
        else:
            gender = -1
        return gender

    def parse_target(self, _age: str, gender: str) -> List[Any]:
        if _age != "-1":
            age = round(float(_age))
            age = self._norm_age(float(age))
        else:
            age = -1

        target: List[float] = [age, self.parse_gender(gender)]
        return target

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        # Disable pretrained monkey-patched transforms
        if not transform:
            return

        _trans = []
        for trans in transform.transforms:
            if "Resize" in str(trans):
                continue
            if "Crop" in str(trans):
                continue
            _trans.append(trans)
        self._transform = transforms.Compose(_trans)

    def apply_tranforms(self, image: Optional[np.ndarray]) -> np.ndarray:
        if image is None:
            return None

        if self.transform is None:
            return image

        image = convert_to_pil(image, self.img_mode)
        for trans in self.transform.transforms:
            image = trans(image)
        return image

    def __getitem__(self, index):
        # get preprocessed face and person crops (np.ndarray)
        # resize + pad, for person crops: cut off other bboxes
        images, target = self.reader[index]

        target = self.parse_target(*target)

        if self.model_with_persons:
            face_image, person_image = images
            person_image: np.ndarray = self.apply_tranforms(person_image)
        else:
            face_image = images[0]
            person_image = None

        face_image: np.ndarray = self.apply_tranforms(face_image)

        if person_image is not None:
            img = np.concatenate([face_image, person_image], axis=0)
        else:
            img = face_image

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


def convert_to_pil(cv_im: Optional[np.ndarray], img_mode: str = "RGB") -> "Image":
    if cv_im is None:
        return None

    if img_mode == "RGB":
        cv_im = cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("Incorrect image mode has been passed!")

    cv_im = np.ascontiguousarray(cv_im)
    pil_image = Image.fromarray(cv_im)
    return pil_image
