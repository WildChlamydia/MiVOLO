import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd

IMAGES_EXT: Tuple = (".jpeg", ".jpg", ".png", ".webp", ".bmp", ".gif")


@dataclass
class PictureInfo:
    image_path: str
    age: Optional[str]  # age or age range(start;end format) or "-1"
    gender: Optional[str]  # "M" of "F" or "-1"
    bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # face bbox: xyxy
    person_bbox: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])  # person bbox: xyxy

    @property
    def has_person_bbox(self) -> bool:
        return any(coord != -1 for coord in self.person_bbox)

    @property
    def has_face_bbox(self) -> bool:
        return any(coord != -1 for coord in self.bbox)

    def has_gt(self, only_age: bool = False) -> bool:
        if only_age:
            return self.age != "-1"
        else:
            return not (self.age == "-1" and self.gender == "-1")

    def clear_person_bbox(self):
        self.person_bbox = [-1, -1, -1, -1]

    def clear_face_bbox(self):
        self.bbox = [-1, -1, -1, -1]


class AnnotType(Enum):
    ORIGINAL = "original"
    PERSONS = "persons"

    @classmethod
    def _missing_(cls, value):
        print(f"WARN: Unknown annotation type {value}.")
        return AnnotType.NONE


def get_all_files(path: str, extensions: Tuple = IMAGES_EXT):
    files_all = []
    for root, subFolders, files in os.walk(path):
        for name in files:
            # linux tricks with .directory that still is file
            if "directory" not in name and sum([ext.lower() in name.lower() for ext in extensions]) > 0:
                files_all.append(os.path.join(root, name))
    return files_all


def read_csv_annotation_file(annotation_file: str, images_dir: str, ignore_without_gt=False):
    bboxes_per_image: Dict[str, List[PictureInfo]] = defaultdict(list)

    df = pd.read_csv(annotation_file, sep=",")

    annot_type = AnnotType("persons") if "person_x0" in df.columns else AnnotType("original")
    print(f"Reading {annotation_file} (type: {annot_type})...")

    missing_images = 0
    for index, row in df.iterrows():
        img_path = os.path.join(images_dir, row["img_name"])
        if not os.path.exists(img_path):
            missing_images += 1
            continue

        face_x1, face_y1, face_x2, face_y2 = row["face_x0"], row["face_y0"], row["face_x1"], row["face_y1"]
        age, gender = str(row["age"]), str(row["gender"])

        if ignore_without_gt and (age == "-1" or gender == "-1"):
            continue

        if annot_type == AnnotType.PERSONS:
            p_x1, p_y1, p_x2, p_y2 = row["person_x0"], row["person_y0"], row["person_x1"], row["person_y1"]
            person_bbox = list(map(int, [p_x1, p_y1, p_x2, p_y2]))
        else:
            person_bbox = [-1, -1, -1, -1]

        bbox = list(map(int, [face_x1, face_y1, face_x2, face_y2]))
        pic_info = PictureInfo(img_path, age, gender, bbox, person_bbox)
        assert isinstance(pic_info.person_bbox, list)

        bboxes_per_image[img_path].append(pic_info)

    if missing_images > 0:
        print(f"WARNING: Missing images: {missing_images}/{len(df)}")
    return bboxes_per_image, annot_type
