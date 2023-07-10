import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import pandas as pd
import tqdm
from mivolo.data.data_reader import PictureInfo, get_all_files
from mivolo.model.yolo_detector import Detector, PersonAndFaceResult
from preparation_utils import get_additional_bboxes, get_main_face, save_annotations


def read_adience_annotations(annotations_files):
    annotations_per_image = {}
    stat_per_fold = defaultdict(int)
    cols = ["user_id", "original_image", "face_id", "age", "gender"]
    for file in annotations_files:
        fold_name = os.path.basename(file).split(".")[0]
        df = pd.read_csv(file, sep="\t", usecols=cols)
        for index, row in df.iterrows():
            face_id, img_name, user_id = row["face_id"], row["original_image"], row["user_id"]
            aligned_face_path = f"faces/{user_id}/coarse_tilt_aligned_face.{face_id}.{img_name}"

            age, gender = row["age"], row["gender"]
            gender = gender.upper() if isinstance(gender, str) and gender != "u" else None
            age = age if isinstance(age, str) else None

            annotations_per_image[aligned_face_path] = {"age": age, "gender": gender, "fold": fold_name}
            stat_per_fold[fold_name] += 1

    print(f"Per fold images: {stat_per_fold}")
    return annotations_per_image


def read_data(images_dir, annotations_files, data_dir) -> List[PictureInfo]:
    dataset_pictures: List[PictureInfo] = []

    all_images = get_all_files(images_dir)
    annotations_per_file = read_adience_annotations(annotations_files)

    total, missed = 0, 0
    stat_per_gender: Dict[str, int] = defaultdict(int)
    missed_gender, missed_age, missed_gender_and_age = 0, 0, 0
    stat_per_ages: Dict[str, int] = defaultdict(int)

    # final age classes: '0;2', "4;6", "8;12", "15;20", "25;32", "38;43", "48;53", "60;100"

    age_map = {
        "2": "(0, 2)",
        "3": "(0, 2)",
        "13": "(8, 12)",
        "(8, 23)": "(8, 12)",
        "22": "(15, 20)",
        "23": "(25, 32)",
        "29": "(25, 32)",
        "(27, 32)": "(25, 32)",
        "32": "(25, 32)",
        "34": "(25, 32)",
        "35": "(25, 32)",
        "36": "(38, 43)",
        "(38, 42)": "(38, 43)",
        "(38, 48)": "(38, 43)",
        "42": "(38, 43)",
        "45": "(38, 43)",
        "46": "(48, 53)",
        "55": "(48, 53)",
        "56": "(48, 53)",
        "57": "(60, 100)",
        "58": "(60, 100)",
    }
    for image_path in all_images:
        total += 1
        relative_path = image_path.replace(f"{data_dir}/", "")
        if relative_path not in annotations_per_file:
            missed += 1
            print("Can not find annotation for ", relative_path)
        else:
            annot = annotations_per_file[relative_path]
            age, gender = annot["age"], annot["gender"]

            if gender is None and age is not None:
                missed_gender += 1
            elif age is None and gender is not None:
                missed_age += 1
            elif gender is None and age is None:
                missed_gender_and_age += 1
                # skip such image
                continue

            if gender is not None:
                stat_per_gender[gender] += 1

            if age is not None:
                age = age_map[age] if age in age_map else age
                stat_per_ages[age] += 1

            dataset_pictures.append(PictureInfo(image_path, age, gender))

    print(f"Missed annots for images: {missed}/{total}")
    print(f"Missed genders: {missed_gender}")
    print(f"Missed ages: {missed_age}")
    print(f"Missed ages and gender: {missed_gender_and_age}")
    print(f"\nPer gender images: {stat_per_gender}")
    ages = list(stat_per_ages.keys())
    print(f"Per ages categories ({len(ages)} cats) :")
    ages = sorted(ages, key=lambda x: int(x.split("(")[-1].split(",")[0].strip()))
    for age in ages:
        print(f"Age: {age} Count: {stat_per_ages[age]}")

    return dataset_pictures


def main(faces_dir: str, annotations: List[str], data_dir: str, detector_cfg: dict = None):
    """
    Generate a .txt annotation file with columns:
        ["img_name", "age", "gender",
        "face_x0", "face_y0", "face_x1", "face_y1",
        "person_x0", "person_y0", "person_x1", "person_y1"]

    All person bboxes here will be set to [-1, -1, -1, -1]

    If detector_cfg is set, for each face bbox will be refined using detector.
        Also, other detected faces wil be written to txt file (needed for further preprocessing)
    """
    # out directory for annotations
    out_dir = os.path.join(data_dir, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    # load annotations
    images: List[PictureInfo] = read_data(faces_dir, annotations, data_dir)

    if detector_cfg:
        # detect faces with yolo detector
        faces_not_found, images_with_other_faces = 0, 0
        other_faces: List[PictureInfo] = []

        detector_weights, device = detector_cfg["weights"], detector_cfg["device"]
        detector = Detector(detector_weights, device, verbose=False, conf_thresh=0.1, iou_thresh=0.2)
        for image_info in tqdm.tqdm(images, desc="Detecting faces: "):
            cv_im = cv2.imread(image_info.image_path)
            im_h, im_w = cv_im.shape[:2]

            detected_objects: PersonAndFaceResult = detector.predict(cv_im)
            main_bbox, other_bboxes_inds = get_main_face(detected_objects)

            if main_bbox is None:
                # use a full image as face bbox
                faces_not_found += 1
                image_info.bbox = [0, 0, im_w, im_h]
            else:
                image_info.bbox = main_bbox

            if len(other_bboxes_inds):
                images_with_other_faces += 1

            additional_faces = get_additional_bboxes(detected_objects, other_bboxes_inds, image_info.image_path)
            other_faces.extend(additional_faces)

        print(f"Faces not detected: {faces_not_found}/{len(images)}")
        print(f"Images with other faces: {images_with_other_faces}/{len(images)}")
        print(f"Other faces: {len(other_faces)}")

        images = images + other_faces

    else:
        # use a full image as face bbox
        for image_info in tqdm.tqdm(images, desc="Collect face bboxes: "):
            cv_im = cv2.imread(image_info.image_path)
            im_h, im_w = cv_im.shape[:2]
            image_info.bbox = [0, 0, im_w, im_h]  # xyxy

    save_annotations(images, faces_dir, out_file=os.path.join(out_dir, "adience_annotations.csv"))


def get_parser():
    parser = argparse.ArgumentParser(description="Adience")
    parser.add_argument(
        "--dataset_path",
        default="data/adience",
        type=str,
        required=True,
        help="path to dataset with faces/ and fold_{i}_data.txt files",
    )
    parser.add_argument(
        "--detector_weights", default=None, type=str, required=False, help="path to face and person detector"
    )
    parser.add_argument("--device", default="cuda:0", type=str, required=False, help="device to inference detector")

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    data_dir = args.dataset_path
    faces_dir = os.path.join(data_dir, "faces")

    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]

    annotations = [
        os.path.join(data_dir, "fold_0_data.txt"),
        os.path.join(data_dir, "fold_1_data.txt"),
        os.path.join(data_dir, "fold_2_data.txt"),
        os.path.join(data_dir, "fold_3_data.txt"),
        os.path.join(data_dir, "fold_4_data.txt"),
    ]

    detector_cfg: Optional[Dict[str, str]] = None
    if args.detector_weights is not None:
        detector_cfg = {"weights": args.detector_weights, "device": "cuda:0"}

    main(faces_dir, annotations, data_dir, detector_cfg)
