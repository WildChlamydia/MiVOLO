import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import tqdm
from mivolo.data.data_reader import PictureInfo, get_all_files
from mivolo.modeling.yolo_detector import Detector, PersonAndFaceResult
from preparation_utils import get_additional_bboxes, get_main_face, save_annotations
from prepare_fairface import find_persons_on_image


def get_im_name(img_path):
    im_name = img_path.split("/")[-1]
    im_name = im_name.replace("é", "e").replace("é", "e")
    im_name = im_name.replace("ó", "o").replace("ó", "o")
    im_name = im_name.replace("å", "a").replace("å", "a")
    im_name = im_name.replace("ñ", "n").replace("ñ", "n")
    im_name = im_name.replace("ö", "o").replace("ö", "o")
    im_name = im_name.replace("ä", "a").replace("ä", "a")
    im_name = im_name.replace("ü", "u").replace("ü", "u")
    im_name = im_name.replace("á", "a").replace("á", "a")
    im_name = im_name.replace("ë", "e").replace("ë", "e")
    im_name = im_name.replace("í", "i").replace("í", "i")

    return im_name


def read_json_annotations(annotations: List[str], splits: List[str]) -> Dict[str, dict]:
    print("Parsing annotations")
    annotations_per_image = {}
    stat_per_split: Dict[str, int] = defaultdict(int)

    missed = 0
    for item_id, face in tqdm.tqdm(enumerate(annotations), total=len(annotations)):
        im_name = get_im_name(face["img_path"])
        split = splits[int(face["folder"])]

        stat_per_split[split] += 1

        gender = face["gender"] if "gender" in face else None
        if "alignment_source" in face and face["alignment_source"] == "file not found":
            missed += 1

        annotations_per_image[im_name] = {"age": str(face["age"]), "gender": gender, "split": split}

    print("missed annots: ", missed)

    print(f"Per split images: {stat_per_split}")
    print(f"Found {len(annotations_per_image)} annotations")
    return annotations_per_image


def read_data(images_dir, annotations, splits) -> Dict[str, List[PictureInfo]]:
    dataset: Dict[str, List[PictureInfo]] = defaultdict(list)
    all_images = get_all_files(images_dir)
    print(f"Found {len(all_images)} images")

    annotations_per_file: Dict[str, dict] = read_json_annotations(annotations, splits)

    total, missed = 0, 0
    missed_gender_and_age = 0
    stat_per_ages: Dict[str, int] = defaultdict(int)
    stat_per_gender: Dict[str, int] = defaultdict(int)

    for image_path in all_images:
        total += 1
        image_name = get_im_name(image_path)

        if image_name not in annotations_per_file:
            missed += 1
            print(f"Can not find annotation for {image_name}")
        else:
            annot = annotations_per_file[image_name]
            age, gender, split = annot["age"], annot["gender"], annot["split"]

            if gender is None and age is None:
                missed_gender_and_age += 1
                # skip such image
                continue

            if age is not None:
                stat_per_ages[age] += 1
            if gender is not None:
                stat_per_gender[gender] += 1

            info = PictureInfo(image_path, age, gender)
            dataset[split].append(info)

    print(f"Missed annots for images: {missed}/{total}")
    print(f"Missed ages and gender: {missed_gender_and_age}")
    ages = list(stat_per_ages.keys())
    print(f"Per gender stat: {stat_per_gender}")
    print(f"Per ages categories ({len(ages)} cats) :")
    ages = sorted(ages, key=lambda x: int(x.split("(")[-1].split(",")[0].strip()))
    for age in ages:
        print(f"Age: {age} Count: {stat_per_ages[age]}")

    return dataset


def collect_faces(
    faces_dir: str,
    annotations: List[dict],
    data_dir: str,
    detector_cfg: dict = None,
    padding: float = 0.1,
    splits: List[str] = [],
    db_name: str = "",
    use_coarse_persons: bool = False,
    find_persons: bool = False,
    person_padding: float = 0.0,
    use_coarse_faces: bool = False,
):
    """
    Generate train, val, test .txt annotation files with columns:
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
    images_per_split: Dict[str, List[PictureInfo]] = read_data(faces_dir, annotations, splits)

    for split_ind, (split, images) in enumerate(images_per_split.items()):
        print(f"Processing {split} split ({split_ind}/{len(images_per_split)})...")
        if detector_cfg:
            # detect faces with yolo detector
            faces_not_found, images_with_other_faces = 0, 0
            other_faces: List[PictureInfo] = []

            detector_weights, device = detector_cfg["weights"], detector_cfg["device"]
            detector = Detector(detector_weights, device, verbose=False, conf_thresh=0.1, iou_thresh=0.2)
            for image_info in tqdm.tqdm(images, desc="Detecting faces: "):
                cv_im = cv2.imread(image_info.image_path)
                im_h, im_w = cv_im.shape[:2]

                pad_x, pad_y = int(padding * im_w), int(padding * im_h)
                coarse_face_bbox = [pad_x, pad_y, im_w - pad_x, im_h - pad_y]  # xyxy

                detected_objects: PersonAndFaceResult = detector.predict(cv_im)
                main_bbox, other_faces_inds = get_main_face(detected_objects, coarse_face_bbox)

                if len(other_faces_inds):
                    images_with_other_faces += 1

                if main_bbox is None:
                    # use a full image as a face bbox
                    faces_not_found += 1
                    main_bbox = coarse_face_bbox
                elif use_coarse_faces:
                    main_bbox = coarse_face_bbox
                image_info.bbox = main_bbox

                if find_persons:
                    additional_faces, additional_persons = find_persons_on_image(
                        image_info, main_bbox, detected_objects, other_faces_inds, device
                    )
                    # add all additional faces
                    other_faces.extend(additional_faces)
                    # add persons with empty faces
                    other_faces.extend(additional_persons)
                else:
                    additional_faces = get_additional_bboxes(detected_objects, other_faces_inds, image_info.image_path)
                    other_faces.extend(additional_faces)
                    # full image as a person bbox
                    coarse_person_bbox = [0, 0, im_w, im_h]  # xyxy
                    if find_persons:
                        image_info.person_bbox = coarse_person_bbox

            print(f"Faces not detected: {faces_not_found}/{len(images)}")
            print(f"Images with other faces: {images_with_other_faces}/{len(images)}")
            print(f"Other faces: {len(other_faces)}")

            images = images + other_faces

        else:
            for image_info in tqdm.tqdm(images, desc="Collect face bboxes: "):

                cv_im = cv2.imread(image_info.image_path)
                im_h, im_w = cv_im.shape[:2]

                # use a full image as a face bbox
                pad_x, pad_y = int(padding * im_w), int(padding * im_h)
                image_info.bbox = [pad_x, pad_y, im_w - pad_x, im_h - pad_y]  # xyxy

                if use_coarse_persons or find_persons:
                    # full image as a person bbox
                    pad_x_p, pad_y_p = int(person_padding * im_w), int(person_padding * im_h)
                    image_info.person_bbox = [pad_x_p, pad_y_p, im_w - pad_x_p, im_h]  # xyxy

        save_annotations(images, faces_dir, out_file=os.path.join(out_dir, f"{db_name}_{split}_annotations.csv"))


def get_parser():
    parser = argparse.ArgumentParser(description="CACD")
    parser.add_argument(
        "--dataset_path",
        default="data/CACD",
        type=str,
        required=True,
        help="path to dataset with CACD200 folder",
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
    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]

    faces_dir = os.path.join(data_dir, "CACD2000")

    # https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases/tree/main
    json_path = os.path.join(data_dir, "CACD2000.json")
    with open(json_path, "r") as stream:
        annotations = json.load(stream)

    detector_cfg: Optional[Dict[str, str]] = None
    if args.detector_weights is not None:
        detector_cfg = {"weights": args.detector_weights, "device": "cuda:0"}

    splits = ["train", "valid", "test"]
    collect_faces(
        faces_dir,
        annotations,
        data_dir,
        detector_cfg,
        padding=0.2,
        splits=splits,
        db_name="cacd",
        find_persons=True,
        use_coarse_faces=True,
    )
