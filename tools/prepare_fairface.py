import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd
import torch
import tqdm
from mivolo.data.data_reader import PictureInfo, get_all_files
from mivolo.modeling.yolo_detector import Detector, PersonAndFaceResult
from preparation_utils import assign_persons, associate_persons, get_additional_bboxes, get_main_face, save_annotations


def read_fairface_annotations(annotations_files):
    annotations_per_image = {}
    cols = ["file", "age", "gender"]

    for file in annotations_files:
        split_name = os.path.basename(file).split(".")[0].split("_")[-1]
        df = pd.read_csv(file, sep=",", usecols=cols)
        for index, row in df.iterrows():
            aligned_face_path = row["file"]

            age, gender = row["age"], row["gender"]
            # M or F
            gender = gender[0].upper() if isinstance(gender, str) else None
            age = age.replace("-", ";") if isinstance(age, str) else None

            annotations_per_image[aligned_face_path] = {"age": age, "gender": gender, "split": split_name}
    return annotations_per_image


def read_data(images_dir, annotations_files) -> Tuple[List[PictureInfo], List[PictureInfo]]:
    dataset_pictures_train: List[PictureInfo] = []
    dataset_pictures_val: List[PictureInfo] = []

    all_images = get_all_files(images_dir)
    annotations_per_file = read_fairface_annotations(annotations_files)

    SPLIT_TYPE = Dict[str, Dict[str, int]]
    splits_stat_per_gender: SPLIT_TYPE = defaultdict(lambda: defaultdict(int))
    splits_stat_per_ages: SPLIT_TYPE = defaultdict(lambda: defaultdict(int))

    age_map = {"more than 70": "70;120"}
    for image_path in all_images:
        relative_path = image_path.replace(f"{images_dir}/", "")

        annot = annotations_per_file[relative_path]
        split = annot["split"]
        age, gender = annot["age"], annot["gender"]
        age = age_map[age] if age in age_map else age

        splits_stat_per_gender[split][gender] += 1
        splits_stat_per_ages[split][age] += 1

        if split == "train":
            dataset_pictures_train.append(PictureInfo(image_path, age, gender))
        elif split == "val":
            dataset_pictures_val.append(PictureInfo(image_path, age, gender))
        else:
            raise ValueError(f"Unknown split name: {split}")

    print(f"Found train/val images: {len(dataset_pictures_train)}/{len(dataset_pictures_val)}")
    for split, stat_per_gender in splits_stat_per_gender.items():
        print(f"\n{split} Per gender images: {stat_per_gender}")

    for split, stat_per_ages in splits_stat_per_ages.items():
        ages = list(stat_per_ages.keys())
        print(f"\n{split} Per ages categories ({len(ages)} cats) :")
        ages = sorted(ages, key=lambda x: int(x.split(";")[0].strip()))
        for age in ages:
            print(f"Age: {age} Count: {stat_per_ages[age]}")

    return dataset_pictures_train, dataset_pictures_val


def find_persons_on_image(image_info, main_bbox, detected_objects, other_faces_inds, device):
    # find person_ind for each face (main + other_faces)
    all_faces: List[torch.tensor] = [torch.tensor(main_bbox).to(device)] + [
        detected_objects.get_bbox_by_ind(ind) for ind in other_faces_inds
    ]
    faces_persons_map, other_persons_inds = associate_persons(all_faces, detected_objects)

    additional_faces: List[PictureInfo] = get_additional_bboxes(
        detected_objects, other_faces_inds, image_info.image_path
    )

    # set person bboxes for all faces (main + additional_faces)
    assign_persons([image_info] + additional_faces, faces_persons_map, detected_objects)
    if faces_persons_map[0] is not None:
        assert all(coord != -1 for coord in image_info.person_bbox)

    additional_persons: List[PictureInfo] = get_additional_bboxes(
        detected_objects, other_persons_inds, image_info.image_path, is_person=True
    )

    return additional_faces, additional_persons


def main(faces_dir: str, annotations: List[str], data_dir: str, detector_cfg: dict = None):
    """
    Generate a .txt annotation file with columns:
        ["img_name", "age", "gender",
        "face_x0", "face_y0", "face_x1", "face_y1",
        "person_x0", "person_y0", "person_x1", "person_y1"]

    If detector_cfg is set, for each face bbox will be refined using detector.
        Person bbox will be assigned for each face.
        Also, other detected faces and persons wil be written to txt file (needed for further preprocessing)
    """
    # out directory for txt annotations
    out_dir = os.path.join(data_dir, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    # load annotations
    dataset_pictures_train, dataset_pictures_val = read_data(faces_dir, annotations)

    for images, split_name in zip([dataset_pictures_train, dataset_pictures_val], ["train", "val"]):

        if detector_cfg:
            # detect faces with yolo detector
            faces_not_found, images_with_other_faces = 0, 0
            other_faces: List[PictureInfo] = []

            detector_weights, device = detector_cfg["weights"], detector_cfg["device"]
            detector = Detector(detector_weights, device, verbose=False, conf_thresh=0.1, iou_thresh=0.2)
            for image_info in tqdm.tqdm(images, desc=f"Detecting {split_name} faces: "):
                cv_im = cv2.imread(image_info.image_path)
                im_h, im_w = cv_im.shape[:2]
                # all images are 448x448 and with 125 padding
                coarse_bbox = [125, 125, im_w - 125, im_h - 125]  # xyxy

                detected_objects: PersonAndFaceResult = detector.predict(cv_im)
                main_bbox, other_faces_inds = get_main_face(detected_objects, coarse_bbox)
                if len(other_faces_inds):
                    images_with_other_faces += 1

                if main_bbox is None:
                    # use a full image as face bbox
                    faces_not_found += 1
                    main_bbox = coarse_bbox
                image_info.bbox = main_bbox

                additional_faces, additional_persons = find_persons_on_image(
                    image_info, main_bbox, detected_objects, other_faces_inds, device
                )

                # add all additional faces
                other_faces.extend(additional_faces)

                # add persons with empty faces
                other_faces.extend(additional_persons)

            print(f"Faces not detected: {faces_not_found}/{len(images)}")
            print(f"Images with other faces: {images_with_other_faces}/{len(images)}")
            print(f"Other bboxes (faces/persons): {len(other_faces)}")

            images = images + other_faces

        else:
            for image_info in tqdm.tqdm(images, desc="Collect face bboxes: "):
                cv_im = cv2.imread(image_info.image_path)
                im_h, im_w = cv_im.shape[:2]
                # all images are 448x448 and with 125 padding
                image_info.bbox = [125, 125, im_w - 125, im_h - 125]  # xyxy

        save_annotations(images, faces_dir, out_file=os.path.join(out_dir, f"{split_name}_annotations.csv"))


def get_parser():
    parser = argparse.ArgumentParser(description="FairFace")
    parser.add_argument(
        "--dataset_path",
        default="data/FairFace",
        type=str,
        required=True,
        help="path to folder with fairface-img-margin125-trainval/ and fairface_label_{split}.csv",
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
    faces_dir = os.path.join(data_dir, "fairface-img-margin125-trainval")

    if data_dir[-1] == "/":
        data_dir = data_dir[:-1]

    annotations = [os.path.join(data_dir, "fairface_label_train.csv"), os.path.join(data_dir, "fairface_label_val.csv")]

    detector_cfg: Optional[Dict[str, str]] = None
    if args.detector_weights is not None:
        detector_cfg = {"weights": args.detector_weights, "device": "cuda:0"}

    main(faces_dir, annotations, data_dir, detector_cfg)
