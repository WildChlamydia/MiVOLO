import logging
import os
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mivolo.data.data_reader import AnnotType, PictureInfo, get_all_files, read_csv_annotation_file
from mivolo.data.misc import IOU, class_letterbox
from timm.data.readers.reader import Reader
from tqdm import tqdm

CROP_ROUND_TOL = 0.3
MIN_PERSON_SIZE = 100
MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4

_logger = logging.getLogger("ReaderAgeGender")


class ReaderAgeGender(Reader):
    """
    Reader for almost original imdb-wiki cleaned dataset.
    Two changes:
        1. Your annotation must be in ./annotation subdir of dataset root
        2. Images must be in images subdir

    """

    def __init__(
        self,
        images_path,
        annotations_path,
        split="validation",
        target_size=224,
        min_size=5,
        seed=1234,
        with_persons=False,
        min_person_size=MIN_PERSON_SIZE,
        disable_faces=False,
        only_age=False,
        min_person_aftercut_ratio=MIN_PERSON_CROP_AFTERCUT_RATIO,
        crop_round_tol=CROP_ROUND_TOL,
    ):
        super().__init__()

        self.with_persons = with_persons
        self.disable_faces = disable_faces
        self.only_age = only_age

        # can be only black for now, even though it's not very good with further normalization
        self.crop_out_color = (0, 0, 0)

        self.empty_crop = np.ones((target_size, target_size, 3)) * self.crop_out_color
        self.empty_crop = self.empty_crop.astype(np.uint8)

        self.min_person_size = min_person_size
        self.min_person_aftercut_ratio = min_person_aftercut_ratio
        self.crop_round_tol = crop_round_tol

        splits = split.split(",")
        self.splits = [split.strip() for split in splits if len(split.strip())]
        assert len(self.splits), "Incorrect split arg"

        self.min_size = min_size
        self.seed = seed
        self.target_size = target_size

        # Reading annotations. Can be multiple files if annotations_path dir
        self._ann: Dict[str, List[PictureInfo]] = {}  # list of samples for each image
        self._associated_objects: Dict[str, Dict[int, List[List[int]]]] = {}
        self._faces_list: List[Tuple[str, int]] = []  # samples from this list will be loaded in __getitem__

        self._read_annotations(images_path, annotations_path)
        _logger.info(f"Dataset length: {len(self._faces_list)} crops")

    def __getitem__(self, index):
        return self._read_img_and_label(index)

    def __len__(self):
        return len(self._faces_list)

    def _filename(self, index, basename=False, absolute=False):
        img_p = self._faces_list[index][0]
        return os.path.basename(img_p) if basename else img_p

    def _read_annotations(self, images_path, csvs_path):
        self._ann = {}
        self._faces_list = []
        self._associated_objects = {}

        csvs = get_all_files(csvs_path, [".csv"])
        csvs = [c for c in csvs if any(split_name in os.path.basename(c) for split_name in self.splits)]

        # load annotations per image
        for csv in csvs:
            db, ann_type = read_csv_annotation_file(csv, images_path)
            if self.with_persons and ann_type != AnnotType.PERSONS:
                raise ValueError(
                    f"Annotation type in file {csv} contains no persons, "
                    f"but annotations with persons are requested."
                )
            self._ann.update(db)

        if len(self._ann) == 0:
            raise ValueError("Annotations are empty!")

        self._ann, self._associated_objects = self.prepare_annotations()
        images_list = list(self._ann.keys())

        for img_path in images_list:
            for index, image_sample_info in enumerate(self._ann[img_path]):
                assert image_sample_info.has_gt(
                    self.only_age
                ), "Annotations must be checked with self.prepare_annotations() func"
                self._faces_list.append((img_path, index))

    def _read_img_and_label(self, index):
        if not isinstance(index, int):
            raise TypeError("ReaderAgeGender expected index to be integer")

        img_p, face_index = self._faces_list[index]
        ann: PictureInfo = self._ann[img_p][face_index]
        img = cv2.imread(img_p)

        face_empty = True
        if ann.has_face_bbox and not (self.with_persons and self.disable_faces):
            face_crop, face_empty = self._get_crop(ann.bbox, img)

        if not self.with_persons and face_empty:
            # model without persons
            raise ValueError("Annotations must be checked with self.prepare_annotations() func")

        if face_empty:
            face_crop = self.empty_crop

        person_empty = True
        if self.with_persons or self.disable_faces:
            if ann.has_person_bbox:
                # cut off all associated objects from person crop
                objects = self._associated_objects[img_p][face_index]
                person_crop, person_empty = self._get_crop(
                    ann.person_bbox,
                    img,
                    crop_out_color=self.crop_out_color,
                    asced_objects=objects,
                )

            if face_empty and person_empty:
                raise ValueError("Annotations must be checked with self.prepare_annotations() func")

        if person_empty:
            person_crop = self.empty_crop

        return (face_crop, person_crop), [ann.age, ann.gender]

    def _get_crop(
        self,
        bbox,
        img,
        asced_objects=None,
        crop_out_color=(0, 0, 0),
    ) -> Tuple[np.ndarray, bool]:

        empty_bbox = False

        xmin, ymin, xmax, ymax = bbox
        assert not (
            ymax - ymin < self.min_size or xmax - xmin < self.min_size
        ), "Annotations must be checked with self.prepare_annotations() func"

        crop = img[ymin:ymax, xmin:xmax]

        if asced_objects:
            # cut off other objects for person crop
            crop, empty_bbox = _cropout_asced_objs(
                asced_objects,
                bbox,
                crop.copy(),
                crop_out_color=crop_out_color,
                min_person_size=self.min_person_size,
                crop_round_tol=self.crop_round_tol,
                min_person_aftercut_ratio=self.min_person_aftercut_ratio,
            )
            if empty_bbox:
                crop = self.empty_crop

        crop = class_letterbox(crop, new_shape=(self.target_size, self.target_size), color=crop_out_color)
        return crop, empty_bbox

    def prepare_annotations(self):

        good_anns: Dict[str, List[PictureInfo]] = {}
        all_associated_objects: Dict[str, Dict[int, List[List[int]]]] = {}

        if not self.with_persons:
            # remove all persons
            for img_path, bboxes in self._ann.items():
                for sample in bboxes:
                    sample.clear_person_bbox()

        # check dataset and collect associated_objects
        verify_images_func = partial(
            verify_images,
            min_size=self.min_size,
            min_person_size=self.min_person_size,
            with_persons=self.with_persons,
            disable_faces=self.disable_faces,
            crop_round_tol=self.crop_round_tol,
            min_person_aftercut_ratio=self.min_person_aftercut_ratio,
            only_age=self.only_age,
        )
        num_threads = min(8, os.cpu_count())

        all_msgs = []
        broken = 0
        skipped = 0
        all_skipped_crops = 0
        desc = "Check annotations..."
        with ThreadPool(num_threads) as pool:
            pbar = tqdm(
                pool.imap_unordered(verify_images_func, list(self._ann.items())),
                desc=desc,
                total=len(self._ann),
            )

            for (img_info, associated_objects, msgs, is_corrupted, is_empty_annotations, skipped_crops) in pbar:
                broken += 1 if is_corrupted else 0
                all_msgs.extend(msgs)
                all_skipped_crops += skipped_crops
                skipped += 1 if is_empty_annotations else 0
                if img_info is not None:
                    img_path, img_samples = img_info
                    good_anns[img_path] = img_samples
                    all_associated_objects.update({img_path: associated_objects})

                pbar.desc = (
                    f"{desc} {skipped} images skipped ({all_skipped_crops} crops are incorrect); "
                    f"{broken} images corrupted"
                )

            pbar.close()

        for msg in all_msgs:
            print(msg)
        print(f"\nLeft images: {len(good_anns)}")

        return good_anns, all_associated_objects


def verify_images(
    img_info,
    min_size: int,
    min_person_size: int,
    with_persons: bool,
    disable_faces: bool,
    crop_round_tol: float,
    min_person_aftercut_ratio: float,
    only_age: bool,
):
    # If crop is too small, if image can not be read or if image does not exist
    # then filter out this sample

    disable_faces = disable_faces and with_persons
    kwargs = dict(
        min_person_size=min_person_size,
        disable_faces=disable_faces,
        with_persons=with_persons,
        crop_round_tol=crop_round_tol,
        min_person_aftercut_ratio=min_person_aftercut_ratio,
        only_age=only_age,
    )

    def bbox_correct(bbox, min_size, im_h, im_w) -> Tuple[bool, List[int]]:
        ymin, ymax, xmin, xmax = _correct_bbox(bbox, im_h, im_w)
        crop_h, crop_w = ymax - ymin, xmax - xmin
        if crop_h < min_size or crop_w < min_size:
            return False, [-1, -1, -1, -1]
        bbox = [xmin, ymin, xmax, ymax]
        return True, bbox

    msgs = []
    skipped_crops = 0
    is_corrupted = False
    is_empty_annotations = False

    img_path: str = img_info[0]
    img_samples: List[PictureInfo] = img_info[1]
    try:
        im_cv = cv2.imread(img_path)
        im_h, im_w = im_cv.shape[:2]
    except Exception:
        msgs.append(f"Can not load image {img_path}")
        is_corrupted = True
        return None, {}, msgs, is_corrupted, is_empty_annotations, skipped_crops

    out_samples: List[PictureInfo] = []
    for sample in img_samples:
        # correct face bbox
        if sample.has_face_bbox:
            is_correct, sample.bbox = bbox_correct(sample.bbox, min_size, im_h, im_w)
            if not is_correct and sample.has_gt(only_age):
                msgs.append("Small face. Passing..")
                skipped_crops += 1

        # correct person bbox
        if sample.has_person_bbox:
            is_correct, sample.person_bbox = bbox_correct(
                sample.person_bbox, max(min_person_size, min_size), im_h, im_w
            )
            if not is_correct and sample.has_gt(only_age):
                msgs.append(f"Small person {img_path}. Passing..")
                skipped_crops += 1

        if sample.has_face_bbox or sample.has_person_bbox:
            out_samples.append(sample)
        elif sample.has_gt(only_age):
            msgs.append("Sample has no face and no body. Passing..")
            skipped_crops += 1

    # sort that samples with undefined age and gender be the last
    out_samples = sorted(out_samples, key=lambda sample: 1 if not sample.has_gt(only_age) else 0)

    # for each person find other faces and persons bboxes, intersected with it
    associated_objects: Dict[int, List[List[int]]] = find_associated_objects(out_samples, only_age=only_age)

    out_samples, associated_objects, skipped_crops = filter_bad_samples(
        out_samples, associated_objects, im_cv, msgs, skipped_crops, **kwargs
    )

    out_img_info: Optional[Tuple[str, List]] = (img_path, out_samples)
    if len(out_samples) == 0:
        out_img_info = None
        is_empty_annotations = True

    return out_img_info, associated_objects, msgs, is_corrupted, is_empty_annotations, skipped_crops


def filter_bad_samples(
    out_samples: List[PictureInfo],
    associated_objects: dict,
    im_cv: np.ndarray,
    msgs: List[str],
    skipped_crops: int,
    **kwargs,
):
    with_persons, disable_faces, min_person_size, crop_round_tol, min_person_aftercut_ratio, only_age = (
        kwargs["with_persons"],
        kwargs["disable_faces"],
        kwargs["min_person_size"],
        kwargs["crop_round_tol"],
        kwargs["min_person_aftercut_ratio"],
        kwargs["only_age"],
    )

    # left only samples with annotations
    inds = [sample_ind for sample_ind, sample in enumerate(out_samples) if sample.has_gt(only_age)]
    out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    if kwargs["disable_faces"]:
        # clear all faces
        for ind, sample in enumerate(out_samples):
            sample.clear_face_bbox()

        # left only samples with person_bbox
        inds = [sample_ind for sample_ind, sample in enumerate(out_samples) if sample.has_person_bbox]
        out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    if with_persons or disable_faces:
        # check that preprocessing func
        # _cropout_asced_objs() return not empty person_image for each out sample

        inds = []
        for ind, sample in enumerate(out_samples):
            person_empty = True
            if sample.has_person_bbox:
                xmin, ymin, xmax, ymax = sample.person_bbox
                crop = im_cv[ymin:ymax, xmin:xmax]
                # cut off all associated objects from person crop
                _, person_empty = _cropout_asced_objs(
                    associated_objects[ind],
                    sample.person_bbox,
                    crop.copy(),
                    min_person_size=min_person_size,
                    crop_round_tol=crop_round_tol,
                    min_person_aftercut_ratio=min_person_aftercut_ratio,
                )

            if person_empty and not sample.has_face_bbox:
                msgs.append("Small person after preprocessing. Passing..")
                skipped_crops += 1
            else:
                inds.append(ind)
        out_samples, associated_objects = _filter_by_ind(out_samples, associated_objects, inds)

    assert len(associated_objects) == len(out_samples)
    return out_samples, associated_objects, skipped_crops


def _filter_by_ind(out_samples, associated_objects, inds):
    _associated_objects = {}
    _out_samples = []
    for ind, sample in enumerate(out_samples):
        if ind in inds:
            _associated_objects[len(_out_samples)] = associated_objects[ind]
            _out_samples.append(sample)

    return _out_samples, _associated_objects


def find_associated_objects(
    image_samples: List[PictureInfo], iou_thresh=0.0001, only_age=False
) -> Dict[int, List[List[int]]]:
    """
    For each person (which has gt age and gt gender) find other faces and persons bboxes, intersected with it
    """
    associated_objects: Dict[int, List[List[int]]] = {}

    for iindex, image_sample_info in enumerate(image_samples):
        # add own face
        associated_objects[iindex] = [image_sample_info.bbox] if image_sample_info.has_face_bbox else []

        if not image_sample_info.has_person_bbox or not image_sample_info.has_gt(only_age):
            # if sample has not gt => not be used
            continue

        iperson_box = image_sample_info.person_bbox
        for jindex, other_image_sample in enumerate(image_samples):
            if iindex == jindex:
                continue
            if other_image_sample.has_face_bbox:
                jface_bbox = other_image_sample.bbox
                iou = _get_iou(jface_bbox, iperson_box)
                if iou >= iou_thresh:
                    associated_objects[iindex].append(jface_bbox)
            if other_image_sample.has_person_bbox:
                jperson_bbox = other_image_sample.person_bbox
                iou = _get_iou(jperson_bbox, iperson_box)
                if iou >= iou_thresh:
                    associated_objects[iindex].append(jperson_bbox)

    return associated_objects


def _cropout_asced_objs(
    asced_objects,
    person_bbox,
    crop,
    min_person_size,
    crop_round_tol,
    min_person_aftercut_ratio,
    crop_out_color=(0, 0, 0),
):
    empty = False
    xmin, ymin, xmax, ymax = person_bbox

    for a_obj in asced_objects:
        aobj_xmin, aobj_ymin, aobj_xmax, aobj_ymax = a_obj

        aobj_ymin = int(max(aobj_ymin - ymin, 0))
        aobj_xmin = int(max(aobj_xmin - xmin, 0))
        aobj_ymax = int(min(aobj_ymax - ymin, ymax - ymin))
        aobj_xmax = int(min(aobj_xmax - xmin, xmax - xmin))

        crop[aobj_ymin:aobj_ymax, aobj_xmin:aobj_xmax] = crop_out_color

    # calc useful non-black area
    remain_ratio = np.count_nonzero(crop) / (crop.shape[0] * crop.shape[1] * crop.shape[2])
    if (crop.shape[0] < min_person_size or crop.shape[1] < min_person_size) or remain_ratio < min_person_aftercut_ratio:
        crop = None
        empty = True

    return crop, empty


def _correct_bbox(bbox, h, w):
    xmin, ymin, xmax, ymax = bbox
    ymin = min(max(ymin, 0), h)
    ymax = min(max(ymax, 0), h)
    xmin = min(max(xmin, 0), w)
    xmax = min(max(xmax, 0), w)
    return ymin, ymax, xmin, xmax


def _get_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    iou = IOU(
        [ymin1, xmin1, ymax1, xmax1],
        [ymin2, xmin2, ymax2, xmax2],
    )
    return iou
