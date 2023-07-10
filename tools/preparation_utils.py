from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from mivolo.data.data_reader import PictureInfo
from mivolo.data.misc import assign_faces, box_iou
from mivolo.model.yolo_detector import PersonAndFaceResult


def save_annotations(images: List[PictureInfo], images_dir: str, out_file: str):
    def get_age_str(age: Optional[str]) -> str:
        age = "-1" if age is None else age.replace("(", "").replace(")", "").replace(" ", "").replace(",", ";")
        return age

    def get_gender_str(gender: Optional[str]) -> str:
        gender = "-1" if gender is None else gender
        return gender

    headers = [
        "img_name",
        "age",
        "gender",
        "face_x0",
        "face_y0",
        "face_x1",
        "face_y1",
        "person_x0",
        "person_y0",
        "person_x1",
        "person_y1",
    ]
    output_data = []
    for image_info in images:
        relative_image_path = image_info.image_path.replace(f"{images_dir}/", "")
        face_x0, face_y0, face_x1, face_y1 = image_info.bbox
        p_x0, p_y0, p_x1, p_y1 = image_info.person_bbox
        output_data.append(
            {
                "img_name": relative_image_path,
                "age": get_age_str(image_info.age),
                "gender": get_gender_str(image_info.gender),
                "face_x0": face_x0,
                "face_y0": face_y0,
                "face_x1": face_x1,
                "face_y1": face_y1,
                "person_x0": p_x0,
                "person_y0": p_y0,
                "person_x1": p_x1,
                "person_y1": p_y1,
            }
        )
    output_df = pd.DataFrame(output_data, columns=headers)
    output_df.to_csv(out_file, sep=",", index=False)
    print(f"Saved annotations for {len(images)} images to {out_file}")


def get_main_face(
    detected_objects: PersonAndFaceResult, coarse_bbox: Optional[List[int]] = None, coarse_thresh: float = 0.2
) -> Tuple[Optional[List[int]], List[int]]:
    """
    Returns:
        main_bbox (Optional[List[int]]): The most cenetered face bbox
        other_bboxes (List[int]): indexes of other faces
    """
    face_bboxes_inds: List[int] = detected_objects.get_bboxes_inds("face")
    if len(face_bboxes_inds) == 0:
        return None, []

    # sort found faces
    face_bboxes_inds = sorted(face_bboxes_inds, key=lambda bb_ind: detected_objects.get_distance_to_center(bb_ind))
    most_centered_bbox_ind = face_bboxes_inds[0]
    main_bbox = detected_objects.get_bbox_by_ind(most_centered_bbox_ind).cpu().numpy().tolist()

    iou_matrix: List[float] = [1.0] + [0.0] * (len(face_bboxes_inds) - 1)

    if coarse_bbox is not None:
        # calc iou between coarse_bbox and found bboxes
        found_bboxes: List[torch.tensor] = [
            detected_objects.get_bbox_by_ind(other_ind) for other_ind in face_bboxes_inds
        ]

        iou_matrix = (
            box_iou(torch.stack([torch.tensor(coarse_bbox)]), torch.stack(found_bboxes).cpu()).numpy()[0].tolist()
        )

    if iou_matrix[0] < coarse_thresh:
        # to avoid fp detections
        main_bbox = None
        other_bboxes = [ind for i, ind in enumerate(face_bboxes_inds[1:]) if iou_matrix[i] < coarse_thresh]
    else:
        other_bboxes = face_bboxes_inds[1:]

    return main_bbox, other_bboxes


def get_additional_bboxes(
    detected_objects: PersonAndFaceResult, other_bboxes_inds: List[int], image_path: str, **kwargs
) -> List[PictureInfo]:
    is_face = True if "is_person" not in kwargs else False
    is_person = False if "is_person" not in kwargs else True

    additional_data: List[PictureInfo] = []
    # extend other faces
    for other_ind in other_bboxes_inds:
        other_box: List[int] = detected_objects.get_bbox_by_ind(other_ind).cpu().numpy().tolist()
        if is_face:
            additional_data.append(PictureInfo(image_path, None, None, other_box))
        elif is_person:
            additional_data.append(PictureInfo(image_path, None, None, person_bbox=other_box))
    return additional_data


def associate_persons(face_bboxes: List[torch.tensor], detected_objects: PersonAndFaceResult):
    person_bboxes_inds: List[int] = detected_objects.get_bboxes_inds("person")
    person_bboxes: List[torch.tensor] = [detected_objects.get_bbox_by_ind(ind) for ind in person_bboxes_inds]

    face_to_person_map: Dict[int, Optional[int]] = {ind: None for ind in range(len(face_bboxes))}

    if len(person_bboxes) == 0:
        return face_to_person_map, []

    assigned_faces, unassigned_persons_inds = assign_faces(person_bboxes, face_bboxes)

    for face_ind, person_ind in enumerate(assigned_faces):
        person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None
        face_to_person_map[face_ind] = person_ind

    unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds]
    return face_to_person_map, unassigned_persons_inds


def assign_persons(
    faces_info: List[PictureInfo], faces_persons_map: Dict[int, int], detected_objects: PersonAndFaceResult
):
    for face_ind, person_ind in faces_persons_map.items():
        if person_ind is None:
            continue

        person_bbox = detected_objects.get_bbox_by_ind(person_ind).cpu().numpy().tolist()
        faces_info[face_ind].person_bbox = person_bbox
