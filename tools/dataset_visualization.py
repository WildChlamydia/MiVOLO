import argparse
from typing import Dict, List

import cv2
from mivolo.data.data_reader import PictureInfo, read_csv_annotation_file
from ultralytics.yolo.utils.plotting import Annotator, colors


def get_parser():
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("--dataset_images", default="", type=str, required=True, help="path to images")
    parser.add_argument("--annotation_file", default="", type=str, required=True, help="path to annotations")

    return parser


def visualize(images_dir, new_annotation_file):

    bboxes_per_image: Dict[str, List[PictureInfo]] = read_csv_annotation_file(new_annotation_file, images_dir)[0]
    print(f"Found {len(bboxes_per_image)} unique images")

    for image_path, bboxes in bboxes_per_image.items():
        im_cv = cv2.imread(image_path)
        annotator = Annotator(im_cv)

        for i, bbox_info in enumerate(bboxes):
            label = f"{bbox_info.gender} Age: {bbox_info.age}"
            if any(coord != -1 for coord in bbox_info.bbox):
                # draw face bbox if exist
                annotator.box_label(bbox_info.bbox, label, color=colors(i, True))

            if any(coord != -1 for coord in bbox_info.person_bbox):
                # draw person bbox if exist
                annotator.box_label(bbox_info.person_bbox, "p " + label, color=colors(i, True))

        im_cv = annotator.result()
        cv2.imshow("image", im_cv)
        cv2.waitKey(0)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    visualize(args.dataset_images, args.annotation_file)
