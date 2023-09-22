import argparse
import json
import os
from typing import Dict, Optional

from prepare_cacd import collect_faces


def get_parser():
    parser = argparse.ArgumentParser(description="AgeDB")
    parser.add_argument(
        "--dataset_path",
        default="data/AgeDB",
        type=str,
        required=True,
        help="path to dataset with AgeDB folder",
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

    faces_dir = os.path.join(data_dir, "AgeDB")

    # https://github.com/paplhjak/Facial-Age-Estimation-Benchmark-Databases/tree/main
    json_path = os.path.join(data_dir, "AgeDB.json")
    with open(json_path, "r") as stream:
        annotations = json.load(stream)

    detector_cfg: Optional[Dict[str, str]] = None
    if args.detector_weights is not None:
        detector_cfg = {"weights": args.detector_weights, "device": "cuda:0"}

    splits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    collect_faces(
        faces_dir,
        annotations,
        data_dir,
        detector_cfg,
        padding=0.1,
        splits=splits,
        db_name="agedb",
        find_persons=True,
        use_coarse_faces=True,
    )
