from typing import Tuple

import torch
from mivolo.model.mi_volo import MiVOLO

from .age_gender_dataset import AgeGenderDataset
from .age_gender_loader import create_loader
from .classification_dataset import AdienceDataset, FairFaceDataset

DATASET_CLASS_MAP = {
    "utk": AgeGenderDataset,
    "lagenda": AgeGenderDataset,
    "imdb": AgeGenderDataset,
    "agedb": AgeGenderDataset,
    "cacd": AgeGenderDataset,
    "adience": AdienceDataset,
    "fairface": FairFaceDataset,
}


def build(
    name: str,
    images_path: str,
    annotations_path: str,
    split: str,
    mivolo_model: MiVOLO,
    workers: int,
    batch_size: int,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:

    dataset_class = DATASET_CLASS_MAP[name]

    dataset: torch.utils.data.Dataset = dataset_class(
        images_path=images_path,
        annotations_path=annotations_path,
        name=name,
        split=split,
        target_size=mivolo_model.input_size,
        max_age=mivolo_model.meta.max_age,
        min_age=mivolo_model.meta.min_age,
        model_with_persons=mivolo_model.meta.with_persons_model,
        use_persons=mivolo_model.meta.use_persons,
        disable_faces=mivolo_model.meta.disable_faces,
        only_age=mivolo_model.meta.only_age,
    )

    data_config = mivolo_model.data_config

    in_chans = 3 if not mivolo_model.meta.with_persons_model else 6
    input_size = (in_chans, mivolo_model.input_size, mivolo_model.input_size)

    dataset_loader: torch.utils.data.DataLoader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=batch_size,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        pin_memory=False,
        device=mivolo_model.device,
        target_type=dataset.target_dtype,
    )

    return dataset, dataset_loader
