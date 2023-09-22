from typing import Any, List, Optional

import torch

from .age_gender_dataset import AgeGenderDataset


class ClassificationDataset(AgeGenderDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dtype = torch.int32

    def set_age_classes(self) -> Optional[List[str]]:
        raise NotImplementedError

    def parse_target(self, age: str, gender: str) -> List[Any]:
        assert self.age_classes is not None
        if age != "-1":
            assert age in self.age_classes, f"Unknown category in {self.name} dataset: {age}"
            age_ind = self.age_classes.index(age)
        else:
            age_ind = -1

        target: List[int] = [age_ind, int(self.parse_gender(gender))]
        return target


class FairFaceDataset(ClassificationDataset):
    def set_age_classes(self) -> Optional[List[str]]:
        age_classes = ["0;2", "3;9", "10;19", "20;29", "30;39", "40;49", "50;59", "60;69", "70;120"]
        # a[i-1] <= v < a[i] => age_classes[i-1]
        self._intervals = torch.tensor([0, 3, 10, 20, 30, 40, 50, 60, 70])
        return age_classes


class AdienceDataset(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.target_dtype = torch.int32

    def set_age_classes(self) -> Optional[List[str]]:
        age_classes = ["0;2", "4;6", "8;12", "15;20", "25;32", "38;43", "48;53", "60;100"]
        # a[i-1] <= v < a[i] => age_classes[i-1]
        self._intervals = torch.tensor([0, 4, 7, 14, 24, 36, 46, 57])
        return age_classes
