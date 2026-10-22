"""Minimal frozen dataset configurations for the standalone backbone."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CoreLiteConfig:
    name: str
    num_layers: int
    num_classes: int
    sub_sampling_ratio: Tuple[int, ...]
    input_channels: int
    use_remission: bool = False
    k_n: int = 16

    @classmethod
    def for_dataset(cls, dataset_name):
        specs = {
            "S3DIS": dict(
                num_layers=5,
                num_classes=13,
                sub_sampling_ratio=(4, 4, 4, 4, 2),
                input_channels=6,
            ),
            "SemanticKITTI": dict(
                num_layers=4,
                num_classes=19,
                sub_sampling_ratio=(4, 4, 4, 4),
                input_channels=4,
                use_remission=True,
            ),
            "SensatUrban": dict(
                num_layers=5,
                num_classes=13,
                sub_sampling_ratio=(4, 4, 4, 4, 2),
                input_channels=6,
            ),
        }
        if dataset_name not in specs:
            raise ValueError("Unknown frozen dataset: {}".format(dataset_name))
        return cls(name=dataset_name, **specs[dataset_name])
