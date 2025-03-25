import logging
from typing import Callable

from datasets.bone.bone import get_bone_dataset
from datasets.bone.bone_attacked import get_bone_attacked_dataset
from datasets.bone.bone_attacked_hm import get_bone_attacked_hm_dataset
from datasets.isic.isic import get_isic_dataset
from datasets.isic.isic_attacked import get_isic_attacked_dataset
from datasets.isic.isic_attacked_hm import get_isic_attacked_hm_dataset
from datasets.isic.isic_hm import get_isic_hm_dataset
from datasets.funnybirds.funnybirds import get_funnybirds
from datasets.funnybirds.funnybirds_attributes import get_funnybirds_attributes

logger = logging.getLogger(__name__)

DATASETS = {
    "funnybirds_forced_concept": get_funnybirds,
    "funnybirds": get_funnybirds,
    "funnybirds_attribute": get_funnybirds_attributes,
    "isic": get_isic_dataset,
    "isic_hm": get_isic_hm_dataset,
    "isic_attacked": get_isic_attacked_dataset,
    "isic_attacked_hm": get_isic_attacked_hm_dataset,
    "bone": get_bone_dataset,
    "bone_attacked": get_bone_attacked_dataset,
    "bone_attacked_hm": get_bone_attacked_hm_dataset,
}


def get_dataset(dataset_name: str) -> Callable:
    """
    Get dataset by name.
    :param dataset_name: Name of the dataset.
    :return: Dataset.

    """
    if dataset_name in DATASETS:
        dataset = DATASETS[dataset_name]
        logger.info(f"Loading {dataset_name}")
        return dataset
    else:
        raise KeyError(f"DATASET {dataset_name} not defined.")
    
def get_dataset_kwargs(config):
    dataset_specific_kwargs = {
        "label_map_path": config["label_map_path"],
        "classes": config.get("classes", None),
        "train": True
    } if "imagenet" in config['dataset_name'] else {}

    return dataset_specific_kwargs
