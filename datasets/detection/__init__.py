from . import coco
from . import objects365
from . import openimages
from . import visualgenome
from . import base

from .coco import CocoDataset
from .transforms import make_transforms
from .openimages import OpenImagesDataset
from .objects365 import Objects365Dataset
from .visualgenome import VisualGenomeDataset
from .base import ObjectDetectionDataset, ObjectDetectionLMDB

from torch.utils.data import ConcatDataset

__all__ = {
    'coco_train': CocoDataset,
    'coco_val': CocoDataset,
    'vg_val': VisualGenomeDataset,
    'vg_test': VisualGenomeDataset,
    'vg_train': VisualGenomeDataset,
    'openimages': ObjectDetectionLMDB,
    'objects365': ObjectDetectionLMDB
}


def _get_kwargs(config, split='train'):
    overfit = getattr(config, 'overfit', False)
    phase = getattr(config, 'phase', 'finetune')
    transforms = make_transforms(split=split, phase=phase)
    return {'overfit': overfit, 'transforms': transforms}


def build_train_lmdb(config):
    datasets = []
    for key in config:
        if key in ['overfit', 'phase']:
            continue
        this_dataset = ObjectDetectionLMDB(**config[key], **_get_kwargs(config, 'train'))
        print(f"{key:<10} dataset: {len(this_dataset):8d} images, with {config[key].num_copies} copies.")
        datasets += [this_dataset] * config[key].num_copies
    return ConcatDataset(datasets)


def build_train_dataset(config):
    datasets = []
    for key in config:
        if key in ['overfit', 'phase']:
            continue
        this_dataset = __all__[key](**config[key], **_get_kwargs(config, 'train'))
        print(f"{key:<10} dataset: {len(this_dataset):8d} images, with {config[key].num_copies} copies.")
        datasets += [this_dataset] * config[key].num_copies
    concat_dataset = ConcatDataset(datasets)
    print("Total samples:", len(concat_dataset))
    return concat_dataset


def build_valid_dataset(config):
    print("We will use COCO for validation.")
    datasets = {}
    for dataset_name in config:
        if dataset_name in __all__:
            datasets[dataset_name] = __all__[dataset_name](**config[dataset_name], **_get_kwargs(config, 'valid'))
    return datasets
