import torch
import torch.utils.data
from . import transforms as T

import os
import json
import lmdb
import random
import pickle
import os.path
from PIL import Image


class ObjectDetectionLMDB:

    def __init__(self, root, lmdb_file, transforms=None, second_root="", **kwargs):
        self.root = root
        self.second_root = second_root
        self.lmdb_file = lmdb_file
        # https://codeslake.github.io/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.env.close()
        self.env = None
        self.txn = None

        self.transforms = transforms
        self.kwargs = kwargs

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin()

    def __len__(self):
        if self.kwargs.get('overfit', False):
            return 64
        return len(self.keys)

    def get_image(self, path):
        path = path.replace("images/v1/", "images/v2/")
        if os.path.exists(os.path.join(self.root, path)):
            return Image.open(os.path.join(self.root, path)).convert('RGB')
        else:
            return Image.open(os.path.join(self.second_root, path)).convert('RGB')

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self))

        if self.txn is None:
            self._init_db()

        try:
            data = self.txn.get(self.keys[idx])
            img_path, target = pickle.loads(data)
            img = self.get_image(img_path)
        except:
            print(f"{img_path} doesn't exist!")
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if 'attributes' not in target:
            target['has_attr'] = torch.as_tensor(False)

        if len(target['labels']) == 0:
            return self.__getitem__(random_idx)

        return img, target


class ObjectDetectionDataset:

    def __init__(self, root, transforms=None, label2ind_file="", second_root="", **kwargs):
        self.root = root
        self.second_root = second_root
        self.transforms = transforms
        self.kwargs = kwargs
        self.label2ind = json.load(open(label2ind_file))
        self.label2ind = {k.lower(): v for k, v in self.label2ind.items()}
        self.ind2label = {v: k for k, v in self.label2ind.items()}

    def __len__(self):
        if self.kwargs.get('overfit', False):
            return 64
        else:
            return len(self.img_ids)

    def map_label2ind(self, label):
        return self.label2ind.get(label, -1)

    def box_clamp(self, boxes, w, h):
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        return boxes

    def remap_labels(self, labels):
        classes = [self.map_label2ind(label.lower()) for label in labels]
        classes = torch.tensor(classes, dtype=torch.int64)
        return classes

    def filter_objects(self, classes, boxes):
        class_keep = classes != -1
        box_keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        keep = torch.logical_and(box_keep, class_keep)

        boxes = boxes[keep]
        classes = classes[keep]
        return classes, boxes, keep

    def apply_transforms(self, img, target=None):
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def get_image(self, path):
        if os.path.exists(os.path.join(self.root, path)):
            return Image.open(os.path.join(self.root, path)).convert('RGB')
        else:
            return Image.open(os.path.join(self.second_root, path)).convert('RGB')

    def __repr__(self):
        out = super().__repr__()
        out += ("\tTransforms")
        out += "\n".join(["\t" + l for l in str(self.transforms).split("\n")])
        return out
