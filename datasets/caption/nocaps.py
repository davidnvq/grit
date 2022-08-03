import os
import json
from PIL import Image
from .transforms.utils import MinMaxResize
from .transforms import *

from engine.utils import nested_tensor_from_tensor_list

OVERFIT_SIZE = 64


class NocapsDataset:

    def __init__(
        self,
        vocab,
        ann_path,
        root,
        pad_idx=3,
        transform=None,
    ):
        anns = json.load(open(ann_path))['images']
        self.imageid_to_anns = {ann['id']: ann for ann in anns}
        self.root = root
        self.image_ids = list(self.imageid_to_anns.keys())
        self.vocab = vocab
        self.pad_idx = pad_idx

        self.transform = transform
        if self.transform is None:
            self.transform = Compose([MinMaxResize((384, 640)), ToTensor(), normalize()])

    def __getitem__(self, index: int):
        item = {}
        item['image_id'] = self.image_ids[index]
        ann = self.imageid_to_anns[item['image_id']]

        img_path = os.path.join(self.root, ann['file_name'])
        item['sample'] = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            item['sample'] = self.transform(item['sample'])
        return item

    def __len__(self):
        return len(self.image_ids)


class NoCapsCollator:

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch_list):
        batch = {}
        imgs = [item['sample'] for item in batch_list]
        batch['samples'] = nested_tensor_from_tensor_list(imgs).to(self.device)
        batch['image_id'] = [item['image_id'] for item in batch_list]
        return batch
