import json
import torch
import random
from .base import ObjectDetectionDataset
import random
import torch.utils.data
import lmdb
import pickle
import os
from PIL import Image


class VisualGenomeDataset(ObjectDetectionDataset):

    def __init__(self,
                 root,
                 ann_file,
                 coco_file="",
                 label2ind_file="",
                 attribute2ind_file="",
                 oid2attr_file="",
                 transforms=None,
                 **kwargs):
        super().__init__(root, transforms=transforms, label2ind_file=label2ind_file, **kwargs)
        self.ann_file = ann_file
        self.coco_file = coco_file
        self.anns = json.load(open(ann_file))
        self.anns = [ann for ann in self.anns if len(ann['objects']) > 0]
        self.img_ids = [ann['image_id'] for ann in self.anns]  # list(range(len(self.anns)))
        self.attribute2ind = json.load(open(attribute2ind_file))
        self.oid2attr = json.load(open(oid2attr_file))
        self.attribute_size = len(set([ind for _, ind in self.attribute2ind.items()]))
        # Toknow: How to get this no_ann_idxes
        # self.no_ann_idxes = {idx: idx for idx, ann in enumerate(self.anns) if len(ann['objects']) == 0}

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self))
        try:
            # assert idx not in self.no_ann_idxes, f"{idx} has no annotations."
            path = self.anns[idx]['img_path']
            img = self.get_image(path)
            target = self.prepare(img, idx)
            img, target = self.apply_transforms(img, target)
        except:
            print(f"{idx} in {self.__class__.__name__} cannot be obtained.")
            return self.__getitem__(random_idx)

        # box: [x1, y1, x2, y2]
        return img, target

    def get_attributes(self, anno):
        all_obj_attributes = []

        for obj in anno:
            attributes = self.oid2attr.get(obj['object_id'], self.oid2attr[str(obj['object_id'])])
            attribute_ids = [self.attribute2ind[attr] for attr in attributes]
            object_attributes = torch.zeros(self.attribute_size, dtype=torch.int64)
            if len(attribute_ids) > 0:
                labels = torch.ones(len(attribute_ids), dtype=torch.int64)
                object_attributes.scatter_(0, torch.tensor(attribute_ids, dtype=torch.int64), labels)
            all_obj_attributes.append(object_attributes)
        return torch.stack(all_obj_attributes)  # [num_bojects, attribute_size]

    def prepare(self, image, idx):
        w, h = image.size

        anno = self.anns[idx]['objects']

        # guard against no boxes via resizing
        boxes = [[obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h']] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        attributes = self.get_attributes(anno)

        labels = [obj['names'][0] for obj in anno]
        classes = self.remap_labels(labels)
        boxes = self.box_clamp(boxes, w=w, h=h)
        classes, boxes, keep = self.filter_objects(classes, boxes)
        attributes = attributes[keep]
        img_id = torch.tensor([self.img_ids[idx]])

        target = {
            'boxes': boxes,
            '_boxes': boxes,
            'labels': classes,
            'image_id': img_id,
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(boxes)),
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            'attributes': attributes,
            'has_attr': torch.as_tensor(True)
        }
        return target

    def get_lmdb(self, idx):
        try:
            img_path = self.anns[idx]['file_name']
            img = self.get_image(img_path)
            target = self.prepare(img, idx)
            img, target = self.apply_transforms(img, target)
            return img_path, target
        except:
            print(f"{idx} in {self.__class__.__name__} doesn't exist!")
            return None, None


class VgObjectDetectionLMDB:

    def __init__(self, root, lmdb_file, transforms=None, second_root="", **kwargs):
        self.root = root
        self.second_root = second_root
        self.lmdb_file = lmdb_file
        # https://codeslake.github.io/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.img_ids = pickle.loads(txn.get(b'img_ids'))
            self.img_ids = [i.item() if isinstance(i, torch.Tensor) else i for i in self.img_ids]

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
            return 512
        return len(self.img_ids)

    def get_image(self, path):
        if os.path.exists(os.path.join(self.root, path)):
            return Image.open(os.path.join(self.root, path)).convert('RGB')
        else:
            return Image.open(os.path.join(self.second_root, path)).convert('RGB')

    def __getitem__(self, idx):

        if self.txn is None:
            self._init_db()

        try:
            data = self.txn.get(str(self.img_ids[idx]).encode('ascii'))
            img_path, target = pickle.loads(data)
            img = self.get_image(img_path)
        except:
            print(f"{img_path} doesn't exist!")
            return None, None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if 'attributes' not in target:
            target['has_attr'] = torch.as_tensor(False)

        # if len(target['labels']) == 0:
        #     return None, None
        return img, target


def create_split_anns(ann_file, split='test'):
    dirname = os.path.dirname(ann_file)
    split_ann_file = os.path.join(dirname, f'{split}_objects.json')
    split_file = os.path.join(dirname, f'{split}.txt')
    with open(split_file, 'r') as f:
        lines = f.readlines()
        img_paths = [line.split(' ')[0] for line in lines]

    _anns = json.load(open(ann_file))

    anns = {}
    count = 0
    for ann in _anns:
        if 'image_url' not in ann:
            count += 1
            continue
        img_id = ann['image_url'].split('/')[-1]
        ann['img_id'] = img_id
        anns[img_id] = ann

    split_anns = []
    for img_path in img_paths:
        if img_path.split('/')[-1] in anns:
            ann = anns[img_path.split('/')[-1]]
            ann['img_path'] = img_path
            split_anns.append(ann)
    print(f"Len of split = {len(img_paths)}, len of split anns = {len(split_anns)}")

    with open(split_ann_file, 'w') as f:
        json.dump(split_anns, f)