import random
import json
import torch
import torch.utils.data
from pycocotools.coco import COCO
from .base import ObjectDetectionDataset
import lmdb
import pickle
import os
from PIL import Image


class CocoDataset(ObjectDetectionDataset):

    def __init__(self,
                 root,
                 ann_file,
                 transforms=None,
                 label2ind_file="",
                 stuff_ann_file=None,
                 karpathy_val_files=[],
                 **kwargs):
        super().__init__(root, transforms=transforms, label2ind_file=label2ind_file, **kwargs)
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))

        if 'val' not in root:
            karpathy_ids = {}
            for file in karpathy_val_files:
                with open(file, 'r') as f:
                    lines = f.readlines()
                    karpathy_ids.update({int(line.split(' ')[1]): line for line in lines})
            self.img_ids = [img_id for img_id in self.img_ids if img_id not in karpathy_ids]

        self.coco_stuff = None
        if stuff_ann_file is not None:
            self.coco_stuff = COCO(stuff_ann_file)

        self.do_map = False if "val" in ann_file else True

        self.image_anns = json.load(open(ann_file))['images']
        self.image_anns = {item['id']: item for item in self.image_anns}

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self))
        try:
            img_id = self.img_ids[idx]
            path = self.coco.loadImgs(img_id)[0]['file_name']
            img = self.get_image(path)
        except:
            print(f"{idx} in {self.__class__.__name__} doesn't exist!")
            return self.__getitem__(random_idx)

        target = self.prepare(self.coco, img_id, img, idx)

        if self.coco_stuff is not None:
            stuff_target = self.prepare(self.coco_stuff, img_id, img, idx)
            for key in ['boxes', 'area', 'iscrowd', 'labels']:
                target[key] = torch.cat([target[key], stuff_target[key]], dim=0)

        # before normalize [x1, y1, x2, y2]
        img, target = self.apply_transforms(img, target)

        if len(target['labels']) == 0:
            return self.__getitem__(random_idx)

        # box: [cx, cy, w, h]
        return img, target

    def map_label2ind(self, label):
        if "-" in label:
            label = " ".join(label.split("-")[::-1])  #
            label = label.replace("stuff", "").strip()
            label = label.replace("other", "").strip()
        return self.label2ind.get(label, -1)

    def prepare(self, coco, img_id, image, idx):
        w, h = image.size

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        if self.do_map:
            labels = [coco.cats[obj["category_id"]]['name'].lower() for obj in anno]
            classes = self.remap_labels(labels)
        else:
            classes = [obj["category_id"] for obj in anno]
            classes = torch.tensor(classes, dtype=torch.int64)

        boxes = self.box_clamp(boxes, w=w, h=h)
        classes, boxes, keep = self.filter_objects(classes, boxes)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])

        img_id = torch.tensor(img_id) if isinstance(img_id, int) else torch.tensor(idx)

        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': img_id,
            'area': area[keep],
            'iscrowd': iscrowd[keep],
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            'has_attr': torch.as_tensor(False)
        }
        return target

    def get_lmdb(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        h = self.image_anns[img_id]['height']
        w = self.image_anns[img_id]['width']

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        if self.do_map:
            labels = [self.coco.cats[obj["category_id"]]['name'].lower() for obj in anno]
            classes = self.remap_labels(labels)
        else:
            classes = [obj["category_id"] for obj in anno]
            classes = torch.tensor(classes, dtype=torch.int64)

        boxes = self.box_clamp(boxes, w=w, h=h)
        classes, boxes, keep = self.filter_objects(classes, boxes)

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])

        # img_id = torch.tensor(img_id) if isinstance(img_id, int) else torch.tensor(idx)

        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': img_id,
            'area': area[keep],
            'iscrowd': iscrowd[keep],
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            'has_attr': torch.as_tensor(False)
        }
        return img_path, target


class CocoObjectDetectionLMDB:

    def __init__(self, root, lmdb_file, transforms=None, second_root="", **kwargs):
        self.root = root
        self.second_root = second_root
        self.lmdb_file = lmdb_file
        # https://codeslake.github.io/research/pytorch/How-to-use-LMDB-with-PyTorch-DataLoader/
        self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.img_ids = pickle.loads(txn.get(b'img_ids'))  # list(img_id)
            self.imgid2idx = {img_id: idx for idx, img_id in enumerate(self.img_ids)}

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
        random_idx = random.randint(0, len(self))

        if self.txn is None:
            self._init_db()

        try:
            data = self.txn.get(str(self.img_ids[idx]).encode('ascii'))
            img_path, target = pickle.loads(data)
            img = self.get_image(img_path)
        except:
            print(f"{img_path} doesn't exist!")
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        if 'attributes' not in target:
            target['has_attr'] = torch.as_tensor(False)

        # if len(target['labels']) == 0:
        #     return self.__getitem__(random_idx)

        return img, target