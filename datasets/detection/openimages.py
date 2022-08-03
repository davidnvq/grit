import json
import torch
import random
from .base import ObjectDetectionDataset


class OpenImagesDataset(ObjectDetectionDataset):

    def __init__(self, root, ann_file, label2ind_file="", transforms=None, **kwargs):
        super().__init__(root, transforms=transforms, label2ind_file=label2ind_file, **kwargs)
        self.anns = json.load(open(ann_file))
        self.img_ids = list(self.anns.keys())

    def __getitem__(self, idx):
        random_idx = random.randint(0, len(self))
        try:
            img_id = self.img_ids[idx]
            path = self.anns[img_id]['file_name']
            img = self.get_image(path)
        except:
            print(f"{idx} in {self.__class__.__name__} doesn't exist!")
            return self.__getitem__(random_idx)

        target = self.prepare(img_id, img, idx)

        img, target = self.apply_transforms(img, target)

        if len(target['labels']) == 0:
            return self.__getitem__(random_idx)

        # box: [x1, y1, x2, y2]
        return img, target

    def get_lmdb(self, idx):
        img_id = self.img_ids[idx]
        path = self.anns[img_id]['file_name']
        img = self.get_image(path)
        target = self.prepare(img_id, img, idx)

        # img, target = self.apply_transforms(img, target)
        return path, target

    def prepare(self, img_id, image, idx):
        w, h = image.size
        anno = self.anns[img_id]['objects']

        # guard against no boxes via resizing
        boxes = [[obj['xmin'] * w, obj['ymin'] * h, obj['xmax'] * w, obj['ymax'] * h] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        labels = [obj['label'] for obj in anno]
        classes = self.remap_labels(labels)
        boxes = self.box_clamp(boxes, w=w, h=h)
        classes, boxes, keep = self.filter_objects(classes, boxes)

        img_id = torch.tensor(img_id) if isinstance(img_id, int) else torch.tensor(idx)

        target = {
            'boxes': boxes,
            'labels': classes,
            'image_id': img_id,
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(boxes)),
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            'has_attr': torch.as_tensor(False)
        }
        return target
