import torch.utils.data
from .coco import CocoDataset


class Objects365Dataset(CocoDataset):

    def __init__(self, root, ann_file, label2ind_file="", transforms=None, **kwargs):
        super().__init__(root, ann_file, transforms=transforms, label2ind_file=label2ind_file, **kwargs)

    def get_image(self, path):
        path = path.replace("images/v1/", "images/v2/")
        return super().get_image(path)

    def map_label2ind(self, label):
        if "/" in label:
            label = " ".join([l.strip() for l in label.split("-")])  #
        if label in self.label2ind:
            return self.label2ind[label]
        if "/" in label:
            for sublabel in label.split("/"):
                if sublabel in self.label2ind:
                    return self.label2ind[sublabel]
        return -1

    def map_label2ind(self, label):
        if "-" in label:
            label = " ".join(label.split("-")[::-1])  #
            label = label.replace("stuff", "").strip()
            label = label.replace("other", "").strip()
        return self.label2ind.get(label, -1)

    def get_lmdb(self, idx):
        img_id = self.img_ids[idx]
        coco = self.coco

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = img_info['file_name']
        w, h = img_info['width'], img_info['height']
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
        return img_path, target
