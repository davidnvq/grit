# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.box_ops import box_xyxy_to_cxcywh
from utils.misc import interpolate

import matplotlib.pyplot as plt


def show(img):
    plt.imshow(img)
    plt.show()


def draw_boxes(img, boxes, labels, scores=None, color='green', thresh=0.1, ind2label=None, ind2attr=None, target=None):

    from PIL import Image, ImageFont, ImageDraw, ImageEnhance
    from copy import deepcopy
    img = deepcopy(img)
    draw = ImageDraw.Draw(img)
    fi_labels = []
    if target.get('attributes', None) is not None:
        box_ids, attr_ids = torch.where(target['attributes'] > 0)

    for i, box in enumerate(boxes):
        if scores is not None and scores[i] < thresh:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle(((x1, y1), (x2, y2)), width=3, outline=color)
        label = int(labels[i])
        label = f"{ind2label[label]}"

        attr_text = ""
        if target.get('attributes', None) is not None:
            if len(torch.where(box_ids == i)) > 0:
                bbix = torch.where(box_ids == i)[0]
                for bbi in bbix:
                    attr_id = attr_ids[bbi]
                    attr_name = ind2attr[attr_id.item()]
                    if len(attr_text) > 0:
                        attr_text += ", "
                    attr_text += f"{attr_name}"

        if scores is not None:
            label += f": {scores[i]:.2f}"
            fi_labels.append(label)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 18)
        attr_fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 14)

        fill_color = (0, 255, 0)
        stroke_color = (0, 255, 0)
        draw.text((x1, y1 - 16), label, font=fnt, fill=fill_color, stroke_fill=stroke_color, stroke_width=1)
        draw.text((x1, y1 + 1), attr_text, font=attr_fnt, fill=(255, 0, 0), stroke_fill=(255, 0, 0), stroke_width=1)
    plt.imshow(img)
    plt.show()
    return img


def draw_item(item):
    import json
    label2ind = json.load(open('/home/quang/datasets/vg/vgcocooiobjects_v1_class2ind.json'))
    attr2ind = json.load(open('/home/quang/datasets/vg/annotations/attribute2ind.json'))
    ind2attr = {ind: attr for attr, ind in attr2ind.items()}
    ind2label = {ind: label for label, ind in label2ind.items()}
    target = item['target']
    draw_boxes(item['image'], target['boxes'], target['labels'], ind2label=ind2label, ind2attr=ind2attr, target=target)


# def draw_boxes(img, boxes, labels, scores=None, color='green', thresh=0.1, ind2label=None):

#     from PIL import Image, ImageFont, ImageDraw, ImageEnhance
#     from copy import deepcopy
#     img = deepcopy(img)
#     draw = ImageDraw.Draw(img)
#     fi_labels = []
#     for i, box in enumerate(boxes):
#         if scores is not None and scores[i] < thresh:
#             continue
#         x1, y1, x2, y2 = box
#         draw.rectangle(((x1, y1), (x2, y2)), width=3, outline=color)
#         # label = f"{ALL_CLASSES[labels[i]]}"
#         label = int(labels[i])
#         label = f"{ind2label[label]}"
#         if scores is not None:
#             label += f": {scores[i]:.2f}"
#             fi_labels.append(label)
#         fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 18)
#         fill_color = (0, 255, 0)
#         stroke_color = (0, 255, 0)
#         draw.text((x1, y1 - 16), label, font=fnt, fill=fill_color, stroke_fill=stroke_color, stroke_width=1)
#     plt.imshow(img)
#     plt.show()
#     return img


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd", "attributes"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            if field in target:
                target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    w, h = padded_image.size[0], padded_image.size[1]
    target["size"] = torch.tensor([h, w])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(min(self.min_size, img.width), min(img.width, self.max_size))
        h = random.randint(min(self.min_size, img.height), min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):

    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomRatioResize(object):

    def __init__(self, ratio_range, max_size=None):
        assert isinstance(ratio_range, (list, tuple))
        self.ratio_range = ratio_range
        self.max_size = max_size

    def __call__(self, img, target=None):
        ratio = random.uniform(*self.ratio_range)
        w = int(img.width * ratio)
        h = int(img.height * ratio)
        return resize(img, target, (w, h), self.max_size)


class PadOrCrop(object):

    def __init__(self, size=(448, 448)):
        assert isinstance(size, (list, tuple))
        self.size = size
        self.crop = RandomCrop(size)

    def __call__(self, img, target=None):
        if img.size[0] < self.size[0] or img.size[1] < self.size[1]:
            return pad(img, target, (self.size[0] - img.size[0], self.size[1] - img.size[1]))
        else:
            return self.crop(img, target)


class RandomPad(object):

    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):

    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


def make_transforms(split, phase='train'):
    normalize = Compose([ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if phase == 'train' or phase == 'finetune':

        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        scales = [
            480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800
        ]

        if split == 'train':
            return Compose([
                RandomHorizontalFlip(),
                RandomSelect(
                    RandomResize(scales, max_size=1333),
                    Compose([
                        RandomRatioResize((0.5, 1.5)),
                        RandomSizeCrop(384, 800),
                        RandomResize(scales, max_size=1333),
                    ]),
                ),
                normalize,
            ])

        if split == 'valid':
            return Compose([
                RandomResize([800], max_size=1333),
                normalize,
            ])

    elif phase == 'pretrain-od':
        if split == 'train':
            return Compose([
                RandomHorizontalFlip(),
                RandomResize([384, 400, 416, 432, 448], max_size=800),
                normalize,  # 
            ])

        if split == 'valid':
            return Compose([
                RandomResize([800], max_size=1333),
                normalize,
            ])

    elif phase == 'pretrain-vl':
        if split == 'train':
            return Compose([
                RandomResize([480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640], max_size=900),
                normalize,  # 
            ])

        if split == 'valid':
            return Compose([
                RandomResize([640], max_size=900),
                normalize,
            ])

    raise ValueError(f'unknown {split}')
