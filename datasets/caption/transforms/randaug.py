# ------------------------------------------------------------------------
# Modified from:
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
# https://github.com/dandelin/vilt/transforms/randaug.py
# ------------------------------------------------------------------------

import random
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Identity(img, v):
    return img


def augment_list():
    l = [
        (Identity, 0, 100),
        (AutoContrast, 0, 100),
        (Rotate, 0, 8),
        (Color, 0.5, 1.5),
        (Contrast, 0.5, 1.5),
        (Brightness, 0.5, 1.5),
        (Sharpness, 0.5, 1.5),
        (ShearX, 0.0, 0.12),
        (ShearY, 0.0, 0.12),
        (TranslateXabs, 0.0, 80),
        (TranslateYabs, 0.0, 80),
    ]
    return l


class RandAugment:

    def __init__(self, n_augments=4):
        self.n_augments = n_augments
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n_augments)
        for op, minval, maxval in ops:
            val = random.random() * (maxval - minval) + minval
            # print(f"{str(op.__name__):<15}{val:.3f} {minval:<2}-{maxval:<3}")
            img = op(img, val)

        return img