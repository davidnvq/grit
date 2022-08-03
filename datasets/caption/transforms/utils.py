from PIL import Image


class MaxWHResize:

    def __init__(self, size):
        self.size = size
        self.max_h = size[0]
        self.max_w = size[1]

    def __call__(self, x):
        w, h = x.size
        scale = min(self.max_w / w, self.max_h / h)
        neww = int(w * scale)
        newh = int(h * scale)
        return x.resize((neww, newh), resample=Image.BICUBIC)


class MinMaxResize:

    def __init__(self, size):
        self.size = size
        self.min = size[0]
        self.max = size[1]

    def __call__(self, x):
        w, h = x.size
        scale = self.min / min(w, h)
        if h < w:
            newh, neww = self.min, scale * w
        else:
            newh, neww = scale * h, self.min

        if max(newh, neww) > self.max:
            scale = self.max / max(newh, neww)
            newh = newh * scale
            neww = neww * scale

        newh, neww = int(newh + 0.5), int(neww + 0.5)
        newh, neww = newh // 32 * 32, neww // 32 * 32

        return x.resize((neww, newh), resample=Image.BICUBIC)
