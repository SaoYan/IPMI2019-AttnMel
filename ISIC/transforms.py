import sys
import collections
import random
import numbers
from PIL import Image
import torchvision.transforms.functional as trF

#----------------------------------------------------------------------------
# Re-write transforms for our own use
# reference: official code https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
#----------------------------------------------------------------------------

if sys.version_info < (3, 3):
    Iterable = collections.Iterable
else:
    Iterable = collections.abc.Iterable

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        img = trF.resize(image, self.size, self.interpolation)
        img_seg = trF.resize(image_seg, self.size, Image.NEAREST)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class RatioCenterCrop(object):
    def __init__(self, ratio=1.):
        assert ratio <= 1. and ratio > 0
        # new_size = 0.8 * min(width, height)
        self.ratio = ratio

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        width, height = image.size
        new_size = self.ratio * min(width, height)
        img = trF.center_crop(image, new_size)
        img_seg = trF.center_crop(image_seg, new_size)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        img = trF.center_crop(image, self.size)
        img_seg = trF.center_crop(image_seg, self.size)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        i, j, h, w = self.get_params(image, self.size)
        img = trF.crop(image, i, j, h, w)
        img_seg = trF.crop(image_seg, i, j, h, w)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class RandomRotate(object):
    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        idx = random.randint(0,3)
        angle = idx * 90
        return angle

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        angle = self.get_params()
        img = trF.rotate(image, angle, self.resample, self.expand, self.center)
        img_seg = trF.rotate(image_seg, angle, self.resample, self.expand, self.center)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
            img = trF.hflip(image)
            img_seg = trF.hflip(image_seg)
            return {'image': img, 'image_seg': img_seg, 'label': label}
        return sample


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
            img = trF.vflip(image)
            img_seg = trF.vflip(image_seg)
            return {'image': img, 'image_seg': img_seg, 'label': label}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        img = trF.to_tensor(image)
        img_seg = trF.to_tensor(image_seg)
        return {'image': img, 'image_seg': img_seg, 'label': label}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, image_seg, label = sample['image'], sample['image_seg'], sample['label']
        img = trF.normalize(image, self.mean, self.std)
        return {'image': img, 'image_seg': image_seg, 'label': label}
