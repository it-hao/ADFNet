import random

import numpy as np
import skimage.io as sio
import skimage.color as sc

import torch
from torchvision import transforms

def get_patch(img_tar, patch_size):
    h, w = img_tar.shape[:2]

    x = random.randrange(0, w - patch_size + 1)
    y = random.randrange(0, h - patch_size + 1)

    img_tar = img_tar[y:y + patch_size, x:x + patch_size, :]

    return img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = sc.rgb2gray(img)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        # if hflip: img = img[:, ::-1]
        # if vflip: img = img[::-1, :]
        # if rot90: img = img.transpose(1, 0)

        return img

    return [_augment(_l) for _l in l]
