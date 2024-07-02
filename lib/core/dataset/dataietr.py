import os
import cv2
import aug
from copy import deepcopy
from functools import partial
from glob import glob
from hashlib import sha1
import numpy as np
from torch.utils.data import Dataset

def subsample(data, bounds, hash_fn, n_buckets=100, salt=''):
    data = list(data)
    buckets = split_into_buckets(data, n_buckets=n_buckets, salt=salt, hash_fn=hash_fn)
    lower_bound, upper_bound = [x * n_buckets for x in bounds]
    return np.array([sample for bucket, sample in zip(buckets, data) if lower_bound <= bucket < upper_bound])

def hash_from_paths(x, salt):
    path_a, path_b = x
    names = ''.join(map(os.path.basename, (path_a, path_b)))
    return sha1(f'{names}_{salt}'.encode()).hexdigest()

def split_into_buckets(data, n_buckets, hash_fn, salt=''):
    hashes = map(partial(hash_fn, salt=salt), data)
    return np.array([int(x, 16) % n_buckets for x in hashes])

def _read_img(x: str):
    img = cv2.imread(x)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class TrainDataset(Dataset):
    def __init__(self, cfg):
        blur, sharp = map(lambda x: sorted(glob(x, recursive=True)), (cfg.DATA.train_blur, cfg.DATA.train_sharp))
        data = subsample(data=zip(blur, sharp),
                         bounds=[0, 1],
                         hash_fn=hash_from_paths)
        blur, sharp = map(list, zip(*data))
        self.blur = blur
        self.sharp = sharp
        self.transform_fn = aug.get_transforms(size=cfg.DATA.Patch_size)
        self.normalize_fn = aug.get_normalize()

    def _preprocess(self, img, res):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))

    def __len__(self):
        return len(self.blur)

    def __getitem__(self, idx):
        blur, sharp = self.data_a[idx], self.data_b[idx]
        blur, sharp = map(_read_img, (blur, sharp))

        blur_patch_1, sharp_patch_1 = self.transform_fn(blur, sharp)
        blur_patch_2, sharp_patch_2 = self.transform_fn(blur, sharp)

        blur_patch_1, sharp_patch_1 = self._preprocess(blur_patch_1, sharp_patch_1)
        blur_patch_2, sharp_patch_2 = self._preprocess(blur_patch_2, sharp_patch_2)

        return blur_patch_1, blur_patch_2, sharp_patch_1, sharp_patch_2

class ValDataset(Dataset):
    def __init__(self, cfg):
        files_a, files_b = map(lambda x: sorted(glob(x, recursive=True)), (cfg.DATA.val_blur, cfg.DATA.val_sharp))

        self.data_a = files_a
        self.data_b = files_b
        self.normalize_fn = aug.get_normalize()

    def _preprocess(self, img, res):
        def transpose(x):
            return np.transpose(x, (2, 0, 1))

        return map(transpose, self.normalize_fn(img, res))

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        blur, sharp = self.data_a[idx], self.data_b[idx]
        blur, sharp = map(_read_img, (blur, sharp))
        blur, sharp = self._preprocess(blur, sharp)
        return blur, sharp