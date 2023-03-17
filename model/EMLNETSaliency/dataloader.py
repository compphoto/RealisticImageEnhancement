# Two arguments are needed to load image files: one is the root of directory,
#                                               the other is a txt file containing filepath and label.
# The created ImageFile object can be passed to a pytorch DataLoader for multi-threading process.
#
# Author : Sen Jia 
#

import torch.utils.data as data

from PIL import Image
from PIL import ImageFilter
import os
import os.path
import numpy as np
import scipy.misc as misc
import torch
import torchvision.transforms as transforms

from random import randint
import random

def make_dataset(root,txt_file):
    images = []
    with open(txt_file,"r") as f:
        for line in f:
            strs = line.rstrip("\n").split(" ")
            images.append((os.path.join(root,strs[0]), os.path.join(root, strs[1]),os.path.join(root, strs[2])))
    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def map_loader(path):
    return Image.open(path).convert('L')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageList(data.Dataset):
    def __init__(self, root, txt_file, transform=None, target_transform=None, map_size=None,
                 loader=default_loader, map_loader=map_loader, size_out=None, aug=False):
        imgs = make_dataset(root, txt_file)
        if not imgs:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.map_loader = map_loader
        self.mid = False

    def __getitem__(self, index):
        def _post_process(smap):
            smap = smap - smap.min()
            smap = smap / smap.max()
            return smap

        img_path, fix_path, map_path = self.imgs[index]

        img = self.loader(img_path)
        w, h = img.size

        s_map = self.map_loader(map_path)

        if self.transform is not None:
            img = self.transform(img)
            s_map = self.transform(s_map)

        s_map = _post_process(s_map)

        return img, s_map,

    def __len__(self):
        return len(self.imgs)

