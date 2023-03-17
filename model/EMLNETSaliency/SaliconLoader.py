# This file contains the dataloader used to read imagres and annotations from the SALICON dataset.
#
# Author : Sen Jia 
#

import random

import torch.utils.data as data
from PIL import Image
from scipy import io

def make_trainset(root):

    img_root = root / "images/train"
    fix_root = root / "fixations/train"
    map_root = root / "maps/train"

    files = [f.stem for f in img_root.glob("*.jpg")]

    images = []

    for f in files:
        img_path = (img_root / f).with_suffix(".jpg")
        fix_path = (fix_root / f).with_suffix(".mat")
        map_path = (map_root / f).with_suffix(".png")
        images.append([img_path, fix_path, map_path])

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def map_loader(path):
    return Image.open(path).convert('L')

def mat_loader(path, shape):
    mat = io.loadmat(path)["gaze"]
    fix = []
    for row in mat:
        data = row[0].tolist()[2]
        for p in data:
            if p[0]<shape[0] and p[1]<shape[1]: # remove noise at the boundary.
                fix.append(p.tolist())
    return fix

class ImageList(data.Dataset):
    def __init__(self, root, transform=None, train=False,
                 loader=default_loader, mat_loader=mat_loader, map_loader=map_loader):

        imgs = make_trainset(root)
        if not imgs:
            raise(RuntimeError("Found 0 images in folder: " + root + "\n"))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.train = train
        self.loader = loader
        self.map_loader = map_loader
        self.mat_loader = mat_loader

    def __getitem__(self, index):

        img_path, fix_path, map_path = self.imgs[index]

        img = self.loader(img_path)
        w, h = img.size
        fixpts = self.mat_loader(fix_path, (w, h))
        smap = self.map_loader(map_path)

        fixmap = self.pts2pil(fixpts, img)

        if self.train:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                smap = smap.transpose(Image.FLIP_LEFT_RIGHT)
                fixmap = fixmap.transpose(Image.FLIP_LEFT_RIGHT)


        if self.transform is not None:
            img = self.transform(img)
            smap = self.transform(smap)
            fixmap = self.transform(fixmap)
        return img, fixmap, smap

    def pts2pil(self, fixpts, img):
        fixmap = Image.new("L", img.size)
        for p in fixpts:
            fixmap.putpixel((p[0], p[1]), 255)
        return fixmap

    def __len__(self):
        return len(self.imgs)

