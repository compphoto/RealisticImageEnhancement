from __future__ import annotations
import os
from pyparsing import anyOpenTag

import torch
from  torchvision.transforms.functional import resize as Tresize
import torchvision.transforms as transforms
import torchvision

from utils.datautils import get_transform
from PIL import Image
import cv2
import csv
import random
import numpy as np
from PIL import ImageFilter

def save_mask(mask,name):
    cv2.imwrite(name+'.jpg',(mask*255).astype(np.uint8))

def alphatobox(alpha):
    validmap = alpha != 0 
    positions = np.where(validmap == True)
    x_min = positions[0].min()
    x_max = positions[0].max()
    y_min = positions[1].min()
    y_max = positions[1].max()
    return [x_min, y_min, x_max, y_max]

class MixLoader:
    def __init__(self, args):
        self.args = args

        self.rgb_root = '/localhome/smh31/Repositories/SalBasedImageEnhancement/multimasksamples/rgb'
        self.mask_root = '//localhome/smh31/Repositories/SalBasedImageEnhancement/multimasksamples/mask'

        mask_files = sorted([file for file in os.listdir(self.mask_root) if file.endswith('.jpg')])
        
        image_mask_dict = {}
        for file in mask_files:
            split_ =  os.path.basename(file).split('_')
            if split_[-1] in ['increase.jpg','decrease.jpg']:
                prefer_dir = split_[-1].replace('.jpg','')
                mask_index = '_'.join(split_[-2:]).replace('.jpg','')
                basename = '_'.join(split_[:-2])
            else:
                mask_index = split_[-1].replace('.jpg','')
                basename = '_'.join(split_[:-1])
                # prefer_dir = 'increase' if random.random() < 0.5 else 'decrease'
                prefer_dir = 'NA'
            if basename in image_mask_dict:
                image_mask_dict[basename] = image_mask_dict[basename] + [(mask_index,prefer_dir)]
            else:
                image_mask_dict[basename] = [(mask_index,prefer_dir)]
        
        self.image_mask_dict = image_mask_dict

        self.files = [item for item in image_mask_dict.keys()]
        self.mask_size = len(self.files)

        opt = {}
        opt['load_size'] = args.load_size
        opt['crop_size'] = args.crop_size
        opt['preprocess'] = 'resize'
        opt['no_flip'] = True

        self.rgb_transform = get_transform(opt, grayscale=False)
        self.mask_transform = get_transform(opt, grayscale=True)
        
        self.org_transform = torchvision.transforms.Compose([transforms.ToTensor()])



    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files)

    def __getitem__(self, index_):
        index = index_ % self.mask_size

        base_name = self.files[index]
        rgb_path = os.path.join(self.rgb_root,'{}.jpg'.format(base_name))
        rgb_img = Image.open(rgb_path).convert('RGB')
        rgb = self.rgb_transform(rgb_img)
        rgb_org = self.org_transform(rgb_img)

        mask_info_list = self.image_mask_dict[base_name]

        mask_list = []
        mask_org_list = []
        direction_list = []
        for mask_info in mask_info_list:
            direction_list.append(mask_info[1])
            mask_file_name = base_name + "_" + mask_info[0] + ".jpg"
            mask_path = os.path.join(self.mask_root,'{}'.format(mask_file_name))
        
            mask_img = Image.open(mask_path)
            # mask_img = mask_img.filter(ImageFilter.MaxFilter(13))

            mask_ = self.mask_transform(mask_img)
            mask_org_ = self.org_transform(mask_img)[0:1,:,:]


            # mask_[mask_ < 0.5] = 0
            # mask_[mask_ >= 0.5] = 1

            mask_list.append(mask_)
            mask_org_list.append(mask_org_)

        direction_list = direction_list
        mask = torch.stack(mask_list,dim=0)
        mask_org = torch.stack(mask_org_list,dim=0)
        category = torch.Tensor([-1])
        
        return {'rgb': rgb, 'mask': mask,  'path':rgb_path, 'category':category, 'direction':direction_list, 'rgb_org': rgb_org, 'mask_org': mask_org}
        
