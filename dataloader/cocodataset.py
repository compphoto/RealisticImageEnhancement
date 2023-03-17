import os

import torch

from utils.datautils import get_transform
from PIL import Image
import csv
import random

class COCOdataset:
    def __init__(self, args, subset = 'train'):
        self.args = args

        self.rgb_root = args.rgb_root
        self.mask_root = args.mask_root

        self.subset = subset

        self.path_rgb = os.path.join(self.rgb_root,subset+'2017','images')
        self.path_mask = os.path.join(self.mask_root,subset,'matte_kszrtio_0.05_crp_1','decrease')

        self.files = [file for file in os.listdir(self.path_mask) if file.endswith('.jpg')]
        if self.subset == 'val':
            random.shuffle(self.files)
            # using 100 images for validation
            self.files = self.files[:100]
        self.files = sorted(self.files)

        self.mask_size = len(self.files)

        annotation_dict = {}
        with open('data.csv', newline='') as csvfile:
            annotationreader = csv.reader(csvfile, delimiter=',')
            for row in annotationreader:
                dataset = row[0]
                image_id = row[1]
                mask_id = row[2]
                category = row[3]
                if dataset == 'decrease':
                    annotation_dict['0'*(12 - len(image_id))+image_id+'_'+mask_id] = category

        self.annotation_dict = annotation_dict

        opt = {}
        opt['load_size'] = args.load_size
        opt['crop_size'] = args.crop_size
        opt['preprocess'] = 'resize'
        opt['no_flip'] = True

        self.rgb_transform = get_transform(opt, grayscale=False)
        self.mask_transform = get_transform(opt, grayscale=True)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files)

    def __getitem__(self, index_):
        index = index_ % self.mask_size

        mask_file_name = self.files[index]
        rgb_file_name = self.files[index].split('_')[0]

        if self.subset == 'train':
            category = int(self.annotation_dict[mask_file_name.replace('.jpg','')])
            category = torch.as_tensor([category])
        else:
            category = torch.as_tensor([-1])

        rgb_path = os.path.join(self.path_rgb,'{}.jpg'.format(rgb_file_name))
        mask_path = os.path.join(self.path_mask,'{}'.format(mask_file_name))
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        mask_img = Image.open(mask_path)
        
        rgb = self.rgb_transform(rgb_img)
        mask = self.mask_transform(mask_img)

        return {'rgb': rgb, 'mask': mask, 'path':mask_path, 'category':category}
        
