import torch
import torchvision.transforms as transforms
from  torchvision.transforms.functional import resize as Tresize

from utils.datautils import get_transform
from PIL import Image
import os
import numpy as np
import cv2
import random

class VideoDataset:
    def __init__(self, args):
        self.args = args

        self.rgb_root = args.rgb_root
        self.mask_root = args.mask_root

        self.files = sorted([file for file in os.listdir(self.mask_root) if file.endswith('.png')])
        
        self.mask_size = len(self.files)

        opt = {}
        opt['load_size'] = args.load_size
        opt['crop_size'] = args.crop_size
        opt['preprocess'] = 'resize'
        opt['no_flip'] = True

        self.rgb_transform = get_transform(opt, grayscale=False)
        self.mask_transform = get_transform(opt, grayscale=True)
        self.org_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.files)

    def __getitem__(self, index_):
        index = index_ % self.mask_size

        mask_file_name = self.files[index]
        rgb_file_name = mask_file_name.replace('.png','.jpg') 

        rgb_path = os.path.join(self.rgb_root,'{}'.format(rgb_file_name))
        mask_path = os.path.join(self.mask_root,'{}'.format(mask_file_name))
        
        rgb_img = Image.open(rgb_path).convert('RGB')
        mask_img = Image.open(mask_path)

        mask_np = np.array(mask_img)

        mask_selected = np.zeros_like(mask_np)
        count_obj = len(np.unique(np.array(mask_img))) - 1 # -1 for background
        if count_obj > 0:
            selected_mask_ind = 1
            mask_selected[mask_np == selected_mask_ind[0]] = 1
        else:
            mask_selected[mask_np != 0] = 1

        rgb = self.rgb_transform(rgb_img)

        rgb_org = self.org_transform(rgb_img)
        mask_org = torch.from_numpy(mask_selected).float().unsqueeze(0)
        mask = cv2.resize(mask_selected, (self.args.crop_size, self.args.crop_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        category = torch.Tensor([-1])
        
        return {'rgb': rgb, 'mask': mask,  'path':mask_path, 'category':category, 'rgb_org':rgb_org, 'mask_org':mask_org}
        
