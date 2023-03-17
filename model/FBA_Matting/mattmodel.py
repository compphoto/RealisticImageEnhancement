from http.client import ImproperConnectionState
from pydoc import describe
from termios import CINTR
from turtle import position
import torch
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import argparse

from networks.transforms import trimap_transform, normalise_image
from networks.models import build_model

def scale_input(x: np.ndarray, scale: float, scale_type) -> np.ndarray:
    ''' Scales inputs to multiple of 8. '''
    h, w = x.shape[:2]
    h1 = int(np.ceil(scale * h / 8) * 8)
    w1 = int(np.ceil(scale * w / 8) * 8)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale

def np_to_torch(x, permute=True):
    if permute:
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
    else:
        return torch.from_numpy(x)[None, :, :, :].float().cuda()

def read_image(name):
    return (cv2.imread(name) / 255.0)[:, :, ::-1]


def conv_trimap_2_twochannels(trimap_im):
    h, w = trimap_im.shape
    trimap = np.zeros((h, w, 2))
    trimap[trimap_im == 1, 1] = 1
    trimap[trimap_im == 0, 0] = 1
    return trimap

class MattModel:
    def __init__(self,opt):
            self.opt = opt
            self.iterations = 20
            
            self.model = build_model('FBA.pth')
            self.model.eval().cuda()

    def inference(self, image_np: np.ndarray, trimap_np: np.ndarray) -> np.ndarray:
        ''' Predict alpha, foreground and background.
            Parameters:
            image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
            trimap_np -- two channel trimap, first background then foreground. Dimensions: (h, w, 2)
            Returns:
            fg: foreground image in rgb format between 0 and 1. Dimensions: (h, w, 3)
            bg: background image in rgb format between 0 and 1. Dimensions: (h, w, 3)
            alpha: alpha matte image between 0 and 1. Dimensions: (h, w)
        '''
        h, w = trimap_np.shape[:2]
        image_scale_np = scale_input(image_np, 1, cv2.INTER_LANCZOS4)
        trimap_scale_np = scale_input(trimap_np, 1, cv2.INTER_LANCZOS4)

        with torch.no_grad():
            image_torch = np_to_torch(image_scale_np)
            trimap_torch = np_to_torch(trimap_scale_np)

            trimap_transformed_torch = np_to_torch(
                trimap_transform(trimap_scale_np), permute=False)
            image_transformed_torch = normalise_image(
                image_torch.clone())

            output = self.model(
                image_torch,
                trimap_torch,
                image_transformed_torch,
                trimap_transformed_torch)
            output = cv2.resize(
                output[0].cpu().numpy().transpose(
                    (1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

        alpha = output[:, :, 0]
        fg = output[:, :, 1:4]
        bg = output[:, :, 4:7]

        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1
        fg[alpha == 1] = image_np[alpha == 1]
        bg[alpha == 0] = image_np[alpha == 0]

        return fg, bg, alpha

    def readinputs(self, im_path, mask_path):
        rgb = read_image(im_path).astype('float32')
        mask = cv2.imread(mask_path,0)
        # rgb = cv2.ximgproc.guidedFilter(rgb, rgb, 3, 0.01)
        # rgb[rgb>1] = 1
        # rgb[rgb<0] = 0
        # compute k_size 
        mask_pixels_count = np.sum(mask != 0)
        k_size = max(int(np.sqrt(mask_pixels_count) * self.opt.k_size_ratio),3)
        trimap = self.mask2trimap(mask, k_size)

        return rgb,mask,trimap

    def mask2trimap(self, mask, k_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        dilated = cv2.dilate(mask, kernel, self.iterations)
        eroded = cv2.erode(mask, kernel, self.iterations)
        h,w = mask.shape
        trimap = np.zeros((h,w))
        trimap.fill(128)
        trimap[eroded > 128] = 255
        trimap[dilated < 128] = 0
        return trimap

def alphatobox(alpha):
    validmap = alpha != 0 
    positions = np.where(validmap == True)
    x_min = positions[0].min()
    x_max = positions[0].max()
    y_min = positions[1].min()
    y_max = positions[1].max()
    return [x_min, y_min, x_max, y_max]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset',choices=['train','val'], default='train')
    parser.add_argument('--mode',choices=['decrease','increase'], default='decrease')
    parser.add_argument('--rgb_root', type=str,required=True)
    parser.add_argument('--mask_root', type=str,required=True)
    parser.add_argument('--k_size_ratio',type=float, default=0.05)
    parser.add_argument('--do_crop',type=int, default=1)

    args = parser.parse_args()

    maskdir = os.path.join(args.mask_root,args.subset,'mask',args.mode)
    rgb_dir = os.path.join(args.rgb_root,'{}2017'.format(args.subset),'images')
    result_path = os.path.join(args.mask_root,args.subset,'matte',args.mode)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    matmodel = MattModel(args) 

    filepaths = [f for f in os.listdir(maskdir) if
                os.path.isfile(os.path.join(maskdir, f)) and f.endswith('.jpg')]
    filepaths = sorted(filepaths)
    
    print(args)
    for filename_ in tqdm(filepaths,desc = 'Processing Images'):
        filename = filename_.split('_')[0]
        rgb_path = os.path.join(rgb_dir,filename+'.jpg')
        mask_path = os.path.join(maskdir,filename_)


        rgb, mask, trimap = matmodel.readinputs(rgb_path,mask_path)
        if args.do_crop:
            boundingbox = alphatobox(trimap) # [x0,y0,x1,y1]

            w,h = trimap.shape[0:2]
            #incease box size :
            b_w = boundingbox[2] - boundingbox[0]
            b_h = boundingbox[3] - boundingbox[1]

            b_w = int(b_w * 0.125)
            b_h = int(b_h * 0.125)

            new_x_min = min(max(boundingbox[0] - b_w,0),w)
            new_x_max = min(max(boundingbox[2] + b_w,0),w)
            new_y_min = min(max(boundingbox[1] - b_h,0),h)
            new_y_max = min(max(boundingbox[3] + b_h,0),h)

            boundingbox = [new_x_min, new_y_min, new_x_max, new_y_max]

            rgb_cropped = rgb[boundingbox[0]:boundingbox[2]+1, boundingbox[1]:boundingbox[3]+1]  
            mask_cropped = mask[boundingbox[0]:boundingbox[2]+1, boundingbox[1]:boundingbox[3]+1]  
            trimap_cropped = trimap[boundingbox[0]:boundingbox[2]+1, boundingbox[1]:boundingbox[3]+1]           
        else:
            rgb_cropped = rgb
            trimap_cropped = trimap
            mask_cropped = mask

        alphamatt_cropped = matmodel.inference(rgb_cropped,conv_trimap_2_twochannels(trimap_cropped / 255.0))[2]
        
        if args.do_crop:
            alphamatt = np.zeros_like(trimap)
            alphamatt[boundingbox[0]:boundingbox[2]+1, boundingbox[1]:boundingbox[3]+1] = alphamatt_cropped  
            alphamatt = alphamatt * 255.
        else:
            alphamatt = alphamatt_cropped * 255. 
        cv2.imwrite(os.path.join(result_path,filename_),(alphamatt).astype('uint8'))
