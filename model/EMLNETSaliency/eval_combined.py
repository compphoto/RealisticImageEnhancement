# This is the evaluation code to output prediction using our saliency model.
#
# Author: Sen Jia
# Date: 09 / Mar / 2020
#
import argparse
import os
import pathlib as pl

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

import numpy as np
from skimage import filters
import skimage.io as sio

import resnet
import decoder
import SaliconLoader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('image_model_path', type=pl.Path,
                    help='the path of the pre-trained model based on ImageNet')
parser.add_argument('place_model_path', type=pl.Path,
                    help='the path of the pre-trained model based on PLACE')
parser.add_argument('decoder_model_path', type=pl.Path,
                    help='the path of the pre-trained decoder model')

parser.add_argument('img_path', type=pl.Path,
                    help='the folder of salicon data')

parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='resize the input image, (640,480) is from the training data, SALICON.')
parser.add_argument('--num_feat', default=5, type=int,
                    help='the number of features collected from each model')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def normalize(x):
    x -= x.min()
    x /= x.max()

def post_process(pred):
    pred = filters.gaussian(pred, 5)
    normalize(pred)
    pred = (pred * 255).astype(np.uint8)
    return pred


def main():
    global args

    preprocess = transforms.Compose([
        transforms.Resize(args.size),
	transforms.ToTensor(),
    ])

    img_model = resnet.resnet50(args.image_model_path).cuda().eval()
    pla_model = resnet.resnet50(args.place_model_path).cuda().eval()
    decoder_model = decoder.build_decoder(args.decoder_model_path, args.size, args.num_feat, args.num_feat).cuda().eval()

    pil_img = Image.open(args.img_path).convert('RGB')
    processed = preprocess(pil_img).unsqueeze(0).cuda()

    with torch.no_grad():

        img_feat = img_model(processed, decode=True)
        pla_feat = pla_model(processed, decode=True)

        pred = decoder_model([img_feat, pla_feat])

    fig, ax = plt.subplots(1, 2)

    pred = pred.squeeze().detach().cpu().numpy()
    pred = post_process(pred)

    pred_path = args.img_path.stem + "_smap.png"
    print ("Saving prediction", pred_path)
    sio.imsave(pred_path, pred)

    processed = processed.squeeze().permute(1,2,0).cpu()

    ax[0].imshow(processed)
    ax[0].set_title("Input Image")
    ax[1].imshow(pred)
    ax[1].set_title("Prediction")
    plt.show()

if __name__ == '__main__':
    main()
