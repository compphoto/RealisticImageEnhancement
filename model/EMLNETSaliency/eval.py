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

import resnet

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('model_path', type=pl.Path,
                    help='the path of the pre-trained model')
parser.add_argument('img_path', type=pl.Path,
                    help='the folder of salicon data')

parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='resize the input image, (640,480) is from the training data, SALICON.')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def normalize(img):
    img -= img.min()
    img /= img.max()

def main():
    global args

    preprocess = transforms.Compose([
        transforms.Resize(args.size),
	transforms.ToTensor(),
    ])

    model = resnet.resnet50(args.model_path).cuda()
    model.eval()

    pil_img = Image.open(args.img_path).convert('RGB')
    processed = preprocess(pil_img).unsqueeze(0).cuda()

    with torch.no_grad():
        pred_batch = model(processed)

    for img, pred in zip(processed, pred_batch):
        fig, ax = plt.subplots(1, 2)

        pred = pred.squeeze()
        normalize(pred)
        pred = pred.detach().cpu()

        img = img.permute(1,2,0).cpu()

        ax[0].imshow(img)
        ax[0].set_title("Input Image")
        ax[1].imshow(pred)
        ax[1].set_title("Prediction")
        plt.show()

if __name__ == '__main__':
    main()
