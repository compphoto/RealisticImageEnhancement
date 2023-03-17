# This is the code used to train a decoder to combine two pre-treained saliency models.
#
# Author: Sen Jia
# Date: 10 / Mar / 2020
#
import argparse
import os
import pathlib as pl

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import SaliconLoader
import EMLLoss
import resnet
import decoder

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data_folder', type=pl.Path,
                    help='the folder of salicon data')
parser.add_argument('output_folder', type=str,
                    help='the folder used to save the trained model')
parser.add_argument('image_model_path', default=None, type=pl.Path,
                    help='the path of the pre-trained model')
parser.add_argument('place_model_path', default=None, type=pl.Path,
                    help='the path of the pre-trained model')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--decay_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val_epoch', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--val_thread', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--mse', action="store_true",
                    help='apply MSE as a loss function')
parser.add_argument('--gpu', default='0', type=str,
                    help='The index of the gpu you want to use')
parser.add_argument('--size', default=(480, 640), type=tuple,
                    help='resize the input image, (640,480) is from the training data, SALICON.')
parser.add_argument('--num_feat', default=5, type=int,
                    help='the number of features collected from each model')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

def main():
    global args

    img_model = resnet.resnet50(args.image_model_path).cuda().eval()
    pla_model = resnet.resnet50(args.place_model_path).cuda().eval()

    decoder_model = decoder.build_decoder(None, args.size, args.num_feat, args.num_feat).cuda()

    optimizer = torch.optim.SGD(decoder_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        SaliconLoader.ImageList(args.data_folder, transforms.Compose([
            transforms.ToTensor(),
        ]),
        train=True,
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.mse:
        args.output_folder = args.output_folder + "_mse"
        criterion = nn.MSELoss().cuda()
    else:
        args.lr *= 0.1
        args.output_folder = args.output_folder + "_eml"
        criterion = EMLLoss.Loss().cuda()

    args.output_folder = pl.Path(args.output_folder)

    if not args.output_folder.is_dir():
        args.output_folder.mkdir()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_loader, img_model, pla_model, decoder_model, criterion, optimizer, epoch)

    state = {
        'state_dict' : decoder_model.state_dict(),
        }

    save_path = args.output_folder / ("model.pth.tar")
    save_model(state, save_path)

def save_model(state, path):
    torch.save(state, path)

def train(train_loader, img_model, pla_model, decoder_model, criterion, optimizer, epoch):
    losses = AverageMeter()
    decoder_model.train()

    for i, (input, fixmap, smap) in enumerate(train_loader):

        input = input.cuda()
        fixmap = fixmap.cuda()
        smap = smap.cuda()

        with torch.no_grad(): 
            img_feat = img_model(input, decode=True)
            pla_feat = pla_model(input, decode=True)

        decoded = decoder_model([img_feat, pla_feat])

        if args.mse: 
            loss = criterion(decoded, smap)
        else:
            loss = criterion(decoded, fixmap, smap)

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader),
                   loss=losses))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // args.decay_epoch
    lr = args.lr*(0.1**factor)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
