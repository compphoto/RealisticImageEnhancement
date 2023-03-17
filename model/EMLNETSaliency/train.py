import argparse
import os
import shutil
import time

import dataloader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import nasnet

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size per process (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--val-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--data_root', type=str,
                    help='the root folder for the input data.')
parser.add_argument('--train_file', type=str,
                    help='the text file that contains the training data, each line represents (img_path gt_path), space separated.')
parser.add_argument('--output', type=str,
                    help='the output folder used to store the trained model.')

cudnn.benchmark = True

best_prec1 = 0
args = parser.parse_args()

IMG_WIDTH = 320
IMG_HEIGHT = 320 

def main():

    def build_data_loader(data_root, data_file, train=False):

        data_loader = torch.utils.data.DataLoader(
            dataloader.ImageList(data_root, data_file, transforms.Compose([
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                transforms.ToTensor(),
            ])),
            batch_size=args.batch_size, shuffle=train,
            num_workers=args.workers, pin_memory=True)
        return data_loader


    model = nasnet.nasnetalarge()
    model.cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.output and not os.path.isdir(args.output):
        os.makedirs(args.output)

    for epoch in range(args.start_epoch, args.epochs):
        train_loader = build_data_loader(args.data_root, args.train_file, train=True)
        train(train_loader, model, criterion, optimizer, epoch)
        if args.output and epoch+1 > 5:
            state = {
                'state_dict' : model.state_dict(),
                }
            path = os.path.join(args.output, "model"+str(epoch)+".pth.tar")
            torch.save(state, path)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    model.train()

    for i, (input, s_map) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        input = input.cuda()
        s_map = s_map.cuda(non_blocking=True)
        output = model(input)
        mse = criterion(output, s_map)
        losses.update(mse.item(), input.size(0))
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})'.format(
                   epoch, i, len(train_loader), loss=losses))

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 3
    lr = args.lr*(0.1**factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

if __name__ == '__main__':
    main()
