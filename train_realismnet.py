import os 
import numpy as np
from tqdm import tqdm
import random
import cv2

import torch
import torch.nn as nn

from argumentsparser import args

from utils.utils import create_exp_name_disc
from utils.networkutils import init_net
from utils.applyedits import apply_whitebalancing, apply_colorcurve, apply_saturation, apply_exposure, EDITS

from model.pix2pix.models.networks import GANLoss
from model.discriminator import VOTEGAN
from dataloader.cocodataset import COCOdataset



torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])

device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')

def create_fake_human_result(rgb):
    # Randomly choose an edit.
    edited = rgb.clone()
    ne = np.random.randint(2, 4)
    perm = torch.randperm(len(EDITS)-1)
    perm = perm + 1 # not selecting whitebalancing(0)

    for i in range(ne):
        edit_id = perm[i]
        #wb_param = torch.rand(args.batch_size, 3).to(device)*0.9 + 0.1
        colorcurve = torch.rand(args.batch_size, 24).to(device)*1.5 + 0.5
        
        sat_param = torch.rand(args.batch_size, 1)*0.25
        sat_param = torch.where(torch.rand(args.batch_size, 1) > 0.5, 0.5+sat_param, sat_param + 1.25)
        sat_param = sat_param.to(device)

        expos_param = torch.rand(args.batch_size, 1)*0.25
        expos_param = torch.where(torch.rand(args.batch_size, 1) > 0.5, expos_param+0.5, 1.25+expos_param)
        expos_param = expos_param.to(device)

        parameters = {'colorcurve':colorcurve, 'saturation':sat_param, 'exposure':expos_param }
        edited = torch.clamp(EDITS[edit_id.item()](edited,parameters),0,1)
    return edited

def create_real_result(rgb):
    # Randomly choose an edit.
    edited = rgb.clone()
    ne = np.random.randint(1, 3)
    perm = torch.randperm(len(EDITS)-1)
    perm = perm + 1 # not selecting whitebalancing(0)

    for i in range(ne):
        edit_id = perm[i]
        #wb_param = torch.rand(args.batch_size, 3).to(device)*0.9 + 0.1
        colorcurve = torch.rand(args.batch_size, 24).to(device)*0.3 + 0.85 # 0.85 1.15
        sat_param = torch.rand(args.batch_size, 1).to(device)*0.3 + 0.85 # 0.85 - 1.15
        expos_param = torch.rand(args.batch_size, 1).to(device)*0.3 + 0.85 # 0.85 - 1.15

        parameters = {'colorcurve':colorcurve, 'saturation':sat_param, 'exposure':expos_param }
        edited = torch.clamp(EDITS[edit_id.item()](edited,parameters),0,1)
    return edited

def create_fake_result(rgb):
    # Randomly choose an edit.
    edited = rgb.clone()
    ne = np.random.randint(2, 5)
    perm = torch.randperm(len(EDITS))

    # edit_intensity_total = 0
    for i in range(ne):
        edit_id = perm[i]
        wb_param = torch.rand(args.batch_size, 3).to(device)*0.9 + 0.1
        colorcurve = torch.rand(args.batch_size, 24).to(device)*1.5 + 0.5
        
        sat_param = torch.rand(args.batch_size, 1)*0.5
        sat_param = torch.where(torch.rand(args.batch_size, 1) > 0.5, sat_param, sat_param + 1.5)
        sat_param = sat_param.to(device)

        expos_param = torch.rand(args.batch_size, 1)*0.25
        expos_param = torch.where(torch.rand(args.batch_size, 1) > 0.5, expos_param+0.5, 1.5+2*expos_param)
        expos_param = expos_param.to(device)

        parameters = {'whitebalancing':wb_param, 'colorcurve':colorcurve, 'saturation':sat_param, 'exposure':expos_param }
        edited = torch.clamp(EDITS[edit_id.item()](edited,parameters),0,1)
    return edited

if __name__ == '__main__':
    dataset = COCOdataset(args)

    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    num_workers=int(args.num_threads),
    pin_memory=True,
    drop_last=True)

    total_loss = 0  


    exp_name = create_exp_name_disc(args, 'RealismNet')
    if not os.path.exists(os.path.join('./checkpoints', exp_name,'images')):
        os.makedirs(os.path.join('./checkpoints', exp_name, 'images'))

    disc_model = init_net(VOTEGAN(args), args.gpu_ids) 
    criterion = GANLoss('lsgan').to(device)

    optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    iteration = 0
    for epoch in tqdm(range(args.epochs)):  
        for episode,data in enumerate(dataloader):
            rgb = data['rgb'].to(device)
            mask = data['mask'].to(device)
            category = data['category'].to(device)
            ishuman = (category == 1).float()

            real_edited = create_real_result(rgb)
            real_edited = real_edited * mask + rgb * (1-mask)

            fake_edited = create_fake_result(rgb)
            fake_edited = fake_edited * mask + rgb * (1-mask)

            fake_edited_human = create_fake_human_result(rgb)
            fake_edited_human = fake_edited_human * mask + rgb * (1-mask)

            real = torch.cat((real_edited, mask), 1)
            fake = torch.cat((fake_edited, mask), 1)
            fake_human = torch.cat((fake_edited_human, mask), 1)
 
            pred_real = disc_model(real)
            pred_fake = disc_model(fake.detach())
            pred_fake_human = disc_model(fake_human.detach())

            loss_real = torch.mean(criterion(pred_real, True))
            loss_fake = torch.mean(criterion(pred_fake, False))
            loss_fake_human = torch.sum(criterion(pred_fake_human, False) * ishuman) / (torch.sum(ishuman) + 1e-6)

            loss = loss_real + loss_fake + loss_fake_human

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()


            if iteration % args.log_interval == 0:
                total_loss = total_loss / args.log_interval

                rgb_np = rgb[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
                mask_np = mask[0,...].cpu().detach().numpy().squeeze()
                mask_np = np.dstack([mask_np, mask_np, mask_np])
                real_edited_np = real_edited[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
                fake_edited_np = fake_edited[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
                
                result = np.concatenate((rgb_np, real_edited_np, fake_edited_np, mask_np), axis=1)
                result = (result * 255).astype(np.uint8)
                cv2.imwrite(os.path.join('./checkpoints', exp_name, 'images', 'epoch_{}_iter_{}.jpg'.format(epoch, iteration)), result[:, :, ::-1],[int(cv2.IMWRITE_JPEG_QUALITY), 50])

                print('Iteration: {}/{},  Loss:{}'.format(iteration, args.epochs*len(dataloader),total_loss))
                total_loss = 0


            if iteration % args.savemodel_interval == 0:
                model_checkpoint_dir = os.path.join('./checkpoints', exp_name)
                save_filename = '%s_net_D.pth' % (iteration)
                save_path = os.path.join(model_checkpoint_dir, save_filename)
                net = disc_model
                if len(args.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(args.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

            iteration = iteration + 1


