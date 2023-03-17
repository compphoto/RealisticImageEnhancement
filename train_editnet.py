import torch
import os 
from tqdm import tqdm
import cv2
import numpy as np
from argumentsparser import args
import random

from model.editnettrainer import EditNetTrainer
from dataloader.cocodataset import COCOdataset
from utils.utils import create_exp_name

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

if __name__ == '__main__':
    
    dataset_train = COCOdataset(args)
    dataset_val = COCOdataset(args, subset='val')

    dataloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    num_workers=int(args.num_threads),
    pin_memory=True,
    drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=int(args.num_threads),
    pin_memory=True,
    drop_last=True)
    

    exp_name = create_exp_name(args, 'EditNet')
    if not os.path.exists(os.path.join('./checkpoints', exp_name,'images')):
        os.makedirs(os.path.join('./checkpoints', exp_name, 'images'))

    trainer = EditNetTrainer(args)
    
    iteration = 0
    realism_change = 0
    saliency_change = 0
    loss_realism = 0
    loss_saliency = 0
    loss_g = 0

    trainer.setTrain()

    for epoch in tqdm(range(args.epochs)):  
        for episode,data in enumerate(dataloader_train):
            trainer.setinput(data)
            trainer.forward()
            trainer.optimize_parameters()

            realism_change += torch.mean(trainer.realism_change).item()
            saliency_change += torch.mean(trainer.saliency_change).item()
            loss_realism += torch.mean(trainer.loss_realism).item()
            loss_saliency += torch.mean(trainer.loss_saliency).item()
            loss_g += trainer.loss_g.item()


            if iteration % args.log_interval == 0:
                realism_change = realism_change / args.log_interval
                saliency_change = saliency_change / args.log_interval
                loss_realism = loss_realism / args.log_interval
                loss_saliency = loss_saliency / args.log_interval
                loss_g = loss_g / args.log_interval

                rgb_in = trainer.rgb[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
                rgb_out = trainer.result[0,...].cpu().detach().numpy().squeeze().transpose([1,2,0])
                mask = trainer.mask[0,...].cpu().detach().numpy().squeeze()
                mask = np.dstack([mask, mask, mask])
                sal_in = trainer.input_saliency[0,...].cpu().detach().numpy().transpose([1,2,0])
                sal_out = trainer.output_saliency[0,...].cpu().detach().numpy().transpose([1,2,0])
                sal_diff_ = (sal_in - sal_out).squeeze()

                sal_in = np.dstack([sal_in, sal_in, sal_in])
                sal_out = np.dstack([sal_out, sal_out, sal_out])
                sal_diff = np.zeros_like(sal_in)
                sal_diff_ = (sal_diff_ - sal_diff_.min()) / (sal_diff_.max() - sal_diff_.min())
                sal_diff[:,:,2] = sal_diff_
                sal_diff[:,:,0] = -sal_diff_
                sal_diff[:,:,1] = 0.2*mask[:,:,0]

                result = np.concatenate((rgb_in, rgb_out, mask, sal_in, sal_out, sal_diff), axis=1)
                result = (result * 255).astype(np.uint8)
                # log the image to image file named using iteration number and epoch
                cv2.imwrite(os.path.join('./checkpoints', exp_name, 'images', 'epoch_{}_iter_{}.png'.format(epoch, iteration)), result)

                print({     'realism_change': realism_change,
                            'saliency_change': saliency_change,
                            'loss_realism': loss_realism,
                            'loss_saliency': loss_saliency,
                            'loss_g': loss_g})
                # print(trainer.logs)

                realism_change = 0
                saliency_change = 0
                loss_realism = 0
                loss_saliency = 0
                loss_g = 0

            trainer.logs = []
            if iteration % args.val_interval == 0:
                trainer.setEval()

                val_realism_change = 0
                val_saliency_change = 0
                val_loss_realism = 0
                val_loss_saliency = 0
                val_loss_g = 0

                for episode_val,data_val in enumerate(dataloader_val):
                    with torch.no_grad():
                        trainer.setinput(data_val)
                        trainer.forward()
                        trainer.compute_gloss()

                        val_realism_change += torch.mean(trainer.realism_change).item()
                        val_saliency_change += torch.mean(trainer.saliency_change).item()
                        val_loss_realism += torch.mean(trainer.loss_realism).item()
                        val_loss_saliency += torch.mean(trainer.loss_saliency).item()
                        val_loss_g += trainer.loss_g.item()

                val_realism_change = val_realism_change / len(dataloader_val)
                val_saliency_change = val_saliency_change / len(dataloader_val)
                
                val_loss_realism = val_loss_realism / len(dataloader_val)
                val_loss_saliency = val_loss_saliency / len(dataloader_val)
                val_loss_g = val_loss_g / len(dataloader_val)
                
                print({'val_realism_change': val_realism_change,
                            'val_saliency_change': val_saliency_change,
                            'val_loss_realism': val_loss_realism,
                            'val_loss_saliency': val_loss_saliency,
                            'val_loss_g': val_loss_g})  
                            
                trainer.setTrain()


            if iteration % args.savemodel_interval == 0:
                model_checkpoint_dir = os.path.join('./checkpoints', exp_name)
                if not os.path.exists(model_checkpoint_dir):
                    os.makedirs(model_checkpoint_dir)
                trainer.savemodel(iteration,checkpointdir=model_checkpoint_dir)

            iteration = iteration + 1