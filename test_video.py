import torch
import os 
import cv2
import numpy as np
from argumentsparser import args
import random


from model.editnettrainer import EditNetTrainer
from dataloader.videodataset import VideoDataset

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids

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
    dataset_val = VideoDataset(args)

    dataloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True)
    
    direction_str = 'attenuation' if args.result_for_decrease else 'amplification'
    result_root = os.path.join(args.result_path, direction_str)
    os.makedirs(result_root, exist_ok=True)
    
    trainer = EditNetTrainer(args)

    pick_strategy_list = ['first']
    for pick_strategy in pick_strategy_list:
        os.makedirs(os.path.join(result_root, 'picked_{}'.format(pick_strategy)), exist_ok=True)

    trainer.setEval()

    video_param = None
    for frame,data in enumerate(dataloader_val):
        mask_path = data['path'][0]
        image_name =  mask_path.split('/')[-1].split('.')[0]+'.jpg'

        print('({}/{})'.format(frame+1, len(dataloader_val)), '----->', image_name)
        
        trainer.setinput_hr(data)

        sal_list = []
        realism_list = []
        result_list = []
        param_list = []
        with torch.inference_mode():
            for result in trainer.forward_allperm_hr(video_param):
                sal_list.append(result[2])
                realism_list.append(result[1])
                edited = (result[6][0,].transpose(1,2,0)[:,:,::-1] * 255).astype('uint8')
                result_list.append(edited.copy())
                param_list.append(result[9])

        if video_param is None:
            video_param = param_list[0]
            print('Video param selected as params from the first frame')
            

        sal_list = [np.asscalar(item) for item in sal_list]
        realism_list = [np.asscalar(item) for item in realism_list]

        # Do the pick
        picked_ind = 0
        pick_strategy = 'first'
        # save picked result
        picked = result_list[picked_ind]
        picked_name = os.path.join('picked_{}'.format(pick_strategy),image_name) 
        cv2.imwrite(os.path.join(result_root, picked_name), picked)

    
                




