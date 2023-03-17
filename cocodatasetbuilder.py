from pycocotools.coco import COCO
import numpy as np
import os
import os
import cv2
from tqdm import tqdm

from model.saliencymodel import EMLNET
import torch
import json,csv
import argparse

np.random.seed(0)

parser = argparse.ArgumentParser(description='Sal Based Image Enhancement Options')
parser.add_argument('--coco_root', type=str,required=True)
parser.add_argument('--result_root', type=str,required=True)
parser.add_argument('--mode', type=str,required=True,choices=['train','val'])
parser.add_argument('--gpu_ids', default=0 ,required=True) # use -1 for CPU


args = parser.parse_args()

dataDir = args.coco_root
mode = 'train'

str_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
    torch.cuda.set_device(args.gpu_ids[0])
device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')



net_Saliency = EMLNET(args=None)
net_Saliency.to(device)



annFile ='{}/annotations/instances_{}2017.json'.format(dataDir,mode)

result_dir = os.path.join(args.result_root,mode,'mask')
os.makedirs(os.path.join(result_dir,'increase'),exist_ok=True)
os.makedirs(os.path.join(result_dir,'decrease'),exist_ok=True)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms_sc = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms_sc)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=nms)
nb_cats = max(catIds)
# map_cats = (((np.array(catIds)+100)/nb_cats) *255).astype(np.int).tolist()
map_cats = (np.array(catIds) + (255 - nb_cats)).tolist()

segDict = dict(zip(catIds, map_cats))
imgIds=[]
for i in catIds:
    imgIds += coco.getImgIds(catIds=i)
imgIds = list(dict.fromkeys(imgIds))
nb_images = len(imgIds)
counter = 0

# while counter < nb_images:
main_counter = 0
with open('data_val.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    for counter in tqdm(range(0,nb_images), desc='image'):
        imId = imgIds[counter]
        img = coco.loadImgs(imId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        rgb = cv2.imread(os.path.join(dataDir, '{}2017'.format(mode),'images', img['file_name']))[:,:,::-1]
        name = str(anns[0]['image_id'])
        h,w = rgb.shape[:2]
        if (rgb[:,:,0] == rgb[:,:,1]).all():
            continue
        if len(anns) < 5:
            continue
        
        rgb_ = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        rgb_ = torch.from_numpy(rgb_).permute(2,0,1).float().unsqueeze(0).to(device)
        with torch.no_grad():
            salmap = net_Saliency(rgb_)
        salmap = salmap.squeeze().cpu().numpy()
        salmap = (salmap - np.min(salmap)) / (np.max(salmap) - np.min(salmap))
        salmap = cv2.resize(salmap, (w,h), interpolation=cv2.INTER_NEAREST)

        valid_mask_list = [] 
        valid_ann_list = []
        mask_salmean_list = []
        mask_numel_list = []

        for i in range(len(anns)):
            mask = coco.annToMask(anns[i])
            ratio = mask.sum()/np.size(mask)
            if (ratio > 0.4):
                continue
            if (ratio < 0.01):
                continue
            valid_mask_list.append(mask)
            valid_ann_list.append(anns[i])

        
        if len(valid_mask_list) < 2:
            continue
        
        for mask in valid_mask_list:
            mask_sal_mean = np.sum(salmap * mask) / np.sum(mask)
            mask_salmean_list.append(mask_sal_mean)
            mask_numel_list.append(np.sum(mask))

        sort_sal_args = np.argsort(mask_salmean_list)[::-1]
        n = len(valid_mask_list)
        m = int(np.floor(n * 1/2))
        mask_name_counter_decrease = 0
        mask_name_counter_increase = 0
        for i in range(m):
            mask_name = '0'*(12 - len(name)) + name + '_%d'%mask_name_counter_decrease + '.jpg'
            cv2.imwrite(os.path.join(result_dir,'decrease', mask_name),(valid_mask_list[sort_sal_args[i]] * 255).astype('uint8'))
            data = ['decrease',name, str(mask_name_counter_decrease),str(valid_ann_list[sort_sal_args[i]]['category_id'])]
            writer.writerow(data)
            mask_name_counter_decrease = mask_name_counter_decrease + 1
        for i in range(m,n):
            mask_name = '0'*(12 - len(name)) + name + '_%d'%mask_name_counter_increase  +'.jpg'
            cv2.imwrite(os.path.join(result_dir,'increase', mask_name),(valid_mask_list[sort_sal_args[i]]*255).astype('uint8'))
            data = ['increase',name, str(mask_name_counter_increase),str(valid_ann_list[sort_sal_args[i]]['category_id'])]
            writer.writerow(data)
            mask_name_counter_increase = mask_name_counter_increase + 1

        csv_file.flush() 
        main_counter = main_counter + 1 
print("total number of generated images: ",main_counter) 