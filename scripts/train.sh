#!/bin/bash


cd ..
## Training RealismNet

# Path to the dataset
COCOROOT=""
MASKROOT=""

python train_realismnet.py --batch_size 8 --lr_d 0.0001 --epochs 100 --mask_root "$MASKROOT" --rgb_root "$COCOROOT"

## Train Editnet

# Train decrease model
python train_editnet.py --gpu_ids 0 --epochs 10 --batch_size 8 --lr_parameters 0.00001 \
                    --beta_r 0.1 --w_sal 1 --human_weight_gan 10 --sal_loss_type percentage \
                    --mask_root "$MASKROOT" --rgb_root "$COCOROOT"
                     

# train increase model
python train_editnet.py --gpu_ids 0 --epochs 10 --batch_size 8 --lr_parameters 0.00001 \
                    --beta_r 0.1 --w_sal 5 --human_weight_gan 10 --sal_loss_type percentage_increase \
                    --mask_root "$MASKROOT" --rgb_root "$COCOROOT"

