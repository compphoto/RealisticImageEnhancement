#!/bin/bash


cd ..
## Building DATASET

# Path to COCO dataset Root files. File structure should be as follows:
# -COCOROOT
#   -annotations
#   -train2017/images/*.jpg
#   -val2017/images/*.jpg
COCOROOT=""

# Indicate a path to save the created dataset
RESULTROOT=""

python cocodatasetbuilder.py --coco_root "$COCOROOT" --result_root "$RESULTROOT" --mode train --gpu_ids 0
python cocodatasetbuilder.py --coco_root "$COCOROOT" --result_root "$RESULTROOT" --mode val --gpu_ids 0


## Computing Alpha Matts

## Download the model weigths from https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1 and place it in "model/FBA_Matting/" (https://github.com/MarcoForte/FBA_Matting)
cd model/FBA_Matting/
python mattmodel.py --subset train --rgb_root "$COCOROOT" --mask_root "$RESULTROOT"
python mattmodel.py --subset val --rgb_root "$COCOROOT" --mask_root "$RESULTROOT"
