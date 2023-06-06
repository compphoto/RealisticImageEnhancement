## Realistic Saliency Guided Image Enhancement [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/https://github.com/compphoto/RealisticImageEnhancement/blob/main/demo.ipynb) [![Arxiv](http://img.shields.io/badge/cs.CV-arXiv-B31B1B.svg)]()


> S. Mahdi H. Miangoleh, Zoya Bylinskii, Eric kee, Eli Shechtman, Yağız Aksoy.
> [Main pdf](http://yaksoy.github.io/papers/CVPR23-RealisticEditing.pdf),
> [Supplementary pdf](http://yaksoy.github.io/papers/CVPR23-RealisticEditing-Supp.pdf),
> [Project Page](http://yaksoy.github.io/realisticEditing/). 

Proc. CVPR, 2023

[![video](figures/gitplay.jpg)](https://www.youtube.com/watch?v=5dKUDMnnjuo)



We train and expliot a problem specific realism network (RealismNet) to train a saliency-guided image enhancement network (EditNet) which allows maintaining high realism across varying image types while attenuating distractors and amplifying objects of interest. Ours model achieves both realism and effectiveness, outperforming recent approaches on their own datasets, while also requiring a smaller memory footprint and runtime. **Our proposed approach offers a viable solution for automating image enhancement and photo cleanup operations**.


Try our model easily on Colab : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/https://github.com/compphoto/RealisticImageEnhancement/blob/main/demo.ipynb)



# Inference

1. Create a python conda environment as following:

```
conda create -n ENVNAME python=3.8
pip install -r requirements.txt

conda activate ENVNAME
```

2. Download our model weights from [here](https://drive.google.com/file/d/1NUN9xmD3p8G7n-HpD03UY9LHEF6J82-Q/view?usp=drive_link) and place them inside "./bestmodels/" folder.

3. Set the path to your input image and mask pairs. 
```
rgb_root=""
mask_root=""
```

-> input mask and rgb files should have **matching name**. 

```
result_path="./result/mydataset"
```

* Attenuate the saliency of the masked region
```
python test.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_attenuate.pth" --result_for_decrease 1 --batch_size 1
```
* Amplify the saliency of the masked region
```
python test.py --mask_root "$mask_root" --rgb_root "$rgb_root" --result_path "$result_path" --init_parameternet_weights "bestmodels/editnet_amplify.pth" --result_for_decrease 0 --batch_size 1
```

# AdobeStock Images

The Sample images we used in our user study are picked from [AdobeStock](https://stock.adobe.com) dataset. We cannot share the images directly due to licence restrictions but we are providing the original AdobeStock link to these images [here](./scripts/adobestockdatasetlink.md) to ease future comparisons.  

# Training RealismNet and EditingNet

## Building the Datasets
The datasets needed to train our RealismNet and EditingNet can be generated by running the `builddataset.sh` script. 

**Before running the script**, fill in the needed variables as explained in the following.

### Building Object Mask DATASET from COCO

To build the dataset, follow these steps:

Download MSCOCO dataset from [here](https://cocodataset.org/#home).

1. Set the path to the COCO dataset root files:
```
COCOROOT=""
```

File structure should be as follows:

```
-COCOROOT
   -annotations
   -train2017/images/*.jpg
   -val2017/images/*.jpg
```

2. Indicate a path to save the created dataset:
```
RESULTROOT=""
```

3. Run the `cocodatasetbuilder.py` script to create the training dataset:
```
python cocodatasetbuilder.py --coco_root "$COCOROOT" --result_root "$RESULTROOT" --mode train --gpu_ids 0
```

4. Run the `cocodatasetbuilder.py` script to create the validation dataset:
```
python cocodatasetbuilder.py --coco_root "$COCOROOT" --result_root "$RESULTROOT" --mode val --gpu_ids 0
```

Results will be saved under "$RESULTROOT/[train|val]/mask".

### Computing Alpha Mattes

To compute the alpha mattes we use [FBAMatting]((https://github.com/MarcoForte/FBA_Matting)). follow these steps:

1. Download the model weight from [here](https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1) and place it in `model/FBA_Matting/` .

2. Navigate to the `model/FBA_Matting/` directory:
```
cd model/FBA_Matting/
```

3. Run the `mattmodel.py` script to compute the alpha mattes for the training set:
```
python mattmodel.py --subset train --rgb_root "$COCOROOT" --mask_root "$RESULTROOT"
```

4. Run the `mattmodel.py` script to compute the alpha mattes for the validation set:
```
python mattmodel.py --subset val --rgb_root "$COCOROOT" --mask_root "$RESULTROOT"
```

Results will be saved under "$RESULTROOT/[train|val]/matte".

<br/><br/>

## Training RealismNet

Set the path to the datasets. MASKROOT is path to generated dataset above. Should be set the same as "RESULTROOT". 
```
COCOROOT=""
MASKROOT=""
```

Run the following to train the network. We used a batch size of 128 for training. Adjust according to your GPU memory. 
```
python train_realismnet.py --batch_size 128 --lr_d 0.0001 --epochs 100 --mask_root "$MASKROOT" --rgb_root "$COCOROOT"
```

<br/><br/>

## Training EditingNet

We train two sepparate models for each saliency *Attenuation* and *Amplification* tasks. Run the following scripts to train each model. 

* Train attenuation model
```
python train_editnet.py --gpu_ids 0 --epochs 10 --batch_size 8 --lr_parameters 0.00001 --beta_r 0.1 --w_sal 1 --human_weight_gan 10 --sal_loss_type percentage --mask_root "$MASKROOT" --rgb_root "$COCOROOT"
```                   

* Train amplification model

```
python train_editnet.py --gpu_ids 0 --epochs 10 --batch_size 8 --lr_parameters 0.00001 --beta_r 0.1 --w_sal 5 --human_weight_gan 10 --sal_loss_type percentage_increase --mask_root "$MASKROOT" --rgb_root "$COCOROOT"
```

## Running on Video 

Download the DAVIS dataset from [their webpage](https://davischallenge.org/). Helper scripts to run our method on videos can be found [here](./scripts/test_video.sh).


## Citation

This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models.
```
@INPROCEEDINGS{Miangoleh2023RealisticImageEnhancement,
author={S. Mahdi H. Miangoleh and Zoya Bylinskii and Eric Kee and Eli Shechtman and Ya\u{g}{\i}z Aksoy},
title={Realistic Saliency Guided Image Enhancement},
journal={Proc. CVPR},
year={2023},
}
```

## Credits

The EML-Net implementation was adapted from [EML-NET-Saliency](https://github.com/SenJia/EML-NET-Saliency) repository.

"./model/pix2pix" folder was adapted from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository. 

"./model FBA_Matting" folder was adapted from authors [repository](https://github.com/MarcoForte/FBA_Matting).

"./model/MiDaS" is adapted from [MiDas](https://github.com/intel-isl/MiDaS/tree/v2) for their EfficientNet implementation.   

