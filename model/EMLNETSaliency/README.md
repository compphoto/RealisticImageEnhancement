# EML-NET-Saliency
This repo contains the code and the pre-computed saliency maps used in our paper: "EML-NET: An Expandable Multi-Layer NETwork for saliency prediction". It has shown that visual saliency relies on objectness within an image, but this may also limit the performance when there is no (known) objects. Our work attempts to broaden the horizon of a saliency model by introducing more types prior knowledge in an efficient way, deeper model architectures, e.g., NasNet can be applied in an "almost" end-to-end fashion. You can also try our modified combined loss funciton as a plug-in to see how it works in your saliency system. 

<img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/EMLNET.jpg" width="80%">

| GroundTruth  | Combined | ImageNet | PLACE |
| ------------- | ------------- | ----------| ----------|
| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/gt_COCO_val2014_000000198590.png" width="300px" height="200px"> |  <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/combined_COCO_val2014_000000198590.png" width="300px" height="200px">| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/imagenet_COCO_val2014_000000198590.png" width="300px" height="200px">| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/place_COCO_val2014_000000198590.png" width="300px" height="200px">
| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/gt_COCO_val2014_000000203754.png" width="300px" height="200px"> |  <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/combined_COCO_val2014_000000203754.png" width="300px" height="200px">| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/imagenet_COCO_val2014_000000203754.png" width="300px" height="200px">| <img src="https://github.com/SenJia/EML-NET-Saliency/blob/master/examples/place_COCO_val2014_000000203754.png" width="300px" height="200px">

## Train a model
Our training code is based on the [SALICON](http://salicon.net/challenge-2017/) dataset, we assume you already download and unzip the images and annotations under your workspace.
```
salicon
└───Images
│     │   *.jpg
|     |
└───fixations
|     |  *.mat
|     |
└───maps
      │   *.png
```

Our training code "train_resnet.py" taks two compulsory arguements, 1. "data_folder"(the path of your workspace). 2 "output_folder"(the folder you want to save the trained model). One more optional arguement you might want to set is "--model_path", pre-trained on ImageNet or PLACE365 for classification, it will train from scratch if not specified.

```
python train_resnet.py ~/salicon imagenet_resnet --model_path backbone/resnet50.pth.tar
```
A suffix of "_eml" will be added to the output path, e.g., imagenet_resnet_eml in this case. If you specify the loss flag, --mse, the added suffix will be "_mse". You can simply compare our proposed loss function against the standard mean squared error.

The ImageNet pre-trained model can be obtained from torchvision, the PLACE pre-trained one can be downloaded from their official project [here](https://github.com/CSAILVision/places365). If you want to try a deeper CNN model, e.g., the NasNet used in our paper, you can download the backbone from this [project](https://github.com/Cadene/pretrained-models.pytorch). We would like to thank the authors and coders of: the Pytorch framework, the PLACE dataset and Remi Cadene for the pre-trained models. 

After finetuning a backbone(resnet50 from ImageNet) on the SALICON dataset, we can combine multiple saliency models(ImageNet and PLACE) by training a decoder. In this case, we need two more compulsory arguments are needed, the model paths for imagenet and place. (You can change the code slightly to combine more for a wider horizon.)
```
python train_decoder.py ~/salicon imagenet_resnet pretrained_sal/imagenet_sal.pth.tar pretrained_sal/place_sal.pth.tar
```
TODO: the nasnet training code will be merged into this training file and the dataloder will be discarded.

## Make a prediction
Update: You can download our pre-trained models based on the resnet backbone, a) pre-trained on imagenet [google drive](https://drive.google.com/open?id=1-a494canr9qWKLdm-DUDMgbGwtlAJz71); b) pre-trained on place365 [google drive](https://drive.google.com/open?id=18nRz0JSRICLqnLQtAvq01azZAsH0SEzS); c) the pre-trained decoder [google drive](https://drive.google.com/open?id=1vwrkz3eX-AMtXQE08oivGMwS4lKB74sH) to combine a) and b). Please save the models under folder "backbone/".

Then you can run our code to load the pre-trained models and make a prediction on your images. The evaluation file "eval_combined.py" requires four arguments: 1. the model path for the imagenet model. 2. the model path for the place model. 3. the model path for the decoder. 4. the path of your input image. It will save the prediction under current folder, you might want to change the path.

```
python eval_combined.py backbone/res_imagenet.pth backbone/res_places.pth backbone/res_decoder.pth examples/115.jpg
```
Download the pre-trained model first and save it somewhere, e.g., pretrained_sal.

<!-- TODO: upload the pre-trained model. -->

Pre-computed maps on the SALICON-val set: [ResNet50](https://drive.google.com/file/d/1gxm8MiNnw7_jPY4hS3hrg4IbOvFwiYHZ/view?usp=sharing), [NasNet](https://drive.google.com/file/d/16m2RC8MZSdvOVxK7p2NfX9rCFTACuFfj/view?usp=sharing).

If you think our project is helpful, please cite our work:

@article{JIA20EML,  
title = "EML-NET: An Expandable Multi-Layer NETwork for saliency prediction",  
journal = "Image and Vision Computing",  
volume = "95",  
pages = "103887",  
year = "2020",  
issn = "0262-8856",  
doi = "https://doi.org/10.1016/j.imavis.2020.103887",  
url = "http://www.sciencedirect.com/science/article/pii/S0262885620300196",  
author = "Sen Jia and Neil D.B. Bruce",  
keywords = "Saliency detection, Scalability, Loss function",  
}
