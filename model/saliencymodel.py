import torch
import torch.nn as nn
import torchvision.transforms as T
from  torchvision.transforms.functional import resize as Tresize

from model.EMLNETSaliency import resnet
from model.EMLNETSaliency import decoder


class EMLNET(nn.Module):
    def __init__(self, args):
        super(EMLNET, self).__init__()
        self.size = (480, 640) 
        self.num_feat = 5

        self.img_model = resnet.resnet50('model/EMLNETSaliency/res_imagenet.pth')
        self.pla_model = resnet.resnet50('model/EMLNETSaliency/res_places.pth')
        self.decoder_model = decoder.build_decoder('model/EMLNETSaliency/res_decoder.pth', self.size, self.num_feat, self.num_feat)

        self.gaussianblur = T.GaussianBlur(kernel_size=(33, 33), sigma=(5, 5))

    def forward(self, rgb):
        # rgb expected: [B, 3, 480, 640]

        rgb = Tresize(rgb, (self.size[0], self.size[1]))
        img_feat = self.img_model(rgb, decode=True)
        pla_feat = self.pla_model(rgb, decode=True)

        pred = self.decoder_model([img_feat, pla_feat])
        # pred expected : [B, 1, 480, 640]
        
        # apply gaussian with sigma = 5
        pred = self.gaussianblur(pred)
        
        return pred
        
