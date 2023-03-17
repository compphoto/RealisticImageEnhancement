import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Decoder(nn.Module):

    def __init__(self, shape, num_img_feat, num_pla_feat):
        super(Decoder, self).__init__()
        self.shape = shape
        self.img_model = self._make_layer(num_img_feat)
        self.pla_model = self._make_layer(num_pla_feat)

        self.combined = self._make_output(num_img_feat+num_pla_feat) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_feat):
        ans = nn.ModuleList()
        for _ in range(num_feat):
            m =  nn.Sequential(
                nn.Conv2d(1, 1, 3, padding=1),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True)
            )
            ans.append(m)
        return ans

    def _make_output(self, planes, readout=1):
        return nn.Sequential(
            nn.Conv2d(planes, readout, 3, stride=1, padding=1),
            nn.BatchNorm2d(readout),
            nn.Sigmoid()
            )

    def forward(self, x):
        img_feat, pla_feat = x
        feat = []

        for a, b in zip(img_feat, self.img_model):
            f = F.interpolate(b(a), self.shape)
            feat.append(f)

        for a, b in zip(pla_feat, self.pla_model):
            f = F.interpolate(b(a), self.shape)
            feat.append(f)

        feat = torch.cat(feat, dim=1)
        feat = self.combined(feat)
        return feat

def build_decoder(model_path=None, *args):
    decoder = Decoder(*args)
    if not model_path is None:
        loaded = torch.load(model_path)['state_dict']
        decoder.load_state_dict(loaded)
        print ("Loaded decoder", model_path)
    return decoder


