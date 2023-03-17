from cgitb import reset
from operator import imod
from turtle import forward
from unittest import case
import torch
import torch.nn as nn
from kornia.color import rgb_to_hsv, hsv_to_rgb

COLORCURVE_L = 8

class EditOperator(nn.Module):
    def __init__(self):
        super(EditOperator,self).__init__()

    def forward(self, rgb, parameters):
        edited_wb = apply_whitebalancing(rgb,parameters)
        edited_sat = apply_saturation(rgb,parameters)
        edited_cc = apply_colorcurve(rgb,parameters)
        edited_ex = apply_exposure(rgb,parameters)
        return [edited_wb,edited_cc,edited_sat,edited_ex]

def apply_whitebalancing(input, parameters):
    param = parameters['whitebalancing']
    param = param / (param[:,1:2] + 1e-9)
    result = input / (param[:,:,None,None] + 1e-9)
    return result

def apply_colorcurve(input, parameters):
    color_curve_param = torch.reshape(parameters['colorcurve'],(-1,3,COLORCURVE_L))
    color_curve_sum = torch.sum(color_curve_param,dim=[2])
    total_image = torch.zeros_like(input)
    for i in range(COLORCURVE_L):
        total_image += torch.clip(input * COLORCURVE_L - i, 0, 1) * color_curve_param[:,:,i][:,:,None,None]
    result =  total_image / (color_curve_sum[:,:,None,None] + 1e-9)
    return result

def apply_saturation(input, parameters):
    hsv = rgb_to_hsv(input)
    param = parameters['saturation'][:,:,None,None]
    s_new = hsv[:,1:2,:,:] * param
    hsv_new = hsv.clone()
    hsv_new[:,1:2,:,:] = s_new
    result = hsv_to_rgb(hsv_new)
    return result

def apply_exposure(input, parameters):
    result = input * parameters['exposure'][:,:,None,None]
    return result
    
EDITS = {
    0: apply_whitebalancing,
    1: apply_colorcurve,
    2: apply_saturation,
    3: apply_exposure,
}

def applyedit(rgb, mask, result_batch, action):
    resultB = torch.zeros_like(rgb)
    for B in range(rgb.shape[0]):
        edited = result_batch[action[B].item()]
        result = edited * (mask) + rgb * (1 - mask)
        resultB[B,:,:,:] = result[B,:,:,:]
    resultB = torch.clamp(resultB,0,1)
    return resultB

# def applyedit(rgb, mask, result_batch, action, args):
#     result_batch = torch.stack(result_batch,dim=1)
#     indices = action.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#     indices = indices.repeat_interleave(torch.tensor([args.crop_size]).to('cuda'),dim=4)
#     indices = indices.repeat_interleave(torch.tensor([args.crop_size]).to('cuda'),dim=3)
#     indices = indices.repeat_interleave(torch.tensor([3]).to('cuda'),dim=2)
#     edited = torch.gather(result_batch, 1,indices).squeeze()
#     result = edited * mask + rgb * (1-mask)
#     return result