import numpy as np
import os
import torch
import torch.nn.functional as F
from  torchvision.transforms.functional import resize as Tresize

from utils.networkutils import init_net, loadmodelweights
from utils.applyedits import apply_colorcurve, apply_exposure, apply_saturation, apply_whitebalancing, EDITS

from model.parametermodel import EditNet
from model.saliencymodel import EMLNET
from model.discriminator import VOTEGAN


class EditNetTrainer:
    def __init__(self, args):
        
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')

        self.model_names = ['Parameters']
        self.net_Parameters = init_net(EditNet(args),args.gpu_ids)
        self.net_D = init_net(VOTEGAN(args), args.gpu_ids)

        loadmodelweights(self.net_D,'bestmodels/realismnet.pth', self.device) 
        if args.init_parameternet_weights is not None:
            loadmodelweights(self.net_Parameters,args.init_parameternet_weights, self.device) 

        self.net_Saliency = init_net(EMLNET(args),args.gpu_ids)
        
        self.set_requires_grad(self.net_Saliency, False)
        self.set_requires_grad(self.net_D, False)

        self.optimizer_Parameters = torch.optim.Adam(self.net_Parameters.parameters(), lr=args.lr_parameters)

        self.net_D.eval()
        self.net_Saliency.eval()

        self.net_Parameters.train()

        # self.logs = []

        self.all_permutations = torch.tensor([
                                [0,1,2,3],[0,2,1,3],[0,3,1,2],[0,1,3,2],[0,2,3,1],[0,3,2,1],
                                [1,0,2,3],[1,2,0,3],[1,3,0,2],[1,0,3,2],[1,2,3,0],[1,3,2,0],
                                [2,0,1,3],[2,1,0,3],[2,3,0,1],[2,0,3,1],[2,1,3,0],[2,3,1,0],
                                [3,0,1,2],[3,1,0,2],[3,2,0,1],[3,0,2,1],[3,1,2,0],[3,2,1,0]
                                ]).float().to(self.device)

    def setEval(self):
        self.net_Parameters.eval()

    def setTrain(self):
        self.net_Parameters.train()

    def setinput(self, input):
        self.rgb = input['rgb'].to(self.device)
        self.mask = input['mask'].to(self.device)

        self.category = input['category'].to(self.device)
        self.ishuman = (self.category == 1).float()

        self.input = torch.cat((self.rgb,self.mask),dim=1).to(self.device)

        self.input_saliency = Tresize(self.net_Saliency(self.rgb),(self.args.crop_size,self.args.crop_size))
        self.numelmask = torch.sum(self.mask,dim=[1,2,3])

    def forward(self):
        permutation = torch.randperm(len(EDITS)).float().to(self.device)
        params_dic = self.net_Parameters(self.input, permutation.repeat(self.args.batch_size,1))
        # self.logs.append(params_dic)

        current_rgb = self.rgb
        for ed_in in range(self.args.nops):
            current_edited = torch.clamp(EDITS[permutation[ed_in].item()](current_rgb,params_dic),0,1)
            current_result = (1 - self.mask) * current_rgb + self.mask * current_edited

            current_rgb = current_result

        self.result = current_result


    def compute_gloss(self):
        before = torch.cat((self.rgb, self.mask), 1)
        before_D_value = self.net_D(before).squeeze(1)

        after = torch.cat((self.result, self.mask), 1)
        after_D_value = self.net_D(after).squeeze(1)
        self.realism_change = before_D_value - after_D_value

        self.output_saliency = Tresize(self.net_Saliency(self.result),(self.args.crop_size,self.args.crop_size))
        if self.args.sal_loss_type == 'percentage':
            saliency_change = (self.output_saliency - self.input_saliency) / (self.input_saliency + 1e-8)
            self.saliency_change = torch.sum(saliency_change * self.mask,axis=[1,2,3]) / self.numelmask
            self.saliency_change = torch.clip(self.saliency_change,-1,1)
            self.loss_saliency = torch.clip(torch.exp(self.args.w_sal*self.saliency_change),min=-10, max=10)
        elif self.args.sal_loss_type == 'percentage_increase':
            saliency_change = (self.output_saliency - self.input_saliency) / (self.input_saliency + 1e-8)
            self.saliency_change = torch.sum(saliency_change * self.mask,axis=[1,2,3]) / self.numelmask
            self.saliency_change = torch.clip(self.saliency_change,-1,1)
            self.loss_saliency = torch.clip(torch.exp(-self.args.w_sal*self.saliency_change),min=-10, max=10) # clipping to avoid exploding

        realism_component_human = (1+self.args.human_weight_gan * F.relu(self.realism_change))
        realism_component_other = (1+F.relu(self.realism_change - self.args.beta_r))
        self.loss_realism = self.ishuman.squeeze(1) * realism_component_human + (1-self.ishuman.squeeze(1)) * realism_component_other

        self.loss_g = torch.mean(self.loss_realism *  self.loss_saliency)


    def optimize_parameters(self):         
        self.optimizer_Parameters.zero_grad()  
        self.compute_gloss()
        self.loss_g.backward()
        self.optimizer_Parameters.step()     


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def savemodel(self,iteration, checkpointdir):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (iteration, name)
                save_path = os.path.join(checkpointdir, save_filename)
                net = getattr(self, 'net_' + name)
                if len(self.args.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.args.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)






    def setinput_hr(self, input):
        self.rgb = input['rgb'].to(self.device)
        self.mask = input['mask'].to(self.device)

        self.rgb_org = input['rgb_org'].to(self.device)
        self.mask_org = input['mask_org'].to(self.device)

        self.category = input['category'].to(self.device)
        self.ishuman = (self.category == 1).float()

        self.input = torch.cat((self.rgb,self.mask),dim=1).to(self.device)

        self.input_saliency = Tresize(self.net_Saliency(self.rgb),(self.args.crop_size,self.args.crop_size))
        self.numelmask = torch.sum(self.mask,dim=[1,2,3])
    

    def forward_allperm_hr(self):
            for permutation in self.all_permutations:
                params_dic = self.net_Parameters(self.input, permutation.repeat(self.args.batch_size,1))

                current_rgb = self.rgb_org
                for ed_in in range(self.args.nops):
                    current_edited = torch.clamp(EDITS[permutation[ed_in].item()](current_rgb,params_dic),0,1)
                    current_result = (1 - self.mask_org) * current_rgb + self.mask_org * current_edited

                    current_rgb = current_result

                self.result = Tresize(current_result,(self.args.crop_size,self.args.crop_size))
                self.compute_gloss()
                
                yield ( permutation.cpu().numpy(),
                        self.realism_change.cpu().numpy(), 
                        self.saliency_change.cpu().numpy(),
                        self.loss_realism.cpu().numpy(), 
                        self.loss_saliency.cpu().numpy(), 
                        self.loss_g.cpu().numpy(),
                        current_result.cpu().numpy(),
                        self.input_saliency.cpu().numpy(),
                        self.output_saliency.cpu().numpy(),)

    

    

    





