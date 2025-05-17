import argparse
import torch
import numpy as np
from numpy import *

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution",
                     prop_rules = False,
                     #gamma_rule = False, 
                     #epsilon_rule = False, 
                     cp_rule = False, 
                     #default_op = True, 
                     is_ablation=False, 
                     conv_prop_rule = None,
                     start_layer=0):
        batch_size = input.shape[0]
        output = self.model(input)

       
        

        kwargs = {"alpha": prop_rules["linear_alpha_rule"] , 
                  "epsilon_rule": prop_rules['epsilon_rule'], 
                  "gamma_rule": prop_rules['linear_gamma_rule'],
                  "conv_gamma_rule": prop_rules['conv_gamma_rule'],
                  "default_op": prop_rules["default_op"] ,
                  }

      
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            index = torch.tensor(index)

        one_hot = np.zeros((batch_size, output.shape[-1]), dtype=np.float32)
        one_hot[torch.arange(batch_size), index.data.cpu().numpy()] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, cp_rule = cp_rule, is_ablation=is_ablation,
                                  start_layer=start_layer, conv_prop_rule = conv_prop_rule, **kwargs)




class LRP_RAP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP_RAP(self, input, index=None, method="transformer_attribution",gamma_rule = False, epsilon_rule = False, cp_rule = False, default_op = True, is_ablation=False, start_layer=0):
        batch_size = input.shape[0]
        output = self.model(input)
        kwargs = {"alpha": 1, "epsilon_rule": epsilon_rule, "gamma_rule": gamma_rule, "default_op": default_op}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)
            index = torch.tensor(index)

        one_hot = np.zeros((batch_size, output.shape[-1]), dtype=np.float32)
        one_hot[torch.arange(batch_size), index.data.cpu().numpy()] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.RAP_relprop(torch.tensor(one_hot_vector).to(input.device), method=method, cp_rule = cp_rule, is_ablation=is_ablation,
                                  start_layer=start_layer)




class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_cam_attn(self, input, index=None):
        output = self.model(input.cuda(), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    def generate_rollout(self, input, start_layer=0):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        return rollout[:,0, 1:]
