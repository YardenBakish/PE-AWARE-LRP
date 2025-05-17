import torch
import torch.nn as nn
from einops import rearrange
from modules.layers_ours import *
from baselines.ViT.weight_init import trunc_normal_
from baselines.ViT.layer_helpers import to_2tuple
import math




__all__ = ['PatchEmbed', 'ExpandedPatchEmbed', 'RefinedPatchEmbed', 'GatedPatchEmbedding']

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,padding = 0,kernel_size = None, stride = None, isWithBias = True, isWithReshape = True):
        super().__init__()
        self.isWithReshape = isWithReshape
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        kernel_size = patch_size if kernel_size is None else kernel_size
        stride      = patch_size if stride is None else stride

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding = padding, bias = isWithBias )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.isWithReshape:
            x = x.flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, conv_prop_rule, **kwargs):
        if self.isWithReshape:
            cam = cam.transpose(1,2)


            cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam,conv_prop_rule, **kwargs)
    



###############################################################################################
class ExpandedPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,padding = 0,kernel_size = None, stride = None, isWithBias = True, isWithReshape = True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size1 = to_2tuple(patch_size)
        patch_size2 = to_2tuple(patch_size*3)

        num_patches = (img_size[1] // patch_size1[1]) * (img_size[0] // patch_size1[0])
        self.img_size = img_size
        self.patch_size = patch_size1
        self.num_patches = num_patches

        self.sigma = 0 # nn.Parameter(torch.ones(1, num_patches, embed_dim) *0.7 )


        self.proj  = Conv2d(in_chans, embed_dim, kernel_size=patch_size1, stride=patch_size, bias = isWithBias)
        self.proj2 = Conv2d(in_chans, embed_dim, kernel_size=patch_size2, stride=patch_size,padding =16, bias = isWithBias)

        self.add = Add()
        self.clone = Clone()


    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x1, x2 = self.clone(x, 2)
        
        x1 = self.proj(x1).flatten(2).transpose(1, 2)
        x2 = self.proj2(x2)
       
        x2 = x2.flatten(2)
        x2 = x2.transpose(1, 2)
        x = self.add([self.sigma * x1,  x2])
        
        return x

    def relprop(self, cam, conv_prop_rule, **kwargs):
        (cam1, cam2) = self.add.relprop(cam, **kwargs)
        cam1 = cam1.transpose(1,2)
        cam1 = cam1.reshape(cam1.shape[0], cam1.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        
        cam2 = cam2.transpose(1,2)
        cam2 = cam2.reshape(cam2.shape[0], cam2.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        
        cam1 = self.proj.relprop(cam1, conv_prop_rule, **kwargs)

        cam2 = self.proj2.relprop(cam2, conv_prop_rule, **kwargs)
        cam = self.clone.relprop((cam1, cam2), **kwargs)
        return cam

###############################################################################################
class RefinedPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, isWithBias = False):
        super().__init__()

        depth = 4
        patch_embed_blocks = []
        tmp_img_size = img_size
        tmp_in_chans = in_chans


        img_size_tupple = to_2tuple(img_size)
        patch_size_tupple = to_2tuple(patch_size)
        num_patches = (img_size_tupple[1] // patch_size_tupple[1]) * (img_size_tupple[0] // patch_size_tupple[0])

        self.num_patches = num_patches

        for i in range(depth):
            tmp_embed_dim = embed_dim // (2 ** (depth - 1 - i))
            patch_embed_blocks += [PatchEmbed(
                img_size=tmp_img_size,  in_chans=tmp_in_chans, embed_dim=tmp_embed_dim,padding=1,kernel_size=3, patch_size = 3,stride=1, isWithReshape = False, isWithBias = isWithBias)  , ReLU()]
            tmp_in_chans = tmp_embed_dim
            
            patch_embed_blocks += [PatchEmbed(
                img_size=tmp_img_size,  in_chans=tmp_in_chans, embed_dim=tmp_embed_dim,padding=1,kernel_size=3,patch_size = 3,stride=1, isWithReshape = False, isWithBias = isWithBias)  , ReLU()]
           
            patch_embed_blocks += [MaxPool2d(kernel_size=2, stride=2)]

            tmp_img_size = tmp_img_size // 2
        patch_embed_blocks += [PatchEmbed(
                img_size=tmp_img_size,  in_chans=tmp_in_chans, embed_dim=tmp_embed_dim,padding=1,kernel_size=3,patch_size = 1,stride=1, isWithBias = isWithBias), ReLU()]

        self.patch_embed_blocks =  nn.ModuleList(patch_embed_blocks)
  

    def forward(self, x):
        for blk in self.patch_embed_blocks:
            x = blk(x)

            #print(x.shape)

        return x

    def relprop(self, cam, conv_prop_rule, **kwargs):
        for blk in reversed(self.patch_embed_blocks):
            if isinstance(blk, PatchEmbed):
                cam = blk.relprop(cam, conv_prop_rule,  **kwargs)
            else:
                cam = blk.relprop(cam,  **kwargs)
        return cam
    





###############################################################################################
class GatedPatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, isWithBias = False):
        super().__init__()


        patch_embed_blocks = []

        img_size_tupple = to_2tuple(img_size)
        patch_size_tupple = to_2tuple(patch_size)
        num_patches = (img_size_tupple[1] // patch_size_tupple[1]) * (img_size_tupple[0] // patch_size_tupple[0])

        self.num_patches = num_patches

        patch_embed_blocks = [PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, isWithBias= isWithBias ) ,ReLU()]

        self.patch_embed_blocks =  nn.ModuleList(patch_embed_blocks)
  

    def forward(self, x):
        for blk in self.patch_embed_blocks:
            x = blk(x)

        return x

    def relprop(self, cam, conv_prop_rule, **kwargs):
        for blk in reversed(self.patch_embed_blocks):
            if isinstance(blk, PatchEmbed):
                cam = blk.relprop(cam, conv_prop_rule,  **kwargs)
            else:
                cam = blk.relprop(cam,  **kwargs)
        return cam
    

###############################################################################################
'''

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,padding = 0,kernel_size = None, stride = None, isWithBias = True, isWithReshape = True):
        super().__init__()
        self.isWithReshape = isWithReshape
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        kernel_size = patch_size if kernel_size is None else kernel_size
        stride      = patch_size if stride is None else stride

        self.lambda_q1 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.lambda_k1 = nn.Parameter(0.3 * torch.ones((self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]), dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.proj1 = Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding = padding, bias = isWithBias )
        self.proj2 = Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding = padding, bias = isWithBias )


    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.isWithReshape:
            x = x.flatten(2).transpose(1, 2)
        return x

    def relprop(self, cam, conv_prop_rule, **kwargs):
        if self.isWithReshape:
            cam = cam.transpose(1,2)


            cam = cam.reshape(cam.shape[0], cam.shape[1],
                     (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        return self.proj.relprop(cam, **kwargs)'''