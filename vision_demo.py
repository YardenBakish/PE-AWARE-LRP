#from models.model_wrapper import model_env 
#python check_conservation.py --auto --mode analyze_conservarion_per_image --method custom_lrp_gamma_rule_full

from ViT_explanation_generator import LRP, LRP_RAP
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from tqdm import tqdm
import json
from torchvision import datasets
import argparse
from PIL import Image
import torch
from samples.CLS2IDX import CLS2IDX
import numpy as np
import matplotlib.pyplot as plt
import os
import config
from misc.helper_functions import is_valid_directory ,create_directory_if_not_exists, update_json
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.model_visualizations import deit_tiny_patch16_224 as vit_LRP
from models.model_visualizations import deit_base_patch16_224 as vit_LRP_base
from models.model_visualizations import deit_small_patch16_224 as vit_LRP_small




import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])




def model_handler(pretrained=False,args  = None , hooks = False,  **kwargs):
    if "size" in args.model_components:
        if args.model_components['size'] == 'base':
            return vit_LRP_base(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    patch_embed          = args.model_components["patch_embed"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],


                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
                    pretrained      = pretrained
            )
        elif args.model_components['size'] == 'small':
            return vit_LRP_small(
                    isWithBias           = args.model_components["isWithBias"],
                    isConvWithBias       = args.model_components["isConvWithBias"],

                    layer_norm           = args.model_components["norm"],
                    last_norm            = args.model_components["last_norm"],
                    attn_drop_rate       = args.model_components["attn_drop_rate"],
                    FFN_drop_rate        = args.model_components["FFN_drop_rate"],
                    projection_drop_rate = args.model_components['projection_drop_rate'],

                    patch_embed          = args.model_components["patch_embed"],

                    activation      = args.model_components["activation"],
                    attn_activation = args.model_components["attn_activation"],
                    num_classes     = args.nb_classes,
                    pretrained      = pretrained

            )
    
    return vit_LRP(
            isWithBias           = args.model_components["isWithBias"],
            isConvWithBias       = args.model_components["isConvWithBias"],

            layer_norm           = args.model_components["norm"],
            last_norm            = args.model_components["last_norm"],
            attn_drop_rate       = args.model_components["attn_drop_rate"],
            FFN_drop_rate        = args.model_components["FFN_drop_rate"],
            projection_drop_rate = args.model_components['projection_drop_rate'],

            patch_embed          = args.model_components["patch_embed"],

            activation      = args.model_components["activation"],
            attn_activation = args.model_components["attn_activation"],
            num_classes     = args.nb_classes,
            pretrained      = pretrained

        )
   

def print_top_classes(predictions, **kwargs):    
    # Print Top-5 predictions
    prob = torch.softmax(predictions, dim=1)
    class_indices = predictions.data.topk(5, dim=1)[1][0].tolist()
    max_str_len = 0
    class_names = []
    for cls_idx in class_indices:
        class_names.append(CLS2IDX[cls_idx])
        if len(CLS2IDX[cls_idx]) > max_str_len:
            max_str_len = len(CLS2IDX[cls_idx])
    
    print('Top 5 classes:')
    for cls_idx in class_indices:
        output_string = '\t{} : {}'.format(cls_idx, CLS2IDX[cls_idx])
        output_string += ' ' * (max_str_len - len(CLS2IDX[cls_idx])) + '\t\t'
        output_string += 'value = {:.3f}\t prob = {:.1f}%'.format(predictions[0, cls_idx], 100 * prob[0, cls_idx])
        print(output_string)



def concatenate_images_with_gaps(images, gap_size=10):
    """
    Concatenate multiple RGB images horizontally with gaps between them.
    
    Args:
        images: List of numpy arrays (RGB images)
        gap_size: Size of gap between images in pixels
    
    Returns:
        Combined RGB image as numpy array
    """
    # Get dimensions
    height = 224  # As specified in the requirements
    width = 224   # As specified in the requirements
    channels = 3  # RGB channels
    n_images = len(images)
    
    # Create empty array for the combined image with white background
    total_width = (n_images * width) + ((n_images - 1) * gap_size)
    combined_image = np.ones((height, total_width, channels), dtype=np.float32)
    
    # Place each image with gaps
    current_position = 0
    for img in images:
        # Ensure image is the correct size
        if img.shape != (height, width, channels):
            img = np.resize(img, (height, width, channels))
            
        # Place the image
        combined_image[:, current_position:current_position + width, :] = img
        current_position += width + gap_size
    
    return combined_image

def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_custom_LRP(original_image, class_index=None, method = None,  prop_rules = None, save_dir = None, posLens = False,  save_images_dir = None,batch_idx=None):
    res = []

    attributions = attribution_generator.generate_LRP(original_image.cuda(), prop_rules = prop_rules, method=method, cp_rule=args.cp_rule,  index=class_index)

    image_transformer_attribution = original_image[0].permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    
    
    image_copy = 255 *image_transformer_attribution
    image_copy = image_copy.astype('uint8')
    attributions = [elem[0,:] for elem in attributions]
    vis = [None,None,None]

    for i in range(3):
        attributions[i] = attributions[i].reshape(14, 14).unsqueeze(0).unsqueeze(0)
        attributions[i] = torch.nn.functional.interpolate(attributions[i], scale_factor=16, mode='bilinear', align_corners=False)
        attributions[i] = attributions[i].squeeze().detach().cpu().numpy()
        attributions[i] = (attributions[i] - attributions[i].min()) / (attributions[i].max() - attributions[i].min())
        vis[i] = show_cam_on_image(image_transformer_attribution, attributions[i])
        vis[i] =  np.uint8(255 * vis[i])
        vis[i] = cv2.cvtColor(np.array(vis[i]), cv2.COLOR_RGB2BGR)
        #plt.imsave(f"{save_images_dir}/img_{batch_idx}_{d[i]}.png" , vis[i])
       



    conc_imgs = concatenate_images_with_gaps([image_copy, vis[0], vis[1], vis[2]])
    conc_imgs = conc_imgs.astype('uint8')
    plt.imsave(f"{save_dir}/output.png" , conc_imgs)
    return attributions





if __name__ == "__main__":
  parser = argparse.ArgumentParser()


  parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', )
  parser.add_argument('--image', type=str, required=True)

  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")

  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='custom_lrp',
                        help='')
  

  args = parser.parse_args()

  save_dir = f"samples/"
  output_dir = f"output/"
  os.makedirs(save_dir, exist_ok=True)
  os.makedirs(output_dir , exist_ok=True)

  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args)

  

  image = Image.open(args.image).convert('RGB')
  print(image)
  image_transformed = transform(image)

 

  
  if args.data_set == "IMNET100":
    args.nb_classes = 100
  else:
     args.nb_classes = 1000

  
  model = model_handler(pretrained=True, 
                      args = args,
                      hooks = True,
                    )
  
  
  #checkpoint = torch.load(args.custom_trained_model, map_location='cpu')
 
  #model.load_state_dict(checkpoint['model'], strict=False)
  model.cuda()
  model.eval()
  attribution_generator = LRP(model)
  output = model(image_transformed.unsqueeze(0).cuda())
  print_top_classes(output)

  generate_visualization_custom_LRP(image_transformed.unsqueeze(0).cuda() , 
                                             None,
                                             prop_rules = args.prop_rules,
                                             method = args.method,
                                             save_dir = output_dir,
                                            )






