#from models.model_wrapper import model_env 
from models.model_handler import model_env 

from ViT_explanation_generator import LRP, LRP_RAP
import torchvision.transforms as transforms
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


import cv2
normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


IMGS = [
"val/n01614925/ILSVRC2012_val_00006571.JPEG",
"val/n01877812/ILSVRC2012_val_00014040.JPEG",
"val/n02006656/ILSVRC2012_val_00028586.JPEG",
"val/n01514859/ILSVRC2012_val_00032162.JPEG",
"val/n01440764/ILSVRC2012_val_00046252.JPEG",
"val/n01985128/ILSVRC2012_val_00032174.JPEG",]

# create heatmap from mask on image
def show_cam_on_image(img, mask):
   
    x = np.uint8(255 * mask)

    heatmap = cv2.applyColorMap(x, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_transformer_attribution(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
  
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis



def generate_visualization_full_LRP(original_image, class_index=None, method = None, prop_rules= None, conv_prop_rule = None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method=method,  cp_rule=args.cp_rule, prop_rules = prop_rules,  index=class_index, conv_prop_rule = conv_prop_rule).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_RAP(original_image, class_index=None, epsilon_rule = False, gamma_rule = False, default_op=None):
    transformer_attribution = RAP_generator.generate_LRP_RAP(original_image.unsqueeze(0).cuda(), method="full",  cp_rule=args.cp_rule,  index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def generate_visualization_custom_LRP(original_image, class_index=None, method = None,  prop_rules = None):
   
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), prop_rules = prop_rules, method=method, cp_rule=args.cp_rule,  index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(14, 14).unsqueeze(0).unsqueeze(0)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear', align_corners=False)
    transformer_attribution = transformer_attribution.squeeze().detach().cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
   
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis




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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--sample-path', 
                        required = True,
                        help='')
  parser.add_argument('--idx', 
                        type=int,
                        help='')
  parser.add_argument('--custom-trained-model', 
                     
                        help='')
  parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)

  parser.add_argument('--data-path', default='',)
  
  parser.add_argument('--variant', default = 'basic' , type=str, help="")
  parser.add_argument('--class-index', 
                       # default = "243",
                       type=int,
                        help='') #243 - dog , 282 - cat
  parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')
  
  parser.add_argument('--auto', action='store_true')
  parser.add_argument('--grid', action='store_true')





      
  
  
  args = parser.parse_args()
  config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
  config.set_components_custom_lrp(args,gridSearch = args.grid)
  if args.auto:
    exp_name      = args.variant
    exp_name = f'{args.data_set}/{exp_name}' 
    results_exp_dir              = f'{args.dirs["results_dir"]}/{exp_name}'
    args.custom_trained_model    =  is_valid_directory(results_exp_dir)
  

  idx = args.idx
  args.sample_path = IMGS[idx]
  image = Image.open(args.sample_path).convert('RGB')
  image_transformed = transform(image)

  
  if args.data_set == "IMNET100":
    args.nb_classes = 100
  else:
     args.nb_classes = 1000

  
  model = model_env(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        #model_LRP.head = torch.nn.Linear(model_LRP.head.weight.shape[1],100)
  if args.variant != 'variant_rope':
 
    checkpoint = torch.load(args.custom_trained_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)

  model.cuda()
  model.eval()
  attribution_generator = LRP(model)
  RAP_generator = LRP_RAP(model)


  output = model(image_transformed.unsqueeze(0).cuda())
  print_top_classes(output)
  #exit(1)
  filename = os.path.basename(args.sample_path)
    # Remove the file extension
  img_name = os.path.splitext(filename)[0]

  img_name = img_name.split("_")[-1]



  method_name = None
  vis = None
  if "custom_RAP" in args.method:
    vis = generate_visualization_RAP(image_transformed, args.class_index, epsilon_rule = epsilon_rule, gamma_rule = gamma_rule, default_op = default_op)
    method_name = "custom_RAP"
  elif args.method == "transformer_attribution" or args.method == "attribution_with_detach":
    vis = generate_visualization_transformer_attribution(image_transformed, args.class_index)
    method_name = "Att"
  elif "custom_lrp" in args.method:

   

    vis = generate_visualization_custom_LRP(image_transformed, 
                                            args.class_index,
                                           prop_rules = args.prop_rules,
                                           method = args.method
                                            )
    method_name = "custom_lrp"
  else:


    conv_prop_rule = args.conv_prop_rule

    vis = generate_visualization_full_LRP(image_transformed, 
                                        
                                          args.class_index,
                                           method = args.method,
                                          prop_rules = args.prop_rules,
                                          conv_prop_rule = conv_prop_rule
                                            )
    method_name = "lrp"


  os.makedirs(f"testing/visualization_final/{args.variant}", exist_ok=True)
  saved_image_path = f"testing/visualization_final/{args.variant}/{img_name}_{args.method}.png"
 
  plt.imsave(saved_image_path, vis)

  




