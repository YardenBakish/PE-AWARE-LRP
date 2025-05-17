
import os
from tqdm import tqdm
import h5py
import config
import argparse
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Import saliency methods and models
from baselines.ViT.misc_functions import *

from ViT_explanation_generator import Baselines, LRP
from old.model_ablation import deit_tiny_patch16_224 as vit_LRP

#from dataset.label_index_corrector  import *


from models.model_handler import model_env 



from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from deit.datasets import build_dataset
import torch

from torchvision.datasets import ImageNet
from torchvision import datasets, transforms




imagenet_normalize = transforms.Compose([
    #transforms.Resize(256, interpolation=3),
    #transforms.CenterCrop(224),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])


def compute_saliency_and_save(args):

    first = True
    #correct_label_dic = convertor_dict()

    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam_pred = f.create_dataset('vis_pred',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        
        data_cam_target = f.create_dataset('vis_target',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        
        
        count_correct = 0 
        
        for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            
 
            if first:
                 first = False
                 data_cam_pred.resize(data_cam_pred.shape[0] + data.shape[0] - 1, axis=0)
                 data_cam_target.resize(data_cam_target.shape[0] + data.shape[0] - 1, axis=0)

                 data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                 data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                 data_cam_pred.resize(data_cam_pred.shape[0] + data.shape[0], axis=0)
                 data_cam_target.resize(data_cam_target.shape[0] + data.shape[0], axis=0)
                 data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                 data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)
            #np.save(f"input_test_{batch_idx}.npy", data.data.cpu().numpy())

            data = imagenet_normalize(data)
          

            data = data.to(device)
            data.requires_grad_()

            #index = None
            #if args.vis_class == 'target':
            #    index = target

            #epsilon_rule = args.epsilon_rule
            #gamma_rule   = args.gamma_rule
            #default_op   = args.default_op

            if 'custom_lrp' in args.method:

                 #print(data.shape[0])
                 Res_pred   = lrp.generate_LRP(data, method=args.method, cp_rule = args.cp_rule, prop_rules = args.prop_rules, index=None).reshape(data.shape[0],14, 14)
                 Res_pred=Res_pred.unsqueeze(1).detach()
                 Res_target = lrp.generate_LRP(data, method=args.method, cp_rule = args.cp_rule, prop_rules = args.prop_rules, index=target).reshape(data.shape[0],14, 14).unsqueeze(1).detach()

            if args.method == 'rollout':
                print("FIXME: make sure which model is sent out for this method")
                exit(1)
                Res_pred = baselines.generate_rollout(data, start_layer=1).reshape(data.shape[0], 1, 14, 14)
                Res_target = Res_pred

                # Res = Res - Res.mean()

            elif args.method == 'lrp':
                Res_pred   = lrp.generate_LRP(data, start_layer=1, cp_rule = args.cp_rule, prop_rules = args.prop_rules, index=None).reshape(data.shape[0],14, 14).detach()
                Res_target = lrp.generate_LRP(data, start_layer=1, cp_rule = args.cp_rule, prop_rules = args.prop_rules, index=target).reshape(data.shape[0],14, 14).detach()

                # Res = Res - Res.mean()

            elif args.method == 'transformer_attribution' or  args.method == 'attribution_with_detach':
                #output = model_LRP(data)
                #print(f"target: {target}")
                #print(f"predicted:  {output.data.topk(5, dim=1)[1][0].tolist() }")
                #if output.data.topk(5, dim=1)[1][0].tolist()[0] == target:
                #    count_correct+=1 
                Res_pred   = lrp.generate_LRP(data, start_layer=1, method="grad",  cp_rule = args.cp_rule, index=None).reshape(data.shape[0], 1, 14, 14)
                Res_target = lrp.generate_LRP(data, start_layer=1, method="grad",  cp_rule = args.cp_rule, index=target).reshape(data.shape[0], 1, 14, 14)
                
                # Res = Res - Res.mean()

            elif 'full_lrp' in args.method:
              
                Res_pred   = lrp.generate_LRP(data, method=args.method,  cp_rule = args.cp_rule, prop_rules = args.prop_rules,  conv_prop_rule = args.conv_prop_rule, index=None).reshape(data.shape[0], 1, 224, 224).detach()
                Res_target = lrp.generate_LRP(data, method=args.method,  cp_rule = args.cp_rule, prop_rules = args.prop_rules,  conv_prop_rule = args.conv_prop_rule, index=target).reshape(data.shape[0], 1, 224, 224).detach()
                # Res = Res - Res.mean()

            elif args.method == 'lrp_last_layer':
              pass
              #  Res = orig_lrp.generate_LRP(data, method="last_layer", is_ablation=args.is_ablation, index=index) \
              #      .reshape(data.shape[0], 1, 14, 14)
                # Res = Res - Res.mean()

            elif args.method == 'attn_last_layer':
                Res_pred = lrp.generate_LRP(data, method="last_layer_attn",  cp_rule = args.cp_rule, is_ablation=args.is_ablation) \
                    .reshape(data.shape[0], 1, 14, 14)
                Res_target = Res_pred
            elif args.method == 'attn_gradcam':
                print("FIXME: make sure which model is sent out for this method")
                exit(1)
                #Res = baselines.generate_cam_attn(data, index=index).reshape(data.shape[0], 1, 14, 14)

            if 'full_lrp' not in args.method and args.method != 'input_grads':
                Res_pred = torch.nn.functional.interpolate(Res_pred, scale_factor=16, mode='bilinear').cuda()
                Res_target = torch.nn.functional.interpolate(Res_target, scale_factor=16, mode='bilinear').cuda()

            
            
            for i in range(data.shape[0]):
              curr_map_pred = Res_pred[i,0,:,:]
              Res_pred[i,0,:,:] = (curr_map_pred - curr_map_pred.min()) / (curr_map_pred.max() - curr_map_pred.min())
              
              curr_map_target = Res_target[i,0,:,:]
              Res_target[i,0,:,:] = (curr_map_target - curr_map_target.min()) / (curr_map_target.max() - curr_map_target.min())
            
            #np.save(f"test_custom_{batch_idx}.npy", Res_pred.data.cpu().numpy() )
            
            #Res_pred = (Res_pred - Res_pred.min()) / (Res_pred.max() - Res_pred.min())
            #Res_target = (Res_target - Res_target.min()) / (Res_target.max() - Res_target.min())


            data_cam_pred[-data.shape[0]:]   = Res_pred.data.cpu().numpy()
            data_cam_target[-data.shape[0]:] = Res_target.data.cpu().numpy()


        print(f"count_correct: {count_correct}") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')

    parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--ablated-component', type=str,
                        choices=['softmax', 'layerNorm', 'bias'],)
    
    parser.add_argument('--normalized-pert', type=int, default=1, choices = [0,1])

    parser.add_argument('--work-env', type=str, help='', required = True)
    parser.add_argument('--variant', default = 'basic', help="")

    parser.add_argument('--custom-trained-model', type=str,
                   
                        help='')
    parser.add_argument('--data-set', default='IMNET100', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 2,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                    
                        help='')
    parser.add_argument('--grid', action='store_true')

    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.', default=True)
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-ia', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-fgx', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-m', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--no-reg', action='store_true',
                        default=False,
                        help='')
    parser.add_argument('--is-ablation', type=bool,
                        default=False,
                        help='')
    parser.add_argument('--data-path', type=str,
                     
                        help='')
    args = parser.parse_args()

    config.get_config(args, skip_further_testing = True)
    config.set_components_custom_lrp(args, gridSearch = args.grid)
    
    
    
    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    if args.work_env:
        PATH = args.work_env
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/results.hdf5'.format(args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}'.format(args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}_{}'.format(args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}'.format(args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Model
    #FIXME: currently only attribution method is tested. Add support for other methods using other variants 
    model = vit_LRP(pretrained=True).cuda()
    baselines = Baselines(model)

    # LRP
    if args.custom_trained_model != None:
        if args.data_set == 'IMNET100':
            args.nb_classes = 100
        else:
            args.nb_classes = 1000
    
        model_LRP = model_env(pretrained=False, 
                      args = args,
                      hooks = True,
                    )
        
        checkpoint = torch.load(args.custom_trained_model, map_location='cpu')

        model_LRP.load_state_dict(checkpoint['model'], strict=False)
        model_LRP.to(device)
    else:
        print("currently have to supply custom trained model")
        exit(1)
        model_LRP = vit_LRP(pretrained=True).cuda()
    model_LRP.eval()
    lrp = LRP(model_LRP)

    # orig LRP
    ''' model_orig_LRP = vit_orig_LRP(pretrained=True).cuda()
    model_orig_LRP.eval()
    orig_lrp = LRP(model_orig_LRP)'''

    #size = int(args.input_size / args.eval_crop_ratio) 
    # Dataset loader for sample images

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.Resize(size, interpolation=3),
        #transforms.CenterCrop(args.input_size), 
        transforms.ToTensor(),
    ])

    dataset_val = None
    if args.normalized_pert:
        dataset_val, _ = build_dataset(is_train=False, args=args)
    else:
        if args.data_set == 'IMNET':
            root = os.path.join(args.data_path,  'val')
            dataset_val = datasets.ImageFolder(root, transform=transform)

        elif args.data_set == 'IMNET100':
            root = os.path.join(args.data_path,  'val')
            dataset_val = datasets.ImageFolder(root, transform=transform)

    
   

    #torch.manual_seed(42)
  
    np.random.seed(42)
    total_size  = len(dataset_val)
    indices = list(range(total_size))
    subset_size = int(total_size *args.fract)
    random_indices = np.random.choice(indices, size=subset_size, replace=False)
    sampler = SubsetRandomSampler(random_indices)

    #first 0.1
    '''total_size  = len(dataset_val)
    subset_size = int(total_size * 0.1)
    indices     = list(range(subset_size))
    dataset_val = Subset(dataset_val, indices)'''

    #sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    #imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', download=False, transform=transform)
    sample_loader = torch.utils.data.DataLoader(    
        dataset_val, sampler=sampler,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_mem,
        num_workers=args.num_workers,
        drop_last = True

    )

    compute_saliency_and_save(args)

