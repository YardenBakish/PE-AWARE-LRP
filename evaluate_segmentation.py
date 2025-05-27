import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import os
from misc.helper_functions import *


from tqdm import tqdm
from utils.metrices import *
from models.model_handler import model_env 
#from utils.saver import Saver
from utils.iou import IoU
import config
from data.imagenet_new import Imagenet_Segmentation

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


from ViT_explanation_generator import Baselines, LRP
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F

plt.switch_backend('agg')


# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--custom-trained-model', type=str, 
                    help='Model path')
parser.add_argument('--threshold-type', choices = ['mean', 'otsu', 'MoV'], required = True)



parser.add_argument('--variant', default = 'basic', help="")
parser.add_argument('--input-size', default=224, type=int, help='images input size')

parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")

parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)
parser.add_argument('--output-dir', required = True)


parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
      
                    help='')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
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
parser.add_argument('--imagenet-seg-path', type=str, required=True)
args = parser.parse_args()

config.get_config(args, skip_further_testing = True)
config.set_components_custom_lrp(args)
os.makedirs(args.output_dir, exist_ok=True)


args.checkname = args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

# Define Saver
#saver = Saver(args)
#saver.results_dir = os.path.join(saver.experiment_dir, 'results')
#if not os.path.exists(saver.results_dir):
#    os.makedirs(saver.results_dir)
#if not os.path.exists(os.path.join(saver.results_dir, 'input')):
#    os.makedirs(os.path.join(saver.results_dir, 'input'))
#if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
#    os.makedirs(os.path.join(saver.results_dir, 'explain'))
#
#args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
#if not os.path.exists(args.exp_img_path):
#    os.makedirs(args.exp_img_path)
#args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
#if not os.path.exists(args.exp_np_path):
#    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


sizeX = int(args.input_size / args.eval_crop_ratio)

imagenet_trans = transforms.Compose([
        #transforms.Resize(sizeX, interpolation=3),
        #transforms.CenterCrop(args.input_size), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])




def otsu_threshold(img):
    """
    Compute Otsu's threshold for a 2D array.
    """
    # Flatten the image into 1D array
    flat = img.flatten()
    
    # Get histogram
    hist, bins = np.histogram(flat, bins=256, range=(0,1))
    hist = hist.astype(float)
    
    # Get bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Get total number of pixels
    total = hist.sum()
    
    best_thresh = 0
    best_variance = 0
    
    # Calculate cumulative sums
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Calculate cumulative means
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Calculate between class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Get threshold with maximum variance
    idx = np.argmax(variance)
    best_thresh = bin_centers[idx]
    
    return best_thresh



test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), Image.NEAREST),
])

ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=imagenet_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)

# Model

#FIXME: currently only attribution method is tested. Add support for other methods using other variants 

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


model_LRP.eval()
lrp = LRP(model_LRP)



metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)



def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    # pred[0, 0] = 282
    # print('Pred cls : ' + str(pred))
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.cuda()

    return Tt


def eval_batch(image, labels, evaluator, index):
    evaluator.zero_grad()
    # Save input image


    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    
    # segmentation test for the rollout baseline
    #epsilon_rule = args.epsilon_rule 
    #gamma_rule   = args.gamma_rule
    #default_op   = args.default_op

    if 'custom_lrp' in args.method:


        Res = lrp.generate_LRP(image.cuda(), method=args.method, cp_rule = args.cp_rule,prop_rules = args.prop_rules,).reshape(14, 14).unsqueeze(0).unsqueeze(0) 
    
    # segmentation test for the LRP baseline (this is full LRP, not partial)
    elif 'full_lrp' in args.method:
        Res  = lrp.generate_LRP(image.cuda(), method=args.method,  cp_rule = args.cp_rule, prop_rules = args.prop_rules,  conv_prop_rule = args.conv_prop_rule, index=None).reshape(1, 1, 224, 224).detach()
    
    # segmentation test for our method
    elif args.method == 'transformer_attribution' or  args.method == 'attribution_with_detach':
        Res = lrp.generate_LRP(image.cuda(), start_layer=1, method="transformer_attribution", cp_rule = args.cp_rule).reshape(batch_size, 1, 14, 14)
    
    # segmentation test for the partial LRP baseline (last attn layer)
    

    if 'full_lrp' not in args.method:
        # interpolate to full image size (224,224)
        Res = torch.nn.functional.interpolate(Res, scale_factor=16, mode='bilinear').cuda()
    
    # threshold between FG and BG is the mean    
    Res = (Res - Res.min()) / (Res.max() - Res.min())

    if args.threshold_type == 'otsu':
        ret =  otsu_threshold(Res.cpu().detach().numpy())
    elif args.threshold_type == 'MoV':
        ret =   Res.mean() / (3* Res.std() + 1e-10 )
    else:
        ret = Res.mean()

    
    #Res1.mean() / (3* Res1.std() + 1e-10 )

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0


    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)
    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)



    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)
    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    # print("output", output.shape)
    # print("ap labels", labels.shape)
    # ap = np.nan_to_num(get_ap_scores(output, labels))
    ap = np.nan_to_num(get_ap_scores(output_AP, labels))
    f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    bg_intersection = batch_inter[0]
    fg_intersection = batch_inter[1]
    tot_labeled_foreground = np.sum(labels[0].cpu().numpy() > 0)


    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target, bg_intersection, fg_intersection, tot_labeled_foreground


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

total_bg_intersection = 0
total_fg_intersection = 0
total_labeled_foreground = 0
total_labeled_background = 0


predictions, targets = [], []

count = 0
for batch_idx, (image, labels) in enumerate(iterator):

    if args.method == "blur":
        images = (image[0].cuda(), image[1].cuda())
    else:
        images = image.cuda()
    labels = labels.cuda()
    # print("image", image.shape)
    # print("lables", labels.shape)

    correct, labeled, inter, union, ap, f1, pred, target, bg_intersection, fg_intersection, labeled_foreground = eval_batch(images, labels, model, batch_idx)

    predictions.append(pred)
    targets.append(target)

    total_bg_intersection+=bg_intersection
    total_fg_intersection+=fg_intersection
    total_labeled_foreground+=labeled_foreground
    total_labeled_background+=(50176 - labeled_foreground)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)

    mean_fg_intersection = total_fg_intersection / total_labeled_foreground
    mean_bg_intersection = total_bg_intersection / (total_labeled_background)
    iterator.set_description('pixAcc: %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

#predictions = np.concatenate(predictions)
#targets = np.concatenate(targets)
#pr, rc, thr = precision_recall_curve(targets, predictions)
#np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
#np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

#plt.figure()
#plt.plot(rc, pr)
#plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

# txtfile = 'result_mIoU_%.4f.txt' % mIoU

update_json(f"{args.output_dir}/seg_results.json", 
            {'mIoU': f'{mIoU:.4f}', 
             'Pixel Accuracy': f'{(pixAcc * 100):.4f}', 
             'mAP': f'{mAp:.4f}',
             'mean_bg_intersection': f'{(mean_bg_intersection*100):.4f}' ,
             'mean_fg_intersection':f'{(mean_fg_intersection*100):.4f}',

             })


print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))










