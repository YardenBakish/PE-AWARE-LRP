import os
import config
import argparse
import subprocess
#import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='evaluate perturbations')
    
 

    parser.add_argument('--threshold-type', default='mean',)
    parser.add_argument('--variant', default = 'basic',  type=str, help="")
    parser.add_argument('--model', type=str, required=True)
  

    parser.add_argument('--check-all', action='store_true')
    parser.add_argument('--analyze-all-lrp', action='store_true')
    parser.add_argument('--analyze-all-full-lrp', action='store_true')


 
    parser.add_argument('--data-path', type=str,
                  
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')

    parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],)
    
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')

 

    parser.add_argument('--imagenet-seg-path', type=str, default = "gtsegs_ijcv.mat",help='')
    args = parser.parse_args()
    return args





def run_segmentation(args):
    eval_seg_cmd        = "python evaluate_segmentation.py"
   
    
    eval_seg_cmd       +=  f' --method {args.method}'
    eval_seg_cmd       +=  f' --imagenet-seg-path {args.imagenet_seg_path}'
  

    eval_seg_cmd += f' --variant {args.variant}'
    eval_seg_cmd += f' --threshold-type {args.threshold_type} '
   
  
    seg_results_dir = 'seg_results' if args.threshold_type == 'mean' else f'seg_results_{args.threshold_type}'
    eval_seg_epoch_cmd = f"{eval_seg_cmd} --output-dir {seg_results_dir}/{args.method}"
    eval_seg_epoch_cmd += f" --custom-trained-model {args.model}" 
    print(f'executing: {eval_seg_epoch_cmd}')
    try:
       subprocess.run(eval_seg_epoch_cmd, check=True, shell=True)
       print(f"generated visualizations")
    except subprocess.CalledProcessError as e:
       print(f"Error: {e}")
       exit(1)





def run_segmentations_env(args):
   args.variant = 'basic'
   for c in ['full_lrp_GammaLinear_POS_ENC_gammaConv','full_lrp_GammaLinear_gammaConv']:
      args.method = c
      run_segmentation(args)
  


   
if __name__ == "__main__":
    args                   = parse_args()
    config.get_config(args, skip_further_testing = True, get_epochs_to_segmentation = True)
   
    if args.check_all:
       run_segmentations_env(args)
    else: 
      run_segmentation(args)

    