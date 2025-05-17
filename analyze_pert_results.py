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
    
    parser.add_argument('--normalized-pert', type=int, default=0, choices = [0,1])
    parser.add_argument('--model', type=str, required=True)


    parser.add_argument('--grid', action='store_true')


    parser.add_argument('--fract', type=float,
                        default=0.1,
                        help='')

 
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--pass-vis', action='store_true')
    parser.add_argument('--gen-latex', action='store_true')
    parser.add_argument('--extended', action='store_true')

    parser.add_argument('--check-all', action='store_true')
    parser.add_argument('--default-norm', action='store_true')



    parser.add_argument('--generate-plots', action='store_true', default=True)
   

  
    parser.add_argument('--work-env', type=str,
                        help='')
    
    
    parser.add_argument('--variant', default = 'basic',  type=str, help="")
    
   
    parser.add_argument('--neg', type=int, choices = [0,1], default = 0)
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--data-set', default='IMNET', choices=['IMNET100','CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--num-workers', type=int,
                        default= 1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='transformer_attribution',
                        help='')

    parser.add_argument('--both',  action='store_true')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Runs the first 5 samples and visualizes ommited pixels')
    parser.add_argument('--wrong', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--scale', type=str,
                        default='per',
                        choices=['per', '100'],
                        help='')

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
    return args





def parse_pert_results(pert_results_path=None, acc_keys=None, args=None, metric=None, optional_method = None ):
    pos_values = {}
    neg_values = {}
    pos_lists = {}    
    neg_lists = {} 
    
    method = args.method if optional_method == None else optional_method

 
    for res_dir in os.listdir(pert_results_path):
        res_path = os.path.join(pert_results_path, res_dir)
   

        if os.path.isdir(res_path):
            # The key corresponds to the number in res_X
            if "res" not in res_dir:
               continue
      
            res_key = int(res_dir.split('_')[1])
            if res_key not in acc_keys:
                continue
          
            if ((args.normalized_pert == 0 and "base" not in res_path) or (args.normalized_pert and "base" in res_path)):
               continue
            
            pert_results_file = os.path.join(res_path, 'pert_results.json')
            with open(pert_results_file, 'r') as f:
                pert_data = json.load(f)
                if f'{method}_pos_auc_{metric}' not in  pert_data:
                   continue
                pos_values[res_key] = pert_data.get(f'{method}_pos_auc_{metric}', 0)
                neg_values[res_key] = pert_data.get(f'{method}_neg_auc_{metric}', 0)

                pos_lists[res_key] = pert_data.get(f'{method}_pos_{metric}', [])
                neg_lists[res_key] = pert_data.get(f'{method}_neg_{metric}', [])
   
    return pos_values, neg_values, pos_lists, neg_lists



def parse_acc_results(acc_results_path):
    acc_dict = {}
    with open(acc_results_path, 'r') as f:
        data = json.load(f)
        for key, value in data.items():
            # Extract the numeric part from key like "2_acc"
            if "acc" in key:
               acc_key = int(key.split('_')[0])
               # Extract the accuracy value after "Acc@1"
               acc_value = float(value.split('Acc@1 ')[1].split(' ')[0])
               acc_dict[acc_key] = acc_value
    return acc_dict



def get_sorted_checkpoints(directory):
    # List to hold the relative paths and their associated numeric values
    checkpoints = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern 'checkpoint_*.pth'
            match = re.match(r'checkpoint_(\d+)\.pth', file)
            if match:
                # Extract the number from the filename
                number = int(match.group(1))
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Append tuple (number, relative_path)
                checkpoints.append((number, relative_path))

    # Sort the checkpoints by the number
    checkpoints.sort(key=lambda x: x[0])

    # Return just the sorted relative paths
    return [f'{directory}/{relative_path}'  for _, relative_path in checkpoints]





def run_perturbations(args):
    eval_pert_cmd        = "python evaluate_perturbations.py"
   
    
    eval_pert_cmd       +=  f' --method {args.method}'
    eval_pert_cmd       +=  f' --both'

    if args.grid:
       eval_pert_cmd       +=  f' --grid'
       
    eval_pert_cmd       +=  f' --data-path {args.data_path}'
    eval_pert_cmd       +=  f' --data-set {args.data_set}'

    eval_pert_cmd       +=  f' --batch-size {args.batch_size}'
    eval_pert_cmd       +=  f' --num-workers {args.num_workers}'

    eval_pert_cmd       +=  f' --normalized-pert {args.normalized_pert}'
    eval_pert_cmd       +=  f' --fract {args.fract}'

    variant          = f'{args.variant}'
    eval_pert_cmd += f' --variant {args.variant}'
  

   
      
    pert_results_dir = 'pert_results/' 
    pert_results_dir = f"{pert_results_dir}/{args.method}"
    os.makedirs(pert_results_dir,exist_ok=True)
    eval_pert_epoch_cmd = f"{eval_pert_cmd} --output-dir {pert_results_dir}"
    eval_pert_epoch_cmd += f" --work-env {pert_results_dir}/work_env/" 
    eval_pert_epoch_cmd += f" --custom-trained-model {args.model}" 
    print(f'executing: {eval_pert_epoch_cmd}')
    try:
       subprocess.run(eval_pert_epoch_cmd, check=True, shell=True)
       print(f"generated visualizations")
    except subprocess.CalledProcessError as e:
       print(f"Error: {e}")
       exit(1)



def run_perturbations_env(args):
   args.variant = 'basic'
   for c in ['full_lrp_GammaLinear_POS_ENC_gammaConv','full_lrp_GammaLinear_gammaConv']:
      args.method = c
      run_perturbations(args)



if __name__ == "__main__":
    args                   = parse_args()
  
    config.get_config(args, skip_further_testing = True, get_epochs_to_perturbate = True)
    if args.check_all:
      run_perturbations_env(args)
    else: 
      run_perturbations(args)
