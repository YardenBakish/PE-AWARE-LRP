import os




import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
from datasets import load_dataset
import matplotlib.pyplot as plt
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
from helper_scripts.helper_functions import update_json
from attDatasets.wiki_dataset import load_wiki_dataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from utils import flip_arc, get_latest_checkpoint
from sklearn.metrics import auc
from llama_engine import run_LRP

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
import copy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import os
import json

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentation')

    parser.add_argument('--mode', type=str,
                                choices=['pert', 'analyze'],
                                required = True,
                                help='')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--reverse_default_abs', action='store_true')
   
    parser.add_argument('--model-size', type=str,
                            choices=['llama_2_7b', 'llama_3_8b', 'llama_tiny'],
                           required = True,
                            help='')

    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--reform', action='store_true')
    parser.add_argument('--without-abs', action='store_true')
    parser.add_argument('--single-norm', action='store_true')
    parser.add_argument('--experimental', action='store_true')


    parser.add_argument('--sep_heads', action='store_true')

    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--eval', action='store_true')

    
    parser.add_argument('--sequence-length', type=int,
                           )

    parser.add_argument('--variant', type=str,
                           default="baseline")
    parser.add_argument('--pe_only', action='store_true')


    args = parser.parse_args()
    args.dataset = 'wiki'

    if (args.pe_only and not args.pe) or (args.reform and not args.pe):
        print("no")
        exit(1)

    config.get_config(args, pert=True)

    if args.debug:
        args.pe = True
        args.reform = True
        args.single_norm = True
        args.without_abs = True
        args.clamp = True
    model_dir = f"pert_results_wiki" 
    model_dir = model_dir if args.reverse_default_abs == False else f"reverse_wiki/{model_dir}"


    model_dir = f"{model_dir}/abs" if args.without_abs == False else f"{model_dir}/no_abs"


    args.save_dir = f'finetuned_models/{model_dir}/{args.ext}'


    if args.debug:
        args.save_dir =f'finetuned_models/{model_dir}/debug/{args.ext}'
    
    if args.eval:
        args.save_dir = f'finetuned_models/eval/wiki/{args.ext}'
       
    os.makedirs(args.save_dir, exist_ok=True)
    return args


#path = "meta-llama/Llama-3.1-8B-Instruct"
#
#if args.model_size == 'llama_tiny':
#    path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#if args.model_size == 'llama_2_7b':
#    path = f"original_models/{args.model_size}/vanilla" 


def analyze(args):
    root_dir = 'finetuned_models'
    model_dir = f"pert_results_wiki" 

    model_dirs = [f'{model_dir}/abs'] #f'{model_dir}/no_abs',
    dirs = [f'{root_dir}/{model_dir}/{args.ext}' for model_dir in model_dirs]

    res = []
    for dir in dirs:
        if os.path.isdir(dir):
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith('.json'):
                        path = os.path.join(dir, file)
                        print(os.path.join(dir, file))
                        name = "baseline"

                        if "peOnly" in path:
                            name = "PE Only"
                        elif "_pe.json" in path:
                            name = "AttnLRP+PE"
                        if "abs" in path and "no_abs" not in path:
                            name = f"{name} (abs)"
                        if "experimental" in path:
                            name = f"{name}+experimental"
                        if "clamp" in path:
                            name = f"{name}+clamp"
                        if "Reform" in path:
                            name = f"{name}+reform"
                        with open(path, 'r') as file:
                            data = json.load(file)
                        curr = [name,data["generate_AU_AC"], data["generate_AU_MSE"], data["pruning_AU_AC"], data["pruning_AU_MSE"], data["LERF"], data["MERF"] ]
                        res.append(curr)
        else:
            print(f"{dir} is not a directory.")
            exit(1)
   
    
    latex_code = r'\begin{table}[h!]\centering' + '\n' + r'\begin{tabular}{c c c c c |c c c}' + '\n' 
    latex_code += r'\hline & \multicolumn{2}{c}{Generation} & \multicolumn{2}{c|}{Pruning}' r'\\ ' +'\n'
    
    latex_code += r'& AUAC $\uparrow$ & AU-MSE $\downarrow$ & AUAC $\uparrow$ & AU-MSE $\downarrow$ & LERF $\downarrow$ & MERF $\uparrow$' r'\\ ' +'\hline \n'
    res.sort( key=lambda x: x[0])

    for elem in res:
      row = elem[0]
      for cat in elem[1:]:
        row += f' & {cat:.3f}'
      row += r'\\ ' f'\n'
      latex_code += row
      
    latex_code += "\\hline\n\\end{tabular}\n\\caption{Segmentation Results using}\n\\end{table}"

    print(latex_code)




def eval_pert(args):


    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
    )

    #model_dir = f"{args.model_size}" if args.quant == False else f"{args.model_size}_QUANT"
    #model_dir = f"{model_dir}_KEEP" if args.should_keep == True else model_dir
    #
    #save_dir = f'finetuned_models/{model_dir}/pert_results_wiki'
    #os.makedirs(save_dir, exist_ok=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


    MAX_LEN = 512
    BATCH_SIZE = 1

    if args.quant:
        model = LlamaForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16,quantization_config=quantization_config, device_map="cuda", low_cpu_mem_usage = True)
    #model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
    else:
        model = LlamaForCausalLM.from_pretrained(args.model_checkpoint, device_map="cuda", torch_dtype=torch.bfloat16)



    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.unk_token_id == None:
        #tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("<|unkown|>")
        tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("#")
     
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    

    #model.to(device)


    model.gradient_checkpointing_enable()
    attnlrp.register(model)

    df = load_wiki_dataset()
    test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

  
    run_LRP(model,
        test_data_loader,
        tokenizer,
        isBinary=False,
        withPE = args.pe,
        reform=args.reform,
        pe_only = args.pe_only,
        withoutABS = args.without_abs,
        clamp = args.clamp,
        sep_heads = args.sep_heads,
        single_norm = args.single_norm,
        experimental = args.experimental,
        save_dir = args.save_dir,

        dataset="wiki",
        eval = args.eval,

        )


if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "pert":
        eval_pert(args)
    else:
        analyze(args)
       