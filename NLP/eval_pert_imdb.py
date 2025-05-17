import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
import matplotlib.pyplot as plt
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from helper_scripts.helper_functions import update_json
from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
from utils import flip, get_latest_checkpoint
from sklearn.metrics import auc

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from llama_engine import run_LRP

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
    parser.add_argument('--model-size', type=str,
                            choices=['llama_2_7b', 'llama_tiny'],
                           required = True,
                            help='')
    parser.add_argument('--variant', type=str,
                           default="baseline")
    parser.add_argument('--fract', type=float,
                            default=0.3,
                            help='')
    parser.add_argument('--pe', action='store_true')
    parser.add_argument('--reform', action='store_true')
    parser.add_argument('--mode', type=str,
                                choices=['pert', 'analyze'],
                                required = True,
                                help='')
    parser.add_argument('--pe_only', action='store_true')
    parser.add_argument('--quant', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clamp', action='store_true')
    parser.add_argument('--reverse_default_abs', action='store_true')


    parser.add_argument('--sep_heads', action='store_true')


    parser.add_argument('--sequence-length', type=int,
                           )
    parser.add_argument('--trained_model', type=str,)

    parser.add_argument('--rule_matmul', action='store_true')


    parser.add_argument('--no-padding', action='store_true')
    parser.add_argument('--without-abs', action='store_true')
    parser.add_argument('--single-norm', action='store_true')





    args = parser.parse_args()
    args.dataset = 'imdb'

    if (args.pe_only and not args.pe) or (args.reform and not args.pe)  or (args.rule_matmul and not args.pe):
        print("no")
        exit(1)

    config.get_config(args, pert=True)
    model_dir = f"pert_results" if args.no_padding == False else f"pert_results/no_padding"
    model_dir = model_dir if args.reverse_default_abs == False else f"{model_dir}/reverse_default_abs"

    if args.single_norm:
        model_dir = f"{model_dir}/abs" if args.without_abs == False else f"{model_dir}/no_abs"


    args.save_dir = f'finetuned_models/{model_dir}/{args.ext}'
    if args.debug:
        args.save_dir = 'finetuned_models/pert_results/debug'
        args.pe = True
        args.reform = True
        args.single_norm = True
        args.without_abs = True
        args.clamp = True



        args.fract = 0.0001


    os.makedirs(args.save_dir, exist_ok=True)
    return args


def analyze(args):
    root_dir = 'finetuned_models'
    model_dirs = ['pert_results', 'pert_results/abs', 'pert_results/no_abs']
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
                        if "ruleMatmul" in path:
                            name = f"{name}+ruleMatmul"
                        if "sepHeads" in path:
                            name = f"{name}+sepHeads"
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    PATH = args.original_models
    MAX_LEN = 512
    BATCH_SIZE = 1

    #if args.model_size == 'llama_tiny': 
    #    model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"
    #if args.model_size == 'llama_2_7b':
    #    model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline/best_checkpoint/pytorch_model.bin"
    #    if args.variant == "baseline2":
    #        model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline2/best_checkpoint/pytorch_model.bin"
    #


    tokenizer = AutoTokenizer.from_pretrained(PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    #original

    if args.quant:
        llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
    else:
        llamaModel = LlamaForSequenceClassification.from_pretrained(PATH,  device_map="cuda",  attn_implementation="eager")
    #torch_dtype=torch.bfloat16,


    conf = llamaModel.config
    conf.num_labels = 2
    conf.pad_token_id = tokenizer.pad_token_id

    #sequence_length = args.sequence_length
    #current
    #kwargs = {"attn_layer": args.attn_layer, "sequence_length": sequence_length }
    #last_checkpoint_dir = get_latest_checkpoint(args.pretrained_model_path)
    #last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 
    print(args.model_checkpoint)
    print(args.save_dir)
    print(args.ext)
    

    model = LlamaForSequenceClassification.from_pretrained(args.model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")

    model.to(device) 




    df = load_imdb()
    df_train, df_test = train_test_split(df, test_size=args.fract, random_state=RANDOM_SEED)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE,args.no_padding)

    # optional gradient checkpointing to save memory (2x forward pass)
    model.gradient_checkpointing_enable()

    # apply AttnLRP rules
    attnlrp.register(model)
    run_LRP(model,
        test_data_loader,
        tokenizer,
        isBinary=True,
        withPE = args.pe,
        reform=args.reform,
        pe_only = args.pe_only,
        withoutABS = args.without_abs,
        clamp = args.clamp,
        sep_heads = args.sep_heads,
        single_norm = args.single_norm,
        rule_matmul = args.rule_matmul,
     
        experimental = args.experimental,
        reverse_default_abs = args.reverse_default_abs,
        save_dir = args.save_dir,

               )

    '''
    UNK_token = tokenizer.unk_token_id
    fracs = np.linspace(0.,1.,11)

    for flip_case in ['generate', 'pruning']:
        all_flips = {}
        all_flips_str = {}

        count = 0
        num_correct = 0

        M, E, E_LOGIT, INSERT_LEAST_IMPORTANT, INSERT_LEAST_IMPORTANT_LOGIT,   EVOLUTION = [],[], [], [],[], []
        M_str,E_str, EVOLUTION_str = [],[], []

        for idx, d in enumerate(tqdm(test_data_loader)):
         
        

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            input_embeds = model.get_input_embeddings()(input_ids)

            if args.pe:
                position_ids = torch.arange(
                        0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
                       dtype=torch.float32
                    ).reshape(1, -1)

                position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for t in range(model.config.num_hidden_layers)]
                position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]

                outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    position_embeddings = position_embeddings,
                    #input_ids=input_ids,
                    use_cache=False,
                    attention_mask=attention_mask
                  )['logits']
            else:
                outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    use_cache=False,
                    attention_mask=attention_mask
                  )['logits']


            max_logits, max_indices = torch.max(outputs, dim=1)
            next_token_id = max_indices.item() 
          
           
            if targets.item() == next_token_id:
                num_correct+=1


            max_logits.backward(max_logits)
            relevance = input_embeds.grad.float().sum(-1).cpu()[0] 
            if args.single_norm:
                pass
            else:
                relevance = relevance / relevance.abs().max()

            if args.pe:
                acc_relevancy = 0.
                for pos_embed in position_embeddings:
                    for i in range(2):
                        #print(pos_embed[i].grad.float().dtype)
                        #print(pos_embed[i].dtype)
                        if args.reform:
                            if args.without_abs:
                                curr_relevancy = torch.matmul(pos_embed[i].grad, pos_embed[i].transpose(-1, -2)).detach()
                            else:
                                curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                            
                            #curr_relevancy /= (2*curr_relevancy.shape[-1])
                            curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                        else:
                            curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]
                        
                        
                        if args.experimental:
                            curr_relevancy = curr_relevancy / curr_relevancy.abs().max()

                        if args.single_norm:
                            pass
                        else:
                            curr_relevancy = curr_relevancy / curr_relevancy.abs().max()
        
                        if args.without_abs:
                            pass
                        else:
                            curr_relevancy = curr_relevancy.abs()
        
                        acc_relevancy += curr_relevancy
        
                if args.clamp:
                    acc_relevancy = acc_relevancy.clamp(min=0)
                if args.single_norm == False:
                    acc_relevancy = acc_relevancy / acc_relevancy.abs().max()
                if args.pe_only:
                    relevance=acc_relevancy.detach()
        
                else:
                    #if args.reform:
                    #    acc_relevancy = acc_relevancy / model.config.num_hidden_layers
                    relevance+=acc_relevancy
                relevance = relevance / relevance.abs().max()




            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        


            
            #tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            #tokens = clean_tokens(tokens)

            #pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')

            





            m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip(model,
                                  x=relevance, 
                                 token_ids=input_ids, 
                                 tokens=tokens,
                                 y_true=targets, 
                                 attention_mask = attention_mask,
                                 fracs=fracs, 
                                 flip_case=flip_case,
                                 tokenizer=tokenizer,
                                 reverse_default_abs = args.reverse_default_abs,
                                 device=device)


            M.append(m)
            E.append(e)

    
            E_LOGIT.append(e_logit)
            if less_relevantArr != []:
                INSERT_LEAST_IMPORTANT.append(less_relevantArr)
                INSERT_LEAST_IMPORTANT_LOGIT.append(less_relevantArr_logit)

            EVOLUTION.append(evolution)


            M_str.append([str(elem) for elem in m])
            E_str.append([str(elem) for elem in e])
            EVOLUTION_str.append(evolution)

            all_flips["res"]= {'E':E, 'M':M, 'E_LOGIT': E_LOGIT, 'INSERT_LEAST_IMPORTANT': INSERT_LEAST_IMPORTANT, 'INSERT_LEAST_IMPORTANT_LOGIT': INSERT_LEAST_IMPORTANT_LOGIT,    'Evolution':EVOLUTION} 
            all_flips_str["res"]= {'E':E_str, 'M':M_str} 



        f, axs = plt.subplots(1, 2, figsize=(14, 8))
        for k, v in all_flips.items():
            print(len(v['M']))
            axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
            axs[0].set_title('($y_0$-$y_p$)$^2$')
            axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
            axs[1].set_title('logits$_k$')    
        plt.legend()

    
        pe = "_pe" if args.pe else ""
        if args.pe_only:
            pe = "_peOnly"
        if args.reform:
            pe = "_peReform"

        if args.experimental:
            pe = f"{pe}_experimental"

        if args.clamp:
            pe = f"{pe}_clamp"
        f.savefig(f'{args.save_dir}/imdb_{flip_case}_{pe}.png' , dpi=300)
        #update_json(f'imdb_{flip_case}.json', all_flips_str)



        if flip_case == "generate":
               update_json(f'{args.save_dir}/imdb_pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)), 'pe': args.pe, 'pe_only': args.pe_only, 'reform': args.reform, 'quant': args.quant, 'single_norm': args.single_norm, 'w.o abs': args.without_abs, 'num_correct': num_correct,
                                            f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0)), f'MERF': auc(fracs, np.nanmean(v['E_LOGIT'], axis=0)), f'LERF': auc(fracs, np.nanmean(v['INSERT_LEAST_IMPORTANT_LOGIT'], axis=0))})
               continue
        update_json(f'{args.save_dir}/imdb_pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),
                                            f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0))})

        #pickle.dump(all_flips, open(os.path.join(args.save_dir, 'all_flips_{}_imdb.p'.format(flip_case)), 'wb'))

    '''


if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "pert":
        eval_pert(args)
    else:
        analyze(args)
       