import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config
from utils import flip_arc, flip, get_latest_checkpoint , flip_wiki
import matplotlib.pyplot as plt
from sklearn.metrics import auc

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

from helper_scripts.helper_functions import update_json
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens


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


def run_LRP(model, data_loader, 
            tokenizer, 
            isBinary=True, 
            withPE=False, 
            reform=False, 
            pe_only=False, 
            withoutABS= False, 
            clamp=False, 
            sep_heads=False, 
            single_norm=False, 
            vis_mode=False, 
            debug_mode=False, 
            sample_num=None,
            experimental = False,
            skip_if_wrong = False,
            mapper_from_token_to_target = None,
            reverse_default_abs = False,
            rule_matmul         = False,
  
            save_dir = None,
            should_keep = False,
            dataset= "imdb",
            eval = False,
            correct_subset = False
            ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    count = 0

    UNK_token = tokenizer.unk_token_id
    fracs = np.linspace(0.,1.,11)

    for flip_case in ['generate', 'pruning']:
        all_flips = {}
        all_flips_str = {}

        count = 0
        num_correct = 0
        total_samples_num=0

        M, E, E_LOGIT, INSERT_LEAST_IMPORTANT, INSERT_LEAST_IMPORTANT_LOGIT,   EVOLUTION = [],[], [], [],[], []
        M_str,E_str, EVOLUTION_str = [],[], []

        for d in tqdm(data_loader):

            if num_correct == 1000:
                break
            
            total_samples_num+=1
            if vis_mode and count !=sample_num:
                count+=1
                continue
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            input_embeds = model.get_input_embeddings()(input_ids)

            if withPE:
                position_ids = torch.arange(
                        0.0, input_ids.shape[1], device=input_embeds.device,  requires_grad=True,
                       dtype=torch.float32
                    ).reshape(1, -1)

                if sep_heads:
                    position_embeddings = []
                    for i in range(model.config.num_hidden_layers):
                        pe_tuple = model.get_input_pos_embeddings()(input_embeds, position_ids)
                        pe1_a = pe_tuple[0].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                        pe1_b = pe_tuple[0].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()

                        pe2_a = pe_tuple[1].repeat(model.config.num_attention_heads, 1, 1).requires_grad_()
                        pe2_b = pe_tuple[1].repeat(model.config.num_key_value_heads, 1, 1).requires_grad_()

                        position_embeddings.append((pe1_a,pe1_b,pe2_a,pe2_b))

                    #position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids)[0] for i in range(model.config.num_hidden_layers)]

                else:
                    position_embeddings = [model.get_input_pos_embeddings()(input_embeds, position_ids) for i in range(model.config.num_hidden_layers)]
                    position_embeddings = [(x[0].requires_grad_(),x[1].requires_grad_()) for x in  position_embeddings ]
                    identity_matrices   = None
                    if rule_matmul:
                        identity_matrices = model.identity_matrices
                        identity_matrices = [x.requires_grad_() for x in identity_matrices] 


                outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    position_embeddings = position_embeddings,
                    sep_heads = sep_heads,
                    identity_matrices = identity_matrices,

                    #input_ids=input_ids,
                    use_cache=False,
                    attention_mask=attention_mask if isBinary else None
                  )['logits']

            else:
                outputs = model(
                    inputs_embeds = input_embeds.requires_grad_(),
                    use_cache=False,
                    attention_mask=attention_mask if isBinary else None
                  )['logits']

            if isBinary:
                max_logits, max_indices = torch.max(outputs, dim=1)
            else:
                if correct_subset:
                    correct_indices = torch.tensor([362, 426, 356, 423]).to(device)
                    
                    _, max_idx = torch.max(outputs[0, -1, :],  dim=-1)
                    if max_idx not in correct_indices:
                        print("ERROR")
                        exit(1)
                  

                    max_logits, max_indices = torch.max(outputs[0, -1, :][correct_indices], dim=-1)
                    #max_indices = correct_indices[max_indices]
                else:
                    max_logits, max_indices = torch.max(outputs[0, -1, :], dim=-1)

     
            next_token_id = max_indices.item() 
           
            if eval:
                #if total_samples_num > 500:
                #    break
                if isBinary:
                    if targets.item() == next_token_id:
                        num_correct+=1
                else:
                     next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
                     if len(next_token)>1 and dataset=="arc":
                         next_token = next_token[-1]
                     #print(next_token_id)
                     #print(next_token)
                     #print(len(next_token))

                     #print(targets.item())
                     #print("------------------")
                     if mapper_from_token_to_target:
                         if next_token not in mapper_from_token_to_target:
                             continue
                         next_token = mapper_from_token_to_target[next_token]
                     if (targets.item() !=  next_token):
                         continue
                     num_correct+=1
                    
                continue    
            
            
            if isBinary:
                if targets.item() == next_token_id:
                    num_correct+=1

            else:
                next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
                if len(next_token)>1 and dataset=="arc":
                    next_token = next_token[-1]

                #print(next_token_id)
                
                #print(next_token)
                #print(targets.item())
            
                #print("------------------")
                if skip_if_wrong:
                  
                    if mapper_from_token_to_target:
                        if (next_token not in mapper_from_token_to_target):
                            continue
                        next_token = mapper_from_token_to_target[next_token]
                    if (targets.item() !=  next_token):
                        continue
           
                    num_correct+=1


            max_logits.backward(max_logits)

            relevance = input_embeds.grad.float().sum(-1).cpu()[0] # cast to float32 before summation for higher precision
            
            if single_norm:
                pass
            else:
                relevance = relevance / relevance.abs().max()

            if withPE:
                acc_relevancy = 0.
                num_pos_embed_layer = 4 if sep_heads else 2
                for t, pos_embed in enumerate(position_embeddings):
                    if rule_matmul:
                        acc_relevancy+=(identity_matrices[t].grad.float().diagonal().cpu() ) # / model.config.num_hidden_layers
                        #print(acc_relevancy.shape)
                        continue
                    for i in range(num_pos_embed_layer):
                        #print(pos_embed[i].grad.float().dtype)
                        #print(pos_embed[i].dtype)
                        if reform:
                            if withoutABS:
                                if clamp:
                                    curr_relevancy = torch.matmul(pos_embed[i].grad.clamp(min=0), pos_embed[i].transpose(-1, -2).clamp(min=0)).detach()
                                else:
                                    curr_relevancy = torch.matmul(pos_embed[i].grad, pos_embed[i].transpose(-1, -2)).detach()
                            else:
                                curr_relevancy = torch.matmul(pos_embed[i].grad.abs(), pos_embed[i].transpose(-1, -2).abs()).detach()
                            #curr_relevancy /= (2*curr_relevancy.shape[-1])

                            curr_relevancy = curr_relevancy.float().sum(-1).cpu()[0]
                        else:
                            curr_relevancy = pos_embed[i].grad.float().sum(-1).cpu()[0]

                        #curr_relevancy = curr_relevancy / curr_relevancy.abs().max()

                        if experimental:
                            curr_relevancy = curr_relevancy / curr_relevancy.abs().max()


                        if single_norm:
                            pass
                        else:
                            curr_relevancy = curr_relevancy / curr_relevancy.abs().max()

                        if withoutABS:
                            pass
                        else:
                            curr_relevancy = curr_relevancy.abs()

                        acc_relevancy += curr_relevancy

                if clamp:
                    acc_relevancy = acc_relevancy.clamp(min=0)

                #acc_relevancy = acc_relevancy.abs()
                if single_norm == False:
                    acc_relevancy = acc_relevancy / acc_relevancy.abs().max()
                if pe_only:
                    relevance=acc_relevancy.detach()

                else:
                    #if reform:
                    #    acc_relevancy = acc_relevancy / model.config.num_hidden_layers
                    relevance+=acc_relevancy
                relevance = relevance / relevance.abs().max()

            if vis_mode==True:
                next_token = tokenizer.convert_ids_to_tokens(next_token_id) 
                print(next_token)
                print(targets.item())
                print("------------------")
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                tokens = clean_tokens(tokens)
            
                #num_elements = int(relevance.numel() * 0.8)
                #top_values, _ = torch.topk(relevance, num_elements)
                #top_values = top_values[-1].item()
                #relevance[relevance>top_values] = 1
                #relevance=relevance[:-3]
                #tokens   =tokens[:-3]

                pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')   
                exit(1)


            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            
            if dataset == "arc":
                m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip_arc(model,
                    x=relevance, 
                    highest_idx = next_token_id,
                    logit0 = max_logits.detach().item(),
                    token_ids=input_ids, 
                    tokens=tokens,
                    y_true=targets, 
                    should_keep = should_keep,
                    attention_mask = attention_mask,
                    fracs=fracs, 
                    flip_case=flip_case,
                    tokenizer=tokenizer,
                    reverse_default_abs = reverse_default_abs,
                    correct_subset = correct_subset,
                    device=device,)
            
            elif dataset == "imdb":
                m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip(model,
                                      x=relevance, 
                                     token_ids=input_ids, 
                                     tokens=tokens,
                                     y_true=targets, 
                                     attention_mask = attention_mask,
                                     fracs=fracs, 
                                     flip_case=flip_case,
                                     tokenizer=tokenizer,
                                     reverse_default_abs = reverse_default_abs,
                                     device=device)

            elif dataset == "wiki":
                m, e, e_logit, evolution, less_relevantArr, less_relevantArr_logit = flip_wiki(model,
                    x=relevance, 
                    highest_idx = next_token_id,
                    logit0 = max_logits.detach().item(),
                    token_ids=input_ids, 
                    tokens=tokens,
                    y_true=targets, 
                    attention_mask = attention_mask,
                    fracs=fracs, 
                    flip_case=flip_case,
                    tokenizer=tokenizer,
                    reverse_default_abs = reverse_default_abs,
                    device=device,)
                

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


        if eval:
            update_json(f'{save_dir}/eval.json', {'num_correct': num_correct, 'samples_num': total_samples_num,})
            exit(1)
        f, axs = plt.subplots(1, 2, figsize=(14, 8))
        for k, v in all_flips.items():
            print(len(v['M']))
            axs[0].plot(np.nanmean(v['M'], axis=0), label=k)
            axs[0].set_title('($y_0$-$y_p$)$^2$')
            axs[1].plot(np.nanmean(v['E'], axis=0), label=k)
            axs[1].set_title('logits$_k$')    
        plt.legend()

    
        pe = "_pe" if withPE else ""
        if pe_only:
            pe = "_peOnly"
        if reform:
            pe = "_peReform"
        if rule_matmul:
            pe = "_peruleMatmul"
        
        if experimental:
            pe = f"{pe}_experimental"
        if clamp:
            pe = f"{pe}_clamp"
        if sep_heads:
            pe = f"{pe}_sepHeads"
        f.savefig(f'{save_dir}/logits_{flip_case}_{pe}.png' , dpi=300)
        #update_json(f'imdb_{flip_case}.json', all_flips_str)



        if flip_case == "generate":
               update_json(f'{save_dir}/pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),rule_matmul:rule_matmul, 'pe': withPE, 'pe_only': pe_only, 'reform': reform,  'single_norm': single_norm, 'w.o abs': withoutABS, 'num_correct': num_correct, 'samples_num': total_samples_num,
                                            f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0)), f'MERF': auc(fracs, np.nanmean(v['E_LOGIT'], axis=0)), f'LERF': auc(fracs, np.nanmean(v['INSERT_LEAST_IMPORTANT_LOGIT'], axis=0))})
               continue
        update_json(f'{save_dir}/pert_res_{pe}.json', {f'{flip_case}_AU_MSE': auc(fracs, np.nanmean(v['M'], axis=0)),
                                            f'{flip_case}_AU_AC': auc(fracs, np.nanmean(v['E'], axis=0))})

        #pickle.dump(all_flips, open(os.path.join(args.save_dir, 'all_flips_{}_imdb.p'.format(flip_case)), 'wb'))


