
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import numpy as np


IMMUTABLE_TOKENS_SUFFIX = ['Please', '▁make', '▁sure', '▁to', '▁answer', '▁(', 'A', ',', 'B', ',', 'C', ',', '▁or', '▁D', ')', '<0x0A>', 'Answer', '▁is', ':']
IMMUTABLE_TOKENS_PREFIX = ['<s>', '▁The', '▁following', '▁is', '▁multiple', '▁choice', '▁question', '▁(', 'with', '▁answers', ').']

SHOULD_KEEP = True

def softmax(x):
    '''Compute softmax values for each sets of scores in x.'''
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def flip(model, x, token_ids, tokens, y_true,attention_mask,  fracs, flip_case,random_order = False, tokenizer=None,reverse_default_abs = False, device='cpu'):

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

    y0 = model(inputs0, attention_mask=attention_mask, labels = None)['logits'].squeeze().detach().cpu().float().numpy()
    orig_token_ids = np.copy(token_ids.detach().cpu().numpy())

    if flip_case=='generate':
        inds_sorted = np.argsort(x)[::-1]
        if reverse_default_abs: 
            inds_sorted2 =  np.argsort(x)
        else:
            inds_sorted2 =  np.argsort(np.abs(x))

        #inds_sorted2 =  np.argsort(x)


    elif flip_case=='pruning':
        if reverse_default_abs: 
            inds_sorted =  np.argsort(x)
        else:
            inds_sorted =  np.argsort(np.abs(x))
        #inds_sorted =  np.argsort(x)
    else:
        raise
   

    inds_sorted = inds_sorted.copy()
    vals = x[inds_sorted]

    mse = []
    evidence = []
    evidence_logit = []
    evidence_insert_least_relevant = [] #calculated for pruning
    evidence_insert_least_relevant_logit = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().cpu().numpy(), 'y0':y0}

    N=len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':

            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        elif flip_case == 'generate':
            inds_generator2 = iter(inds_sorted2)
            n_flip2=int(np.ceil(frac*N))
            inds_flip2 = [next(inds_generator2) for i in range(n_flip2)]
            inputs = UNK_IDX*torch.ones_like(inputs0)
            inputs2 = UNK_IDX*torch.ones_like(inputs0)
            # Set pad tokens
            inputs[inputs0==0] = 0
            inputs2[inputs2==0] = 0

            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]
            for i in inds_flip2:
                inputs2[:,i] = inputs0[:,i]

        y = model(inputs, attention_mask = attention_mask, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().float().numpy()
        y = y.squeeze()

        err = np.sum((y0-y)**2)
        mse.append(err)
        evidence.append(softmax(y)[int(y_true)])
        evidence_logit.append(y[int(y_true)])

        if flip_case == "generate":
            y2 = model(inputs2, attention_mask = attention_mask, labels =  torch.tensor([y_true]*len(token_ids)).long().to(device))['logits'].detach().cpu().float().numpy()
            y2 = y2.squeeze()
            evidence_insert_least_relevant.append(softmax(y2)[int(y_true)])
            evidence_insert_least_relevant_logit.append(y2[int(y_true)])
      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:
       
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
    return mse, evidence, evidence_logit, model_outs, evidence_insert_least_relevant, evidence_insert_least_relevant_logit






def flip_arc(model, x, highest_idx, logit0, token_ids, tokens, y_true, should_keep,attention_mask,  fracs, flip_case,random_order = False, tokenizer=None,reverse_default_abs = False, correct_subset = False, device='cpu'):
    
    correct_indices = np.array([362, 426, 356, 423])  #A, B,C, D

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

    clear_tokens = tokenizer.convert_ids_to_tokens(inputs0[0])
    start_idx = None
    end_idx = None

    
    for i, elem in enumerate(clear_tokens):
        if 'ĊĊ' in elem and start_idx==None:
            start_idx= (i+1)
        elif ('Ċ' in elem) and (start_idx!=None) and (end_idx==None):
            if clear_tokens[i+1] !='A':
                print("PROBLEM WITH QUESTION")
                print(clear_tokens[i+1])
                exit(1)
            end_idx = i

    y0 = model(inputs0)['logits'][0, -1, :]
    max_logits, max_indices = torch.max(y0, dim=-1)
    y0 = y0.detach().cpu().float().numpy()
    max_logits = max_logits.detach()
    if correct_subset == False:
        y_true = max_indices
    


    if flip_case=='generate':
        inds_sorted = np.argsort(x)[::-1]
        if reverse_default_abs:
            inds_sorted2 =  np.argsort(np.abs(x))
        else:
            inds_sorted2 =  np.argsort(x)


        if should_keep:
            inds_sorted = inds_sorted[(inds_sorted >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted <= (len(inds_sorted) - len(IMMUTABLE_TOKENS_SUFFIX)))]
            inds_sorted2 = inds_sorted2[(inds_sorted2 >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted2 <= (len(inds_sorted2) - len(IMMUTABLE_TOKENS_SUFFIX)))]

            #inds_sorted = inds_sorted[(inds_sorted >= start_idx) & (inds_sorted <= end_idx)]
            #inds_sorted2 = inds_sorted2[(inds_sorted2 >= start_idx) & (inds_sorted2 <= end_idx)]
    elif flip_case=='pruning':
        #CHANGEHERE
        if reverse_default_abs:
            inds_sorted  =  np.argsort(np.abs(x))
        else:
            inds_sorted  =  np.argsort(x)

        if should_keep:
            inds_sorted = inds_sorted[(inds_sorted >= len(IMMUTABLE_TOKENS_PREFIX)) & (inds_sorted <= (len(inds_sorted) - len(IMMUTABLE_TOKENS_SUFFIX)))]
            #inds_sorted = inds_sorted[(inds_sorted >= start_idx) & (inds_sorted <= end_idx)]

    else:
        raise
   

    inds_sorted = inds_sorted.copy()
   
    mse = []
    evidence = []
    evidence_logit = []

    evidence_insert_least_relevant = [] #calculated for pruning
    evidence_insert_least_relevant_logit = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().float().cpu().numpy()}


    #CHANGEHERE

    N= len(x)
    if should_keep:
        N  = N -len(IMMUTABLE_TOKENS_SUFFIX) - len(IMMUTABLE_TOKENS_PREFIX) +1
        #N  = end_idx - start_idx +1


    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)
        #print("here")
        #print(N)
        #print(len(inds_sorted))

        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':
            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

            #CHANGEHERE
            if should_keep:
                inputs[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]
            

        elif flip_case == 'generate':

            inds_generator2 = iter(inds_sorted2)
            n_flip2=int(np.ceil(frac*N))
            inds_flip2 = [next(inds_generator2) for i in range(n_flip2)]
            inputs = UNK_IDX*torch.ones_like(inputs0)
            inputs2 = UNK_IDX*torch.ones_like(inputs0)

            # Set pad tokens
            inputs[inputs0==0] = 0
            inputs2[inputs2==0] = 0


            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

            for i in inds_flip2:
                inputs2[:,i] = inputs0[:,i]
            
            
       
            if should_keep:
                inputs[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]
                inputs2[:,:len(IMMUTABLE_TOKENS_PREFIX)] = inputs0[:,:len(IMMUTABLE_TOKENS_PREFIX)]
                inputs2[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ] = inputs0[:,len(inputs0) - len(IMMUTABLE_TOKENS_SUFFIX) : ]
                
    
        
        y = model(inputs)['logits'][0, -1, :].detach().cpu().float().numpy()

     
 
 
        if correct_subset:
            err = np.sum((y0-y)**2)
            y = y[correct_indices]

        else:
          
            err = np.sum((y0[y_true] - y[y_true])**2)

  
        mse.append(err)



        print(f"LOGITS:\t{softmax(y)[y_true]}")


        evidence.append(softmax(y)[y_true])
        evidence_logit.append(y[y_true])

        if flip_case == "generate":
            if correct_subset:
                y2 = model(inputs2)['logits'][0, -1, :][correct_indices].detach().cpu().float().numpy()
            else:
                y2 = model(inputs2)['logits'][0, -1, :].detach().cpu().float().numpy()

            evidence_insert_least_relevant.append(softmax(y2)[y_true])
            evidence_insert_least_relevant_logit.append(y2[y_true])



      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().float().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:


        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
   
    return mse, evidence, evidence_logit, model_outs, evidence_insert_least_relevant, evidence_insert_least_relevant_logit















def flip_wiki(model, x, highest_idx, logit0, token_ids, tokens, y_true, attention_mask,  fracs, flip_case,random_order = False, tokenizer=None,reverse_default_abs = False,  device='cpu'):
    

    x = np.array(x)

    UNK_IDX = tokenizer.unk_token_id
    inputs0 = torch.tensor(token_ids).to(device)

    clear_tokens = tokenizer.convert_ids_to_tokens(inputs0[0])
    #print(clear_tokens)
   
    y0 = model(inputs0)['logits'][0, -1, :]
    max_logits, max_idx = torch.max(y0, dim=-1)
    y0 = y0.detach().cpu().float().numpy()
    max_logits = max_logits.detach()



    if flip_case=='generate':
        inds_sorted = np.argsort(x)[::-1]
        if reverse_default_abs:
            inds_sorted2 =  np.argsort(np.abs(x))
        else:
            inds_sorted2 =  np.argsort(x)


    elif flip_case=='pruning':
        #CHANGEHERE
        if reverse_default_abs:
            inds_sorted  =  np.argsort(np.abs(x))
        else:
            inds_sorted  =  np.argsort(x)

    else:
        raise
   

    inds_sorted = inds_sorted.copy()
   
    mse = []
    evidence = []
    evidence_logit = []

    evidence_insert_least_relevant = [] #calculated for pruning
    evidence_insert_least_relevant_logit = []
    model_outs = {'sentence': tokens, 'y_true':y_true.detach().float().cpu().numpy()}


    N= len(x)

    evolution = {}
    for frac in fracs:
        inds_generator = iter(inds_sorted)


        n_flip=int(np.ceil(frac*N))
        inds_flip = [next(inds_generator) for i in range(n_flip)]

        if flip_case == 'pruning':
            inputs = inputs0
            for i in inds_flip:
                inputs[:,i] = UNK_IDX

        elif flip_case == 'generate':

            inds_generator2 = iter(inds_sorted2)
            n_flip2=int(np.ceil(frac*N))
            inds_flip2 = [next(inds_generator2) for i in range(n_flip2)]
            inputs = UNK_IDX*torch.ones_like(inputs0)
            inputs2 = UNK_IDX*torch.ones_like(inputs0)

            # Set pad tokens
            inputs[inputs0==0] = 0
            inputs2[inputs2==0] = 0


            for i in inds_flip:
                inputs[:,i] = inputs0[:,i]

            for i in inds_flip2:
                inputs2[:,i] = inputs0[:,i]
            
            
        
        y = model(inputs)['logits'][0, -1, :].detach().cpu().float().numpy()
        err = np.sum((y0[max_idx] - y[max_idx])**2)

        mse.append(err)



        print(f"LOGITS:\t{softmax(y)[max_idx]}")


        evidence.append(softmax(y)[max_idx])
        evidence_logit.append(y[max_idx])

        if flip_case == "generate":
            y2 = model(inputs2)['logits'][0, -1, :].detach().cpu().float().numpy()
            evidence_insert_least_relevant.append(softmax(y2)[max_idx])
            evidence_insert_least_relevant_logit.append(y2[max_idx])



      #  print('{:0.2f}'.format(frac), ' '.join(tokenizer.convert_ids_to_tokens(inputs.detach().cpu().numpy().squeeze())))
        evolution[frac] = (inputs.detach().cpu().float().numpy(), inds_flip, y)

    if flip_case == 'generate' and frac == 1.:


        #print(f"RESULT: {np.trapz(fracs,evidence)}" )
        assert (inputs0 == inputs).all()


    model_outs['flip_evolution']  = evolution
   
    return mse, evidence, evidence_logit, model_outs, evidence_insert_least_relevant, evidence_insert_least_relevant_logit






def get_latest_checkpoint(path):
    import os, re
    checkpoints = [d for d in os.listdir(path) if d.startswith('checkpoint_')]
    if not checkpoints:
        print("no directory here")
        exit(1)
    return os.path.join(path, max(checkpoints, key=lambda x: int(re.search(r'checkpoint_(\d+)', x).group(1))))