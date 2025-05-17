
import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import config
import os
from utils import get_latest_checkpoint
from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from helper_scripts.helper_functions import update_json
#from peft import prepare_model_for_kbit_training
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


def model_env(args  = None, tokenizer =None ):
    if "llama" in args.variant:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    kwargs = {"attn_layer": args.attn_layer, "sequence_length": args.sequence_length }
    model = LlamaForSequenceClassification.from_pretrained(args.original_models, local_files_only = True,  device_map="cuda", attn_implementation="eager")

    if args.finetune:
     last_checkpoint_dir = get_latest_checkpoint(args.finetuned_model_path)
     last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 
     conf = model.config
     conf.num_labels = 2
     conf.pad_token_id = tokenizer.pad_token_id
     model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  device_map="cuda", **kwargs)

    if args.resume:
       last_checkpoint_dir = get_latest_checkpoint(args.pretrained_model_path)
       last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 

       start_epoch = int(last_checkpoint_dir.split("/")[-1].split("_")[-1]) +1
       conf = model.config
       conf.num_labels = 2
       conf.pad_token_id = tokenizer.pad_token_id
       model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  device_map="cuda", **kwargs)
    
    model.get_input_embeddings().requires_grad = False
    return model
        


    
