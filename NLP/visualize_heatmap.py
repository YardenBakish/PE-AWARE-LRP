import transformers
import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig, AdamW, get_linear_schedule_with_warmup
import config

#from lxt.models.llama import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification
from lxt.models.llama_PE import LlamaForCausalLM, attnlrp, LlamaForTokenClassification, LlamaForSequenceClassification

from helper_scripts.helper_functions import update_json
from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader
from tqdm import tqdm
from lxt.utils import pdf_heatmap, clean_tokens
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


import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny', ],
                        default = 'llama_tiny',
                       # required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--resume', action='store_true')
parser.add_argument('--pe', action='store_true')
parser.add_argument('--rule_matmul', action='store_true')

parser.add_argument('--reform', action='store_true')

parser.add_argument('--quant', action='store_true')

parser.add_argument('--trained_model', type=str,)
parser.add_argument('--clamp', action='store_true')


parser.add_argument('--sequence-length', type=int,
                       )
parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--sep_heads', action='store_true')



parser.add_argument('--no-padding', action='store_true')
parser.add_argument('--without-abs', action='store_true')
parser.add_argument('--single-norm', action='store_true')





args = parser.parse_args()

if args.sep_heads and args.pe == False:
    print("for sep_head you must include --pe")
    exit(1)
args.dataset = 'imdb'
#rgs.model_size = 'llama_tiny'

config.get_config(args, pert = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#if args.model_size == 'llama_tiny': 
#    model_checkpoint = "finetuned_models/imdb/llama_tiny/baseline/checkpoint_0/pytorch_model.bin"
#if args.model_size == 'llama_2_7b':
#    model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline/best_checkpoint/pytorch_model.bin"
#    if args.variant == "baseline2":
#        model_checkpoint = "finetuned_models/imdb/llama_2_7b/baseline2/best_checkpoint/pytorch_model.bin"


PATH = args.original_models

#print(PATH)
#exit(1)

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



if args.quant:
    llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, local_files_only = True, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
else:
    llamaModel = LlamaForSequenceClassification.from_pretrained(PATH,  device_map="cuda",   attn_implementation="eager")

#llamaModel = LlamaForSequenceClassification.from_pretrained(PATH, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")

conf = llamaModel.config
conf.num_labels = 2
conf.pad_token_id = tokenizer.pad_token_id


model = LlamaForSequenceClassification.from_pretrained(args.model_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda")
model.to(device)


MAX_LEN = 512
BATCH_SIZE = 1

df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE, args.no_padding)

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
             vis_mode = True,
             debug_mode = False,
             sample_num=4,
             rule_matmul = args.rule_matmul,
               )

