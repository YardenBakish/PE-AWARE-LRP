import os

import config

ANSWERS  = {'A': 0, 'B': 1, 'C': 2,'D': 3, '▁A': 0, '▁B': 1, '▁C': 2,'▁D': 3, 'ĠA':0,'ĠB':1,'ĠC':2,'ĠD':3}

import torch
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from llama_engine import run_LRP

from lxt.models.llama_PE import LlamaForCausalLM, attnlrp
from lxt.utils import pdf_heatmap, clean_tokens
from attDatasets.ai2_arc import load_ai2_arc, AI2_ARC_Dataset, create_data_loader
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny', 'llama_3_8b'],
                        default = 'llama_tiny',
                       # required = True,
                        help='')

parser.add_argument('--pe', action='store_true')
parser.add_argument('--pe_only', action='store_true')
parser.add_argument('--clamp', action='store_true')
parser.add_argument('--debug', action='store_true')

parser.add_argument('--reform', action='store_true')
parser.add_argument('--dataset', type=str,
                           default="arc")
parser.add_argument('--variant', type=str,
                           default="baseline")
parser.add_argument('--without-abs', action='store_true')
parser.add_argument('--single-norm', action='store_true')
parser.add_argument('--quant', action='store_true')

parser.add_argument('--sep_heads', action='store_true')


parser.add_argument('--sequence-length', type=int,
                           )


args = parser.parse_args()
config.get_config(args, pert=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16, # use bfloat16 to prevent numerical overflow
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

if args.quant:
    model = LlamaForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.bfloat16,quantization_config=quantization_config, device_map="cuda",  low_cpu_mem_usage = True)
#model = LlamaForCausalLM.from_pretrained(path, quantization_config=quantization_config, torch_dtype=torch.bfloat16, device_map="cuda")
else:
    model = LlamaForCausalLM.from_pretrained(args.model_checkpoint,torch_dtype=torch.bfloat16,  device_map="cuda")



tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

# optional gradient checkpointing to save memory (2x forward pass)
model.gradient_checkpointing_enable()

# apply AttnLRP rules
attnlrp.register(model)
MAX_LEN = 512
BATCH_SIZE = 1
df = load_ai2_arc()
test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)


run_LRP(
    
    
    model,
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
     
        #experimental = args.experimental,
        skip_if_wrong = False,
        sample_num=30,

        mapper_from_token_to_target= ANSWERS,
        #reverse_default_abs = args.reverse_default_abs,
        #save_dir = args.save_dir,
        #should_keep = args.should_keep,
        dataset="arc",
        vis_mode = True,

    )


