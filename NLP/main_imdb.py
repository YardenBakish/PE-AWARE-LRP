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



import argparse
parser = argparse.ArgumentParser(description='Train a segmentation')
parser.add_argument('--model-size', type=str,
                        choices=['llama_2_7b', 'llama_tiny'],
                        required = True,
                        help='')
parser.add_argument('--variant', type=str,
                       default="baseline")
parser.add_argument('--lr', type=float,
                       default=5e-4)


parser.add_argument('--mixed-lr', type=float,
                       )
parser.add_argument('--batch-size', type=int,
                       default=32)

parser.add_argument('--scheduler', type=str,
                     choices = ['linear', 'cosine'], default= 'linear'  )

parser.add_argument('--sequence-length', type=int,
                       )

parser.add_argument('--backup-interval', type=int, default = 5,
                       )
parser.add_argument('--num_epochs', type=int, default = 200
                       )
parser.add_argument('--num-warmup-steps', type=int,
                     default = 1  )
parser.add_argument('--resume', action='store_true')
parser.add_argument('--finetune', type=str)

parser.add_argument('--eval', action='store_true')
parser.add_argument('--quant', action='store_true')



args = parser.parse_args()
args.dataset = 'imdb'
config.get_config(args)
os.makedirs(f'{args.pretrained_model_path}', exist_ok=True)

PATH = args.original_models



import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
sys.path.append(root_dir)

from attDatasets.imdb import load_imdb, MovieReviewDataset, create_data_loader

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

MAX_LEN = 512
BATCH_SIZE = args.batch_size
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

df = load_imdb()
df_train, df_test = train_test_split(df, test_size=0.3, random_state=RANDOM_SEED)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)


torch.cuda.empty_cache()
class_names = ['negative', 'positive']



compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
sequence_length = args.sequence_length
start_epoch = 0

print(sequence_length)
print(args.num_warmup_steps)


kwargs = {"attn_layer": args.attn_layer, "sequence_length": sequence_length }

if args.quant:
  model = LlamaForSequenceClassification.from_pretrained(PATH, local_files_only = True, torch_dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb_config, attn_implementation="eager")
else: 
  model = LlamaForSequenceClassification.from_pretrained(PATH, local_files_only = True,  device_map="cuda", attn_implementation="eager")


if args.finetune:
 #kwargs["attn_layer"] = args.finetuned_attn_layer
 last_checkpoint_dir = get_latest_checkpoint(args.finetuned_model_path)
 last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 
 conf = model.config
 conf.num_labels = 2
 conf.pad_token_id = tokenizer.pad_token_id
 # model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda", **kwargs)
 model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  device_map="cuda", **kwargs)

if args.resume:
   last_checkpoint_dir = get_latest_checkpoint(args.pretrained_model_path)
   last_checkpoint = f'{last_checkpoint_dir}/pytorch_model.bin' 

   start_epoch = int(last_checkpoint_dir.split("/")[-1].split("_")[-1]) +1
   conf = model.config
   conf.num_labels = 2
   conf.pad_token_id = tokenizer.pad_token_id
   #model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  torch_dtype=torch.bfloat16, device_map="cuda", **kwargs)
   model = LlamaForSequenceClassification.from_pretrained(last_checkpoint, config = conf,  device_map="cuda", **kwargs)
   
   #print("Here")


#model = prepare_model_for_kbit_training(model)

last_epoch = start_epoch -1
   #take last checkpoint


model.get_input_embeddings().requires_grad = False








if args.model_components['components_with_grad']['all'] == False and 'partial_unfreeze' in args.model_components['components_with_grad']:
   args.model_components['components_with_grad']['components'].append("model.norm.weight")
   threshold = int(args.model_components['components_with_grad']['partial_unfreeze'])
   for name, param in model.named_parameters():
      if "model.layers" in name:
        layer_num =  int(name.split(".")[2])
        if threshold <=layer_num:
           args.model_components['components_with_grad']['components'].append(name)
      



for name, param in model.named_parameters(): 
    #param.data = param.data.float()
   
    if args.model_components['components_with_grad']['all'] == False: 
       if name not in args.model_components['components_with_grad']['components']:
          param.requires_grad = False    
       else:
        
          #if param.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
          #  print(param.dtype)
            
            #param.data = param.data.float()
          param.requires_grad = True
        
        # Replace in model
    
          
    else:
      if 'embed' in name:
          param.requires_grad = False
      #else:
      #    if param.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
      #      print(param.dtype)
      #      param.data = param.data.float()
      #    param.requires_grad = True
         
        
model.to(device)




EPOCHS = args.num_epochs - start_epoch

# optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

params_to_train = {}
ignore_params   = {}
for k, v in model.named_parameters():
  if args.model_components['components_with_grad']['all'] == False: 
    if k not in args.model_components['components_with_grad']['components']:
          ignore_params[k] = v  
    else:
          params_to_train[k] = v
  else:
    if 'embed' in k:
      ignore_params[k] = v
    else:  
      params_to_train[k] = v
     






assert set(ignore_params.keys()) != set(params_to_train.keys())
#params_to_train = {k:v for k,v in model.named_parameters() if k=="score.weight"}
#ignore_params = [k for k,v in model.named_parameters() if k!="score.weight"]


if args.mixed_lr:
   attention_params = {'params': [v for k, v in params_to_train.items() if ('attention' in k or 'attn' in k)], 'lr': args.lr}

   non_attention_params = {'params': [v for k, v in params_to_train.items() if ('attention' not in k and 'attn' not in k)], 'lr':args.mixed_lr }

   params_to_train = [attention_params,non_attention_params]
   optimizer = AdamW(params_to_train)

else:
  optimizer = AdamW(params_to_train.values(), lr=args.lr)
total_steps = len(train_data_loader) * EPOCHS

scheduler = None

if args.scheduler == 'linear':

  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.num_warmup_steps * len(train_data_loader),
    num_training_steps=total_steps,

  )

if args.scheduler == 'cosine':
  scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.num_warmup_steps * len(train_data_loader),
    num_training_steps=total_steps,

  )

loss_fn = nn.CrossEntropyLoss().to(device)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.num_labels = 2

if args.resume:
  loaded_checkpoint = torch.load(last_checkpoint, map_location='cpu')

  if 'optimizer' in loaded_checkpoint and 'scheduler' in loaded_checkpoint:
    optimizer.load_state_dict(loaded_checkpoint['optimizer'])
    scheduler.load_state_dict(loaded_checkpoint['lr_scheduler'])
   
model.gradient_checkpointing_enable()


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples,
):
  model = model.train()



  losses = []
  correct_predictions = 0
  import torch.nn.functional as F

  current_loss = 0.0
  for i , d in enumerate(tqdm(data_loader)):
  


    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    token_type_ids = d["token_type_ids"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      #token_type_ids=token_type_ids,
      attention_mask=attention_mask,
      labels = targets
    )['logits']
  
    preds = outputs.argmax(dim=1) #torch.max(outputs, dim=1)
    loss = F.cross_entropy(outputs , targets) # , label_smoothing=0.4)#loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    current_loss += loss.item()
    if np.isnan(current_loss):
        print("there is Nan")

        nan_ids =  np.argwhere(np.isnan(outputs.sum(1).detach().cpu().float().numpy())).squeeze().tolist()
        if  isinstance(nan_ids, int):
            nan_ids = [nan_ids]
        
        for id_ in nan_ids:
            print(tokenizer.convert_ids_to_tokens(input_ids[id_]), targets[id_])
        #import pdb;pdb.set_trace()
    
 
    if i % 100 == 0 and i != 0:
        print('Training Loss is {}'.format(str(current_loss / i)))

    
    if args.mixed_lr:
       all_params = []
       for param_group in optimizer.param_groups:
          all_params.extend(param_group['params'])
       nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

    else:
      nn.utils.clip_grad_norm_(params_to_train.values(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  print("sdfsdfsdf")
  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in tqdm(data_loader):
     
   
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )['logits']

      _, preds = torch.max(outputs, dim=1)

      print(torch.sum(preds == targets))
   

    

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

history = defaultdict(list)
best_accuracy = 0
from tqdm import tqdm
import transformers
transformers.logging.set_verbosity_error()





if args.eval:
    val_acc, val_loss = eval_model(
    model,
    test_data_loader,
    loss_fn, 
    device, 
    len(df_test),

  )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    exit(1)
   

for epoch in tqdm(range(EPOCHS)):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  val_acc, val_loss = eval_model(
    model,
    test_data_loader,
    loss_fn, 
    device, 
    len(df_test),

  )

  print(f'Val   loss {val_loss} accuracy {val_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if (val_acc > best_accuracy):
    save_dir = f'{args.pretrained_model_path}/best_checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict() , f'{save_dir}/pytorch_model.bin')

    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()} , f'{save_dir}/pytorch_model.pth')
    best_accuracy = val_acc
  update_json(f'{args.pretrained_model_path}/acc_results.json',{f"{epoch}_acc":f"{val_acc}" } )