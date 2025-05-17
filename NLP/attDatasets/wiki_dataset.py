import os 

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
from datasets import load_dataset



def load_wiki_dataset():
    
    df = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]", trust_remote_code=True)

    df= df['text']
    return df



class Wiki_Dataset(Dataset):

  def __init__(self,text, tokenizer, max_len):
    self.text = text
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.text)
  
  def __getitem__(self, item):
    text = str(self.text[item])

  
    encoding = self.tokenizer(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=True,
      #pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      truncation = True
    )

    return {
      'prompt_text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids': encoding['token_type_ids'].flatten(),
      'targets': torch.tensor(1, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size,debug=False):
 
  ds = Wiki_Dataset(
    text =df,
    tokenizer=tokenizer,
    max_len=max_len
  )
  ds = Subset(ds, range(2500))

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )

