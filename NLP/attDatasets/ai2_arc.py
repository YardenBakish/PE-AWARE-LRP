import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
from datasets import load_dataset

import pyarrow.parquet as pq


def load_ai2_arc(file = 'ai2_arc/ARC-Easy/train-00000-of-00001.parquet'):
    
    df = load_dataset("allenai/ai2_arc", "ARC-Easy")

    df1= df['train'].to_pandas()
    df2= df['test'].to_pandas()
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['answerKey'].isin(['A', 'B', 'C', 'D'])]
    df["answerKey"] = df["answerKey"].map({'A': 0, 'B': 1, 'C': 2,'D': 3})
    #print(df["answerKey"])
    return df



class AI2_ARC_Dataset(Dataset):

  def __init__(self, question, choices, answerKey, tokenizer, max_len):
    self.question = question
    self.choices = choices
    self.answerKey = answerKey

    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.question)
  
  def __getitem__(self, item):
    question = str(self.question[item])
    choice = self.choices[item]["text"]
    answerKey = self.answerKey[item]
    #print(question)
    #print(choice)
    #print(answerKey)

    #prompt = f"Question: {question}\n\nPlease choose from the following (A,B,C or D):\nA. {choice[0]}\nB. {choice[1]}\nC. {choice[2]}\nD. {choice[3]}"

    #print(prompt)
    options = ""
    for i in range(len(choice)):
      options+=f"\n{chr(65 + i)}. {choice[i]}"
    prompt = f"The following is multiple choice question (with answers).\n\n{question}{options}\n\nPlease make sure to answer (A,B,C, or D)\nAnswer is:"
    #print(prompt)
    encoding = self.tokenizer(
      prompt,
      add_special_tokens=True,
      #max_length=self.max_len,
      return_token_type_ids=True,
      #pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
      #truncation = True
    )

    return {
      'prompt_text': prompt,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids': encoding['token_type_ids'].flatten(),
      'targets': torch.tensor(answerKey, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size,debug=False):
  ds = AI2_ARC_Dataset(
    question =df.question.to_numpy(),
    choices =df.choices.to_numpy(),
    answerKey =df.answerKey.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  if debug:
    ds = Subset(ds, range(1500))

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )

