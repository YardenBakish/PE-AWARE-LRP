from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch import nn
import copy
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
import torch
import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"




def load_imdb(file ='attDatasets/imdb.csv' ):
    
    df = pd.read_csv(file)
    df['sentiment_score'] = df.sentiment.apply(to_sentiment)
    return df

def to_sentiment(rating):
  rating = str(rating)
  if rating == 'positive':
    return 0
  else: 
    return 1

class MovieReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len, no_padding):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.no_padding = no_padding

  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    #print(review)
 
    target = self.targets[item]

    padding = (self.no_padding == False)
    #print(padding)
    if padding:
      encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation = True
      )
    else:
      encoding = self.tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt',
        
      )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'token_type_ids': encoding['token_type_ids'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size, no_padding=False):
  ds = MovieReviewDataset(
    reviews=df.review.to_numpy(),
    targets=df.sentiment_score.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len,
    no_padding = no_padding
  )

  ds = Subset(ds, range(5000))

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0
  )