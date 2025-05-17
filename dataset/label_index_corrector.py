

import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from samples.CLS2IDX import CLS2IDX

from torchvision import datasets, transforms


def convertor_dict():
  transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
  
  #from index to cls
  IDX2CLS = {}
  for key, value in CLS2IDX.items():
    IDX2CLS[value] = key

  dataset = datasets.ImageFolder(root="tmp_dataset", transform=transform)

  synset_dict = {}

  with open("dataset/synset_words.txt", 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        synset_id = parts[0]
        category_name = ' '.join(parts[1:])
        synset_dict[synset_id] = category_name

  for x in dataset.classes:
    dataset.class_to_idx[x] = IDX2CLS[synset_dict[x]]
  
  class_lst  = dataset.classes
  last_num = -1
  ptr1 = 0
  ptr2 = 0
  first = True
  d_idx2idx = {}
  while ptr1 < len(dataset.targets):
    if last_num!= dataset.targets[ptr1]:
      last_num = dataset.targets[ptr1]
      if first == False:
        ptr2+=1
      else:
        first = False
      if ptr2 == len(dataset.classes):
        break
    
    d_idx2idx[dataset.targets[ptr1]] = dataset.class_to_idx[dataset.classes[ptr2]]
    ptr1+=1
  return d_idx2idx

  


def correct_label(t, convertor_dictx):
  return torch.tensor([convertor_dictx[t.item()]])



