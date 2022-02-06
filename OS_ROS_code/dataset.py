import torch.utils.data as data
from PIL import Image
from random import random
import random
import torchvision.transforms.functional as TF
import numpy as np
import torch

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

      img = Image.open(self.data_path+self.names[index])
      img = img.convert('RGB')
      img = self._image_transformer(img)
      
      rand_k = (random.randint(0,3))
      index_rot = rand_k
      img_rot = torch.rot90(img,k=rand_k,dims=[1,2])

      return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)



class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):

      img = Image.open(self.data_path+self.names[index])
      img = img.convert('RGB')
      img = self._image_transformer(img)
      
      rand_k = (random.randint(0,3))
      index_rot = rand_k
      img_rot = torch.rot90(img,k=rand_k,dims=[1,2])

      return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)
