
import os
import cv2
from PIL import Image
import PIL
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import torchvision.utils


def LoadImage(file_dir):
  file_name_list = sorted(os.listdir(file_dir))
  image_list = []
  for file_name in file_name_list:
    file_path = os.path.join(file_dir, file_name)
    file_image = Image.open(file_path)
    image_list.append(file_image)
  return image_list

def LoadMulImage(file_dirs):
  image_list = []
  for file_dir in file_dirs:
    image_list += LoadImage(file_dir)
  return image_list

class ImageDataset():
  def __init__(self, label_dir, input_dir, trans):
    self.input_images = LoadImage(input_dir)
    self.label_images = LoadImage(label_dir)
    self.transform = trans

  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]
    label = self.label_images[idx]
    if self.transform:
      image = self.transform(image)
      label = self.transform(label)  
    return image, label


class ImageDataset2():
  def __init__(self, label_dir, input_dirs, trans):
    self.input_images = LoadMulImage(input_dirs)
    self.len = len(self.input_images)
    self.label_images = LoadImage(label_dir)*self.len
    self.transform = trans

  def __len__(self):
    return len(self.input_images)

  def __getitem__(self, idx):
    image = self.input_images[idx]
    label = self.label_images[idx]

    image = self.transform(image)
    label = self.transform(label)# turn into grayscale image in 256 size
    label2 = 1-label

    return image, torch.cat([label, label2], dim=0)