import importlib
import os
import random
import re
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision

import albumentations as img_aug
from config import config


INPUT_DIM = 256
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


def aug_img(im, transform):
  im = np.transpose(im, [1, 2, 0])
  im = transform(image=im)['image']
  im = np.transpose(im, [2, 0, 1])
  return im


def normalize3(vol, rgb=True, transform=None):
  pad = int((vol.shape[2] - INPUT_DIM)/2)
  if pad != 0:
    vol = vol[:,pad:-pad,pad:-pad]

  if transform:
    vol =aug_img(vol, transform)

  vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
  # normalize
  vol = (vol - MEAN) / STDDEV
  # convert to RGB
  if rgb:
    vol = np.stack((vol,) * 3, axis=1)
  else:
    vol = np.expand_dims(vol, 1)
  return vol


class Dataset(data.Dataset):

  def __init__(self, file_list, rgb=True, transform=None, cat='all'):
    super().__init__()
    self.file_list = file_list
    if cat == 'all':
      self.category = ['abnormal', 'acl', 'meniscus']
    else:
      self.category = [cat]
    self.img_type = ['axial', 'coronal', 'sagittal']
    self.rgb = rgb
    self.transform = transform

  def __getitem__(self, index):
    start = index * 3
    data_item = {}
    for i in range(3):
      path = self.file_list[start + i]
      im_type = path.split('/')[-2]
      with open(path, 'rb') as f:
        vol = np.load(f).astype(np.float32)
        data_item[im_type] = normalize3(vol, self.rgb, self.transform)

    return {'data': data_item}

  def __len__(self):
    return len(self.file_list) // 3


def to_device(data, device):
  if isinstance(data, dict):
    return {k: v.to(device) for k, v in data.items()}
  else:
    return data.to(device)


def get_data_loader(file_list, rgb=True, transform=None, cat='all'):
  dataset = Dataset(file_list, rgb=rgb, transform=transform, cat=cat)
  #loader = data.DataLoader(dataset, batch_size=1, num_workers=1,
      #worker_init_fn=lambda x: np.random.seed(
        #int(torch.initial_seed() + x) % (2 ** 32 - 1)),
      #shuffle=False)
  def loader():
    """Somehow the dataloader does not work in the docker image."""
    for i in range(len(dataset)):
      data = dataset[i]['data']
      data = {k: torch.from_numpy(v[None]) for k, v in data.items()}
      yield {'data': data}
  return loader()


def evaluate(model, loader, n_round=5, use_gpu=True, im_type='axial'):
  model.eval()
  prediction_list = []
  for i in range(n_round):
    preds = []
    for batch in loader:
      vol = batch['data']
      if use_gpu:
        vol = to_device(vol, 'cuda:0')

      with torch.no_grad():
        pred = model(vol, im_type)
      pred_npy = pred.data.cpu().numpy()

      preds.append(pred_npy)

    preds_np = np.concatenate(preds, axis=0)
    prediction_list.append(preds_np)
  avg_pred = np.mean(np.stack(prediction_list, axis=0), axis=0)

  return avg_pred


def run_prediction(config):
  file_list = pd.read_csv(config['in_file_path'], header=None)[0]
  assert len(file_list) % 3 == 0, 'length of file must be multiple of 3'

  cat = 'all'
  if 'cat' in config:
    cat = config['cat']

  data_loader = get_data_loader(file_list,
                               rgb=config['rgb'],
                               transform=config['transform'], cat=cat)

  model = config['model']
  state_dict = torch.load(config['state_dict'],
      map_location=(None if config['use_gpu'] else 'cpu'))
  model.load_state_dict(state_dict)

  if config['use_gpu']:
    model = model.cuda()
  prediction = evaluate(model, data_loader, config['n_round'],
      config['use_gpu'], config['im_type'])
  pd.DataFrame(data=prediction).to_csv(config['out_file_path'], index=False,
      header=False)


def main():
  config['in_file_path'] = sys.argv[1]
  config['out_file_path'] = sys.argv[2]
  np.random.seed(config['seed'])
  torch.manual_seed(config['seed'])
  random.seed(config['seed'])

  if config['use_gpu']:
    torch.cuda.manual_seed_all(config['seed'])

  run_prediction(config)


if  __name__ == '__main__':
  main()
