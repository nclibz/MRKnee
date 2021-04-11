import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable

INPUT_DIM = 256
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


class Dataset(data.Dataset):
    def __init__(self, paths, task, view, use_gpu):
        super().__init__()
        self.use_gpu = use_gpu
        self.view = view

        label_dict = {}
        self.paths = []

        self.paths = [p for p in paths if view in p]

        # for i, line in enumerate(open('data/'+split+'-'+task+'.csv').readlines()):
        #     #if i == 0:
        #     #    continue
        #     line = line.strip().split(',')
        #     path = line[0]
        #     label = line[1]
        #     label_dict[path] = int(label)
        #     self.paths.append(path)

        # self.labels = [label_dict[path] for path in self.paths]
        self.labels = [0. for path in paths]
        
        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        # path = 'data/volumes-'+self.view+'/' + self.paths[index] + '.npy'
        path = self.paths[index]
        with open(path, 'rb') as filehandler:
            vol = np.load(filehandler).astype(np.int32)
        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM)/2)
        if pad != 0:
            vol = vol[:,pad:-pad,pad:-pad]
        
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL

        # normalize
        vol = (vol - MEAN) / STDDEV
        
        # convert to RGB
        vol = np.stack((vol,)*3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)

def load_data(paths, task, view, shuffle=False, use_gpu=False):
    
    dataset = Dataset(paths, task, view, use_gpu)

    loader = data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=shuffle)

    return loader


