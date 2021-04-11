import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import glob
import time
import numpy as np
import keras
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import backend as K

IMG_TYPES = ['sagittal', 'coronal', 'axial']

def load_resnet(is_src):
  p = 'models/resnet50.h5'
  if is_src:
    p = 'src/' + p
  return keras.models.load_model(p)

def load_softmax(model_type, is_src):
  #return keras.models.load_model('models/{}.h5'.format(model_type))
  p = 'models/{}.h5'.format(model_type)
  if is_src:
    p = 'src/' + p
  return keras.models.load_model(p)

def mk_softmax(model_type, is_src):
  p = 'models/{}_wts.h5'.format(model_type)
  if is_src:
    p = 'src/' + p
  model = Sequential()
  model.add(Dense(1, activation='sigmoid', input_dim=2048*3))
  model.load_weights(p)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  return model

def get_resnet_output(resnet, paths):
  samples = int(len(paths)/3)
  output = np.zeros((samples, 2048*3))
  for i in range(samples):
    this_output = []
    for j in range(3):
      batch = np.load(paths[ i * 3 + j ])
      batch = np.repeat(batch[:, :, :, np.newaxis], 3, axis=3)
      batch = batch.astype('float64')
      print(batch.dtype)
      batch = preprocess_input(batch)
      pre_output = np.amax(resnet.predict_on_batch(batch), axis=0)
      this_output.append(pre_output)
      print(paths[i*3+j])
      print(i, pre_output.shape)
    this_output.reverse()
    output[i] = np.concatenate(this_output)
    print(i, output[i].shape)
  return output


def read_input_paths(fname):
  with open(fname, 'r') as f:
    return [line.strip() for line in f.readlines()]

def mk_predictions(models, inputs):
  probs = np.hstack([model.predict(inputs) for model in models])
  return probs

def main(args):
  is_src = True
  output_csv = args[2]
  input_paths = read_input_paths(args[1])
  model_types = ['abnormal', 'acl', 'meniscus']
  models = [mk_softmax(x, is_src) for x in model_types]
  print(models)
  output = get_resnet_output(load_resnet(is_src), input_paths)
  preds = mk_predictions(models, output)
  print(preds)
  np.savetxt(args[2], preds, delimiter=',')


if __name__ == '__main__':
  main(sys.argv)
