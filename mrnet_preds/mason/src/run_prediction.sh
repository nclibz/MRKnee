#!/bin/sh

#pip install imgaug
#pip install pandas
pip install --upgrade pip
pip install opencv-python-headless==4.1.0.25
pip install albumentations==0.2.3
#pip install torch torchvision



python src/run_prediction.py $1 $2
