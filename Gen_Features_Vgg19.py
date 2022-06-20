#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from keras.layers import Input
from keras.datasets.cifar10 import load_data
from keras.models import load_model
from keras.layers import Dense
import seaborn as sns
import pandas as pd
# from utils import *
import numpy as np
import cv2
from PIL import Image
from keras.layers.core import Lambda
# from Model_Load import Model_load
from keras.utils import to_categorical
from keras.models import load_model
import keras.backend as KTF
import matplotlib.pyplot as plt
import foolbox
import os
import scipy
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import argparse
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
parser = argparse.ArgumentParser(description='input flags')
parser.add_argument('--flags', type=int,
                    help='running loop')
args = parser.parse_args()
#%%
(x_train, y_train), (x_test, y_test) = load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
x_train = x_train/255.0
x_test = x_test/255.0
model = load_model('/data0/jinhaibo/DGAN/train_model/cifar10_vgg.h5')
advs = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/FGSM/adv.npy')
#%%
fa_values_adv = []
layers_names = 'block3_conv2'
print('in', args.flags)
flags = args.flags
for i in range(50):
    x_adv = x_train[i + flags * 50:i+1 + flags * 50]
    predcitions_adv = np.argmax(model.predict(x_adv))
    output_label = model.output[:, predcitions_adv]
    grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.get_layer(layers_names).output])
    weights_adv, out_value_adv = iterate([x_adv])
    fc_values_adv = out_value_adv
    if i % 10 == 0:
        print(i)
    for i in range(len(weights_adv)):
        fc_values_adv[:, i] = weights_adv[i]
    fa_values_adv.append(fc_values_adv)
trian_x_adv = np.array(fa_values_adv).reshape(len(fa_values_adv), -1)

np.save('/data0/jinhaibo/DGAN/AdvChecker/Features/CIFAR_VGG19/Inf_layers/conv2_benign/feature_{}.npy'.format(flags), trian_x_adv)