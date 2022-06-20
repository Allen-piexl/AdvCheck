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
tf.compat.v1.disable_eager_execution()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%% CIFAR_VGG19_3.h5 is the best
model = load_model('/data0/jinhaibo/DGAN/AdvChecker/META_Models/lenet.h5')
#load benign
benign_path = '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/Benign/benign_feature_'
for i in range(80):
    if i == 0:
        tmp_benign = np.load(benign_path + '{}.npy'.format(i))
    else:
        tmp_benign = np.vstack((tmp_benign, np.load(benign_path + '{}.npy'.format(i))))
benign_label = np.array([np.array([1, 0]) for i in range(4000)])
#%%load adv
# for m in range(9):
for m in [6]:
    attack_list = ['FGSM', 'BIM', 'JSMA', 'PGD', 'AUNA', 'PWA', 'Boundary']
    attack_path = ['/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/FGSM/fgsm_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/BIM/bim_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/JSMA/jsma_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/PGD/pgd_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/AUNA/auna_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/PWA/pwa_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/lenet/Boundary/Boundary_feature_']
    print('Running {}'.format(attack_list[m]))
    for i in range(7):
        if i == 0:
            tmp = np.load(attack_path[m]+'{}.npy'.format(i))
        else:
            tmp = np.vstack((tmp, np.load(attack_path[m]+'{}.npy'.format(i))))

test_y = np.array([np.array([0, 1]) for i in range(tmp.shape[0])])
#%%test acc
num = test_y.shape[0]
tmp_benign = tmp_benign[:num]
benign_label = benign_label[:num]
total_img = np.vstack((tmp_benign, tmp))
total_label = np.vstack((benign_label, test_y))
scores = model.evaluate(total_img, total_label, batch_size=32)
print(scores[1])
#%%
detect_pred = model.predict(total_img)
from sklearn.metrics import roc_auc_score
roc_value = roc_auc_score(total_label, detect_pred)
print("AUC:", roc_value)
#%% Boundary
model = load_model('/data0/jinhaibo/DGAN/AdvChecker/META_Models/CIFAR_VGG19_3.h5')
benign_path = '/data0/jinhaibo/DGAN/AdvChecker/Features/CIFAR_VGG19/Benign/benign_feature_'
for i in range(8):
    if i == 0:
        tmp_benign = np.load(benign_path + '{}.npy'.format(i))
    else:
        tmp_benign = np.vstack((tmp_benign, np.load(benign_path + '{}.npy'.format(i))))
benign_label = np.array([np.array([1, 0]) for i in range(400)])
for m in range(1):
    attack_list = ['Boundary']
    attack_path = ['/data0/jinhaibo/DGAN/AdvChecker/Features/CIFAR_VGG19/Boundary/boundary_feature_']
    print('Running{}'.format(attack_list[m]))
    for i in range(8):
        if i == 0:
            tmp = np.load(attack_path[m]+'{}.npy'.format(i))
        else:
            tmp = np.vstack((tmp, np.load(attack_path[m]+'{}.npy'.format(i))))

    test_y = np.array([np.array([0, 1]) for i in range(400)])
    total_img = np.vstack((tmp_benign, tmp))
    total_label = np.vstack((benign_label, test_y))
    scores = model.evaluate(total_img, total_label, batch_size=32)
    print(scores[1])
    detect_pred = model.predict(total_img)
    from sklearn.metrics import roc_auc_score

    roc_value = roc_auc_score(total_label, detect_pred)
    print("AUC:", roc_value)
