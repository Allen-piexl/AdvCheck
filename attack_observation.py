#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append('/data0/jinhaibo/DGAN/AdvChecker')
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
from utils import *
from keras import backend as K
from keras.models import Model
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%% CIFAR-10
(x_train, y_train), (x_test, y_test) = load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
x_train = x_train/255.0
x_test = x_test/255.0
model = load_model('/data0/jinhaibo/DGAN/train_model/cifar10_vgg.h5')
advs = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/DeepFool/adv.npy')
#%% GTSRB
nb_classes = 43
x_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100.npy')
x_train = x_train/255
y_train = np.load('/data0/jinhaibo/DGAN/GTSRB_img_Crop/Final_Training/GTSRB100-labels.npy')
y_train = to_categorical(y_train, nb_classes)
model = load_model('/data0/jinhaibo/DGAN/train_model/lenet5.h5')
model.summary()
#%%tiny
x_train = np.load("/data0/jinhaibo/DGAN//animals_10_datasets/vgg/train/img_data.npy")/255.0
y_train = np.load("/data0/jinhaibo/DGAN//animals_10_datasets/vgg/train/img_data_label.npy")
y_train = to_categorical(y_train, 10)
model = load_model('/data0/jinhaibo/DGAN//train_model/tiny_imagenet_mobilenet.h5')
#%%
def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom
#%% train
out_sort = []
tale_out_sort = []
out_sort_adv = []
tale_out_sort_adv = []
fa_values = []
fa_values_adv = []
for i in range(200):
    x = x_train[i:i+1]
    x_adv = advs[i:i+1]
    predcitions = np.argmax(model.predict(x))
    output_label = model.output[:, predcitions]
    layers_names = 'flatten'
    grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.get_layer(layers_names).output])
    weights, out_value = iterate([x])
    fc_values = out_value
    print(i)
    for i in range(len(weights)):
        # fc_values[:, i] *= weights[i]
        fc_values[:, i] = weights[i]
    fa_values.append(fc_values)
    # out_sort = np.argsort(fc_values[0])[::-1]
    # tale_out_sort = np.argsort(fc_values[0])


    predcitions_adv = np.argmax(model.predict(x_adv))
    output_label = model.output[:, predcitions_adv]
    grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.get_layer(layers_names).output])
    weights_adv, out_value_adv = iterate([x_adv])
    fc_values_adv = out_value_adv

    for i in range(len(weights_adv)):
        fc_values_adv[:, i] *= weights_adv[i]
    fa_values_adv.append(fc_values_adv)
    # out_sort_adv.append(np.argsort(fc_values_adv[0])[::-1])
    # tale_out_sort_adv.append(np.argsort(fc_values_adv[0]))
#%%
trian_x = np.array(fa_values).reshape(len(fa_values), -1)
trian_x_adv = np.array(fa_values_adv).reshape(len(fa_values_adv), -1)
#%%
trian_y = np.array([np.array([1, 0]) for i in range(200)])
trian_y_adv = np.array([np.array([0, 1]) for i in range(200)])
#%%
x_train_meta = np.vstack((trian_x, trian_x_adv))
y_train_meta = np.vstack((trian_y, trian_y_adv))
#%% train GTSRB tiny
for m in [0]:  #load adv feature
    attack_list = ['FGSM', 'BIM', 'JSMA', 'PGD', 'AUNA', 'PWA', 'Boundary']
    attack_path = ['/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/FGSM/fgsm_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/BIM/bim_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/JSMA/jsma_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/PGD/pgd_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/AUNA/auna_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/PWA/pwa_feature_',
                   '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/Boundary/Boundary_feature_']
    print('Running {}'.format(attack_list[m]))
    for i in range(80):
        if i == 0:
            tmp = np.load(attack_path[m]+'{}.npy'.format(i))
        else:
            tmp = np.vstack((tmp, np.load(attack_path[m]+'{}.npy'.format(i))))
tmp = tmp[:600]
test_y = np.array([np.array([0, 1]) for i in range(600)])
#%% load benign feature GTSRB tiny
benign_path = '/data0/jinhaibo/DGAN/AdvChecker/Features/mobile/Benign/benign_feature_'
for i in range(80):
    if i == 0:
        benign_tmp = np.load(benign_path+'{}.npy'.format(i))
    else:
        benign_tmp = np.vstack((benign_tmp, np.load(benign_path+'{}.npy'.format(i))))
benign_img = benign_tmp[:6]
test_benign = np.array([np.array([1, 0]) for i in range(6)])
#%%
def meta_classify(input_shape):
    x = Dense(1024, name='fc1', activation='relu')(input_shape)   #cifar是512
    x = Dense(512, name='fc2', activation='relu')(x)
    x = Dense(512, name='fc3', activation='relu')(x)
    x = Dense(2, name='predictions', activation='softmax')(x)
    model = Model(input_shape, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%%
x_train_meta = np.vstack((tmp, benign_img))
y_train_meta = np.vstack((test_y, test_benign))
#%%
model_detection = meta_classify(input_shape=Input(shape=x_train_meta.shape[1:]))
model_detection.fit(x_train_meta, y_train_meta, batch_size=5, epochs=7)
model_detection.save('/data0/jinhaibo/DGAN/AdvChecker/META_Models/mobile.h5')
#%%
count = 0
advs = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/FGSM/adv.npy')
for i in range(500):
    x_adv = advs[120+i:121+i]
    predcitions_adv = np.argmax(model.predict(x_adv))
    output_label = model.output[:, predcitions_adv]
    layers_names = 'flatten'
    grads = K.gradients(output_label, model.get_layer(layers_names).output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.get_layer(layers_names).output])
    weights_adv, out_value_adv = iterate([x_adv])
    fc_values_adv = out_value_adv

    for i in range(len(weights_adv)):
        fc_values_adv[:, i] *= weights_adv[i]
    # out_sort_adv = np.argsort(fc_values_adv[0])[::-1][:5]
    # tale_out_sort_adv = np.argsort(fc_values_adv[0])[:5]
    x_adv_fa = fc_values_adv.reshape(1, -1)
    if np.argmax(model_detection.predict(x_adv_fa)) == 1:
        count += 1
    # print(model_detection.predict(x_adv_fa))
print(count/500)
#%% generate images with gray
gray_img = []
for j in range(10):
    gen_img = x_train[j:j+1].copy()
    model_layer_dict1 = init_coverage_tables(model)
    label = np.argmax(model.predict(gen_img)[0])
    update_coverage(gen_img, model, model_layer_dict1, 0)
    loss1 = -1 * K.mean(model.layers[-1].output[..., label])
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    loss1_neuron = K.mean(model.get_layer(layer_name1).output[..., index1])
    layer_output = loss1 + 2 * loss1_neuron
    final_loss = K.mean(layer_output)
    grads = normalize(K.gradients(final_loss, model.input)[0])

    iterate = K.function([model.input], [loss1, loss1_neuron, grads])
    for i in range(200):
        loss_value1, loss_neuron1, grads_value = iterate([gen_img])
        grads_value = constraint_light(grads_value)
        gen_img += grads_value * 10
        gen_img = np.clip(gen_img, a_max=1, a_min=0)
        predictions1 = np.argmax(model.predict(gen_img)[0])
        if predictions1 != label:
            # gray_img.append(gen_img)
            break
    gray_img.append(gen_img)

#%%
gray_imges = np.array(gray_img).reshape(len(gray_img), 32, 32, 3)

#%%
from keras.preprocessing import image
im = image.array_to_img(x_train[1:2][0])
im.save('/data0/jinhaibo/DGAN/AdvChecker/images/1_ori.jpg')
#%%
FPR_examples = np.load('/data0/jinhaibo/DGAN/FP_example/CIFAR-VGG/train_exp.npy')
#%%
advs_fgsm = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/FGSM/adv.npy')
adv_bim = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/BIM/train/adv_test_x.npy')
# adv_mifgsm = np.load('/data0/jinhaibo/DGAN/adv_GTSRB/lenet/MIFGSM/adv.npy')
# adv_jsma = np.load('/data0/jinhaibo/DGAN/adv_GTSRB/lenet/JSMA/adv.npy')
adv_pgd = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/PGD/adv.npy')
# adv_deepfool = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/DeepFool/adv.npy')
# adv_uap = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/UAP/adv.npy')
adv_auna = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/AUNA/train/adv_test_x.npy')
adv_pwa = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/PWA/train/adv_test_x.npy')
# adv_pixel = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/Pixel/adv_pixel_vgg.npy')
# adv_boundary = np.load('/data0/jinhaibo/DGAN/adv_cifar10/vgg19/Boundary/adv.npy')
x_advs = [x_train, advs_fgsm, adv_bim, adv_pgd, adv_pwa, adv_auna]
# x_advs = [x_train, advs_fgsm, adv_bim, adv_mifgsm, adv_jsma, adv_pgd, adv_deepfool, adv_uap, adv_auna, adv_pwa, adv_pixel, adv_boundary, gray_imges, FPR_examples]
#%%
fa_values_adv = []
for m in range(len(x_advs)):
    layers_names = 'flatten'
    x_adv = x_advs[m][5:6]
    predcitions_adv = np.argmax(model.predict(x_adv))
    output_label = model.output[:, predcitions_adv]
    grads = K.gradients(output_label, model.layers[-4].output)[0]
    out_grads = K.mean(grads, axis=0)
    iterate = K.function([model.input],
                         [out_grads, model.layers[-4].output])
    weights_adv, out_value_adv = iterate([x_adv])
    fc_values_adv = out_value_adv

    for i in range(len(weights_adv)):
        fc_values_adv[:, i] = weights_adv[i]
    fa_values_adv.append(list(fc_values_adv[0]))
#%%
fscale_values = fa_values_adv
s = pd.Series(fscale_values[0])
aa = (s * 1000).tolist()
fscale_values[0] = aa
#%%
plt.figure(figsize=(5, 4))
# plt.title('Decision score')
quartile1, medians, quartile3 = np.percentile(fscale_values, [25, 50, 75], axis=1)
inds = np.arange(1, len(medians) + 1)

violin = plt.violinplot(
        fscale_values, showmeans=False, showmedians=False,
        showextrema=False)
# plt.yscale("log")
plt.scatter(inds, medians, marker='o', color='white', s=15, zorder=3)
# labels = ['Benign', 'FGSM', 'BIM', 'MIFGSM', 'JSMA', 'PGD', 'DeepFool', 'UAP', 'AUNA', 'PWA', 'Pixel', 'Boundary', 'Weather', 'FP']
labels = ['Benign\n'+'x1k', 'FGSM', 'BIM', 'PGD', 'PWA', 'Noisy']

violin['bodies'][0].set_facecolor('green')
violin['bodies'][0].set_alpha(1)
for i in range(5):
    violin['bodies'][i+1].set_facecolor('darkorange')
    # violin['bodies'][i + 1].set_edgecolor('black')
    violin['bodies'][i + 1].set_alpha(1)
# for patch in violin['bodies']:
#     patch.set_facecolor('#D43F3A')
#     patch.set_edgecolor('black')
#     patch.set_alpha(1)

plt.ylabel('Local gradient', size=19)
plt.yticks(size=11)
plt.xticks(np.arange(1, len(labels) + 1), labels, size=15, rotation=20)
# plt.xticks()
# plt.tick_params(labelsize=10)#坐标数字大小
# plt.legend(loc="lower left",frameon=False, ncol=1, fontsize = 18) #图例字体大小
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.show()