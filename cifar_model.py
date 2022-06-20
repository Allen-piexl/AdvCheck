# import tflearn
import os
import keras
import numpy as np
import foolbox
from keras.datasets import cifar10
from keras import optimizers, Input, Model
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras import layers
from keras_applications import vgg19
from keras.utils import to_categorical
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
from keras import optimizers, regularizers
from keras.initializers import he_normal
import tensorflow as tf
import keras.backend.tensorflow_backend as K

batch_size = 64
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

def normalize_preprocessing(x_train, x_validation):
    '''
    pre_processing of cifar10 datasets
    :param x_train:
    :param x_validation:
    :return: numpy array
    '''
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_validation[:, :, :, i] = (x_validation[:, :, :, i] - mean[i]) / std[i]

    return x_train, x_validation

def scheduler(epoch):
    if epoch <= 80:
        return 0.01
    if epoch <= 140:
        return 0.005
    return 0.001

def vgg19_model(input_shape):
    model = keras.Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias=True, name='fc_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(4096, name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(10, name='predictions_cifa10'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    # -------- optimizer setting -------- #
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def alexnet_model(input_shape, classes=10):
    weight_decay = 1e-4
    DATA_FORMAT = 'channels_last'
    img_input = keras.layers.Input(shape=input_shape)
    x = Conv2D(96, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(
        img_input)  # valid

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
               activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=DATA_FORMAT)(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(classes, activation='softmax')(x)
    model = keras.Model(img_input, out)
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)
    # x_train, x_validation = normalize_preprocessing(x_train, x_validation)
    x_train, x_validation = x_train / 255., x_validation / 255.
    # print(np.max(x_train), np.min(x_train), np.max(x_validation), np.min(x_validation))

    model = alexnet_model(input_shape=x_train.shape[1:])
    # print(model.summary())
    model.load_weights("./train_model/cifar10_alexnet.h5")

    import logging
    import cv2

    attack_list = ['DeepFool','FGSM','PGD','Boundary']
    attack_name = attack_list[3]
    fmodel = foolbox.models.KerasModel(model, bounds=(0, 1))
    #训练集
    path = './adv_cifar10/alexnet/adv_examples/' + attack_name + "/"
    print("-------------------------------------------------------")
    if not os.path.exists(path):
        os.makedirs(path)

    adv_x_all=[]
    y_validation_all=[]
    print(attack_name)
    for i in range(500):
        image = x_train[i:i+1]
        ground_truth = y_train[i:i+1]
        truth = np.argmax(ground_truth, axis=1)
        image = np.expand_dims(image, axis=0)
        # print(np.shape(image))
        # print(np.shape(truth))
        attack = foolbox.attacks.BoundaryAttack(fmodel)    #5000张按训练集的顺序
        adv_x=attack(image[0], np.array(truth))
        if adv_x is None:
            adv_x = image
        if i % 10 == 0:
            print(attack_name,  i)

        adv_x_all.append(adv_x)
        y_validation_all.append(truth)

    adv_x_all = np.reshape(adv_x_all, [500, 32, 32, 3])
    print(np.shape(adv_x_all))
    print(np.shape(y_validation_all))
    np.save(path + 'adv.npy', adv_x_all)
    np.save(path + 'truth_y.npy', y_validation_all)
    # # # adv_x=np.expand_dims(adv_x_all,-1)
    y_validation_all = to_categorical(y_validation_all, 10)
    # adv_x_all=np.array(adv_x_all)
    y_validation_all=np.array(y_validation_all)
    score = model.evaluate(adv_x_all, y_validation_all)
    print(attack_name,'ACC on train data %s:%.2f%%' % (model.metrics_names[1], 100 - score[1] * 100))


