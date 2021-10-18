from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras.initializers import TruncatedNormal, RandomNormal
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.applications import vgg16
#from keras.layers.core import Lambda

import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

#import pandas as pd
import numpy as np
#import h5py
import math
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append("../tools/")
from model import vgg, Noise_AlexNet


def evaluation_rotated(model_name, dataset, model_path, pretrained=False, times=50, noisestddev=0.01, idx=0):
    # classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    (x_train, y_train), (x_test, y_test) = mnist.load_data() if dataset.lower() == 'mnist' else cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    if dataset.lower() == 'mnist':
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    y_train = keras.utils.to_categorical(y_train, 10 if dataset.lower() == 'mnist' else 10)
    y_test = keras.utils.to_categorical(y_test, 10 if dataset.lower() == 'mnist' else 10)

    if model_name.lower() == 'vgg':
        classifymodel = vgg(x_train.shape[1:], noisestddev=noisestddev, pre_trained=pretrained)
    else:
        classifymodel = Noise_AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=x_train.shape[1:], noisestddev=noisestddev)

    classifymodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    loss_array = []

    print(f'test dataset size = {x_test.shape}')
    y_predict = np.zeros((y_test.shape[0], y_test.shape[1], times))

    for rotatedsita in tqdm(range(0, times)):
        test_data = x_test
        #test_data = test_data + np.random.normal(0, noisestddev, test_data.shape)
        y_predict_classify = classifymodel.predict(test_data)

        y_predict[:, :, rotatedsita] = y_predict_classify
        loss = keras.losses.categorical_crossentropy(y_predict_classify, y_test)
        loss_array.append(loss)

    loss_array = np.array(loss_array)
    print('loss_array shape = ', loss_array.shape)
    ll = np.mean(loss_array)
    with open(f'll_std{noisestddev}.txt', 'a+') as f:
        f.write(f'idx={idx}, ll={ll}\n')
    
    y_predict_mean = np.mean(y_predict, axis=-1)
    y_predict_var = np.sum(np.var(y_predict, axis=-1), axis=-1)
    #print("Total - rotated blend accuracy: " + str(rmse))
    
    var_y = np.reshape(y_predict_var, y_predict_var.shape[0])

    prefix = '{}:{}-{}-noisestddev-{}-{}-'.format(model_name, pretrained, dataset, noisestddev, idx)
    dy = keras.losses.categorical_crossentropy(y_predict_mean, y_test)
    '''
    for idx, v_ in enumerate(var_y):
        with open("./cifar_noise/" + prefix + "uncertainty_rotate9_2.txt", "a+") as f:
            f.write('{:.9f}, class {}\n'.format(v_, np.argmax(y_test[idx])))
    
    for d_ in dy:
        with open("./cifar_noise/"+ prefix
                  + "dy_rotate9_2.txt", "a+") as f1:
            f1.write(str(np.array(d_)) + "\n")'''
    
    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    loss_list = []
    acc_list = []
    
    for i in range(1, total_num + 1):
        sorted_y_pred1 = sorted_y_pred[0:i]
        sorted_y_data1 = sorted_y_data[0:i]
        loss = keras.losses.categorical_crossentropy(sorted_y_data1, sorted_y_pred1)
        acc = np.mean(np.argmax(sorted_y_data1, axis=1) == np.argmax(sorted_y_pred1, axis=1))
        loss_list.append(np.mean(loss))
        acc_list.append(acc)
        '''
        with open("./cifar_noise/"+prefix
                  +"loss_rotate9_2.txt", "a+") as f2:
            f2.write(str(np.mean(loss)) + "\n")
        with open("./cifar_noise/"+prefix
                  +"acc_rotate9_2.txt", "a+") as f2:
            f2.write(str(acc) + "\n")'''

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)

    # 平均输出的loss排序
    sorted_dy = np.array(dy)[sorted_index]
    epochs = np.arange(total_num)
    '''
    plt.figure(1)
    # 根据输出方差排序后，每个样本输出均值的loss
    plt.plot(epochs, sorted_dy, 'bo', label='Training loss')  # bo:blue dot蓝点
    plt.xlabel('Sample', fontsize=16)
    plt.ylabel('f(x)-y', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./cifar_noise/'+ prefix
                +'dy_rotate9_2.png',bbox_inches = 'tight')
    plt.clf()
    '''
    plt.figure(2)
    # 根据输出方差排序样本后，前n个样本输出均值的平均loss
    plt.plot((epochs + 1)/len(epochs), loss_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk(Loss)', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./cifar_noise/'+prefix
                +'loss_rotate9_2.png',bbox_inches = 'tight')
    np.save('./cifar_noise/'+prefix
            +'loss.npy', loss_list)
    plt.clf()
    plt.figure(3)
    # 根据输出方差排序样本后，前n个样本输出均值的平均accuracy
    plt.plot((epochs + 1)/len(epochs), acc_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk(Acc)', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./cifar_noise/'+prefix
                +'acc_rotate9_2.png',bbox_inches = 'tight')
    np.save('./cifar_noise/'+prefix
            +'acc.npy', acc_list)
    plt.clf()

    print(f"{model_path}_Accuracy: {acc_list[-1]}")
    return y_predict_mean, y_predict_var, acc_list[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.parse_args()
    parser.add_argument('-D', "--dataset", default='mnist', type=str)
    parser.add_argument('-STD', "--stddev", default=0.01, type=float)
    parser.add_argument("-M", "--model", type=str, default='alexnet', help="model to train")
    parser.add_argument("-PR", "--pretrained", action="store_true", help="pretrained")
    parser.add_argument("-I", "--idx", default=0, type=int, help="index")

    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="../Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=50, type=int, help="epochs for trainning")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--avg_idx", action="store_true")
    parser.add_argument("--sita", default=10, type=int, help="rotated sita for blending")
    args = parser.parse_args()
    if args.stddev == 0.0:
        args.stddev = int(0) 
    if args.test:
        model_path = []
        prefix = '{}:{}-{}-noisestddev-{}-'.format(args.model, args.pretrained, args.dataset, args.stddev)
        for item in os.listdir('./result_model_1003'):
            if (prefix in item) and ('png' not in item) :
                model_path.append(os.path.join('./result_model_1003/', item))
        print(f"{prefix} model number: {len(model_path)}\n Model_path: {model_path}")
        #model_path = './result_model/Sel_PostNet-vgg_False-cifar-noisestddev-0.01-2-Acc0.9.h5'
        if args.avg_idx:
            y_means, y_vars, accs = [], [], []
            for idx in np.arange(10):
                args.idx = idx
                for model_path_idx in model_path:
                    if f"-{idx}-Acc" in model_path_idx:
                        my_model_path = model_path_idx
                        break
                y_mean, y_var, acc = evaluation_rotated(args.model, args.dataset, my_model_path, pretrained=args.pretrained,
                            times=args.sita, noisestddev=args.stddev, idx=args.idx)
                #y_means.append(y_mean)
                #y_vars.append(y_var)
                accs.append(acc)
                print(f"Idx={idx} Acc: {acc}")
            acc_mean = np.array(accs).mean()
            acc_stddev = np.array(accs).std()
            print(f"ACCs: {accs}")
            print(f"{prefix} model number: {len(model_path)},\n [Avg Acc]: {acc_mean}, [Std Acc]: {acc_stddev}")
            #y_ = np.array(y_means).mean(axis=0)
            #stddev = np.stddev
            #(x_train, y_train), (x_test, y_test) = mnist.load_data() if dataset.lower() == 'mnist' else cifar10.load_data()
            #avg_acc = np.mean(np.mean(np.argmax(y_, axis=1) == np.argmax(y_test, axis=1)))
            #print("{prefix} model number: {len(model_path)}, [Avg Acc]: {avg_acc}")
        else:
            y_mean, y_var, acc = evaluation_rotated(args.model, args.dataset, model_path[0], pretrained=args.pretrained,
                            times=args.sita, noisestddev=args.stddev, idx=args.idx)