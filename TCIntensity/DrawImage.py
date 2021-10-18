# -*- coding: utf-8 -*-

from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='1' 
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import models
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Conv2D, MaxPooling2D, BatchNormalization, GaussianNoise
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.initializers import TruncatedNormal, RandomNormal
#from keras.layers.core import Lambda
from tqdm import tqdm
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
#tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

from TC_dropout import AlexNet, normalize_data, rotate_by_channel

#import pandas as pd
import numpy as np
#import h5py
import math
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append("../tools/")


def evaluation(model_path, test_data_path, y_test_path, plot_dir, rotate_times=10, dropout_net=0, idx=0, tau=1.0):
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2), dropout_net=dropout_net)
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    test_data = np.load(test_data_path).astype('float32')
    y_test = np.load(y_test_path).astype('float32')
    y_predict = np.zeros((y_test.shape[0], rotate_times))

    for i, rotatedsita in enumerate(tqdm(range(0, 360, int(360/rotate_times)))):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress = regressmodel.predict(testx).reshape(-1)

        y_predict[:, i] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
    
    diff2 = np.array((y_predict - np.expand_dims(y_test, -1)) ** 2, dtype=np.float128)
    ll = np.mean(np.log(np.mean(np.exp(- 0.5 * tau * diff2), axis=1)))
    ll += (
        - 0.5* np.log(2 * np.pi)
        + 0.5 * np.log(tau)
    )
    with open(f'll_rotate{rotate_times}.txt', 'a+') as f:
        f.write(f'idx={idx}, ll={ll}\n')
    
    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis = -1)

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    rmse_list = []

    prefix = 'alexnet:rotate-{}-{}-'.format(rotate_times, idx)

    for i in range(1, total_num + 1):
        sorted_y_pred1 = sorted_y_pred[0:i]
        sorted_y_data1 = sorted_y_data[0:i]
        rmse = np.sqrt(np.mean((sorted_y_pred1 - sorted_y_data1) *
                               (sorted_y_pred1 - sorted_y_data1)))
        rmse_list.append(rmse)
    
    epochs = np.arange(total_num)

    plt.figure(2)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(plot_dir, prefix+'rmse.png'), bbox_inches='tight')
    np.save(os.path.join(plot_dir, prefix+'rmse.npy'), rmse_list)
    plt.clf()

    return y_predict, y_predict_var, rmse_list[-1]


def evaluation_rotated(model_path, test_data_path, y_test_path, plot_dir, times=50, noisestddev=0.01, dropout_data=0, dropout_net=0, idx=0, tau=1.0):

    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2), dropout_net=dropout_net)
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    test_data = np.load(test_data_path).astype('float32')
    y_test = np.load(y_test_path).astype('float32')

    y_predict = np.zeros((y_test.shape[0], times))

    for i in tqdm(range(0, times)):

        testx = test_data[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        #testx = testx + np.random.normal(0, noisestddev, testx.shape)

        # if dropout_data > 0:
        #     drop_test = np.random.uniform(0, 1, size=testx.shape)
        #     drop_test[drop_test >= dropout_data] = 1
        #     drop_test[drop_test < dropout_data] = 0
        #     testx = testx * drop_test

        y_predict_regress = regressmodel.predict(testx).reshape(-1)
        # print(y_predict_regress.shape, y_test.shape)
        # print(y_predict_regress)
        # print(y_test)

        y_predict[:, i] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        #print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

    diff2 = np.array((y_predict - np.expand_dims(y_test, -1)) ** 2, dtype=np.float128)
    print(diff2.shape)
    print(diff2.max())
    tmp = np.sum(np.exp(- 0.5 * tau * diff2), axis=1)
    print(tmp.shape)
    print(tmp.min())
    print(tmp)
    ll = np.mean(np.log(np.mean(np.exp(- 0.5 * tau * diff2), axis=1)))
    ll += (
        - 0.5* np.log(2 * np.pi)
        + 0.5 * np.log(tau)
    )
    with open(f'll_std{noisestddev}.txt', 'a+') as f:
        f.write(f'idx={idx}, ll={ll}\n')
    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis = -1)
    # rmse = np.sqrt(np.mean((y_predict_mean-y_test) * (y_predict_mean-y_test)))
    # print("Total - rotated blend RMSE: " + str(rmse))

    # var_y = np.reshape(y_predict_var, y_predict_var.shape[0])
    # for v_ in var_y:
    #     with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
    #             .format(dropout_data, dropout_net)+"uncertainty_rotate2.txt", "a+") as f:
    #         f.write(str(v_) + "\n")

    # dy_list = []
    # dy = y_predict_mean - y_test
    # for d_ in dy:
    #     dy_list.append(d_)
    #     with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
    #             .format(dropout_data, dropout_net)+"dy_rotate2.txt", "a+") as f1:
    #         f1.write(str(d_) + "\n")

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    rmse_list = []

    prefix = 'alexnet:noisestddev-{}-{}-'.format(noisestddev, idx)

    for i in range(1, total_num + 1):
        sorted_y_pred1 = sorted_y_pred[0:i]
        sorted_y_data1 = sorted_y_data[0:i]
        rmse = np.sqrt(np.mean((sorted_y_pred1 - sorted_y_data1) *
                               (sorted_y_pred1 - sorted_y_data1)))
        rmse_list.append(rmse)
        # with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
        #         .format(dropout_data, dropout_net)+"rmse_rotate2.txt", "a+") as f2:
        #     f2.write(str(rmse) + "\n")
    # sorted_dy = dy[sorted_index]
    epochs = np.arange(total_num)

    # plt.figure(1)
    # plt.plot(epochs, sorted_dy, 'bo', label='Training loss')  # bo:blue dot蓝点
    # plt.xlabel('Sample',fontsize=16)
    # plt.ylabel('f(x)-y',fontsize=16)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.savefig('./dropout/TC/'+'dropout_data_{}_dropout_net_{}_'
    #             .format(dropout_data, dropout_net)+'dy_rotate2.png',bbox_inches = 'tight')
    # plt.clf()
    plt.figure(2)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-')  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(plot_dir, prefix+'rmse.png'),bbox_inches = 'tight')
    np.save(os.path.join(plot_dir, prefix+'rmse.npy'), rmse_list)
    plt.clf()

    return y_predict, y_predict_var, rmse_list[-1]


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
    parser.add_argument("--rotate_times", default=0, type=int, help="rotate times (360/rotate_times)")
    parser.add_argument("--sita", default=10, type=int, help="rotated sita for blending")
    args = parser.parse_args()
    if args.stddev == 0.0:
        args.stddev = int(0)
    if args.test:
        y_means, y_vars, rmses = [], [], []
        if args.rotate_times > 0:
            for idx in range(1, 11):
                folder_name = 'TC_data_gaussian_0714_work_No{}'.format(idx)
                plot_dir = '../{}/TC_rotate'.format(folder_name)
                os.makedirs(plot_dir, exist_ok=True)
                for item in os.listdir('../{}/result_model'.format(folder_name)):
                    if 'dropout_data=0_' in item:
                        model_path = os.path.join('../{}/result_model'.format(folder_name), item)
                        break
                y_mean, y_var, rmse = evaluation(model_path, args.testset_xpath, args.testset_ypath, plot_dir=plot_dir, rotate_times=args.rotate_times, idx=idx)
                rmses.append(rmse)
                print(f"Idx={idx} RMSE: {rmse}")
            rmse_mean = np.array(rmses).mean()
            rmse_stddev = np.array(rmses).std()
            print(f"RMSEs: {rmses}")
            prefix = 'alexnet:rotate-{}'.format(args.rotate_times)
            print(f"{prefix} model number: {len(rmses)},\n [Avg RMSE]: {rmse_mean}, [Std RMSE]: {rmse_stddev}")
        else:
            for idx in range(1, 11):
                folder_name = 'TC_data_gaussian_0714_work_No{}'.format(idx)
                plot_dir = '../{}/TC_noise'.format(folder_name)
                os.makedirs(plot_dir, exist_ok=True)
                for item in os.listdir('../{}/result_model'.format(folder_name)):
                    if 'dropout_data={}'.format(args.stddev) in item:
                        model_path = os.path.join('../{}/result_model'.format(folder_name), item)
                        break
                y_mean, y_var, rmse = evaluation_rotated(model_path, args.testset_xpath, args.testset_ypath, plot_dir=plot_dir, times=args.sita, noisestddev=args.stddev, idx=idx)
                rmses.append(rmse)
                print(f"Idx={idx} RMSE: {rmse}")
            rmse_mean = np.array(rmses).mean()
            rmse_stddev = np.array(rmses).std()
            print(f"RMSEs: {rmses}")
            prefix = 'alexnet:noisestddev-{}'.format(args.stddev)
            print(f"{prefix} model number: {len(rmses)},\n [Avg RMSE]: {rmse_mean}, [Std RMSE]: {rmse_stddev}")