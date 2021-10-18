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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.regularizers import l2,l1,l1_l2
from tensorflow.keras.initializers import TruncatedNormal, RandomNormal
from tensorflow.keras.datasets import mnist, cifar10, boston_housing
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
from sklearn.preprocessing import MinMaxScaler
import sys
sys.path.append("../tools/")

#regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
#regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')
#regressmodel.load_weights('./NASA_model/weights-improvement-250-131.65.hdf5')

def plot_history(history, fig_name, ignore_num=0, show=False, acc=True):
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    #acc_values = history_dict['mean_absolute_error']
    #val_acc_values = history_dict['val_mean_absolute_error']

    epochs = range(1, len(loss_values) + 1 - ignore_num)
    plt.figure()
    plt.plot(np.arange(0, len(loss_values)), loss_values, 'bo', label='Training loss')  # bo:blue dot蓝点
    plt.plot(np.arange(0, len(val_loss_values)), val_loss_values, 'ro', label='Validation loss')  # b: blue蓝色
    # plt.plot(epochs, acc_values[ignore_num:], 'b', label='Training mae')#bo:blue dot蓝点
    # plt.plot(epochs, val_acc_values[ignore_num:], 'r-', label='Validation mae')#b: blue蓝色
    plt.title('Training and validation loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
   # plt.savefig(fig_name)
    if show == True:
        plt.show()
    plt.savefig(fig_name)
    plt.close()

    if acc:
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']
        plt.figure()
        plt.plot(np.arange(0, len(acc_values)), acc_values, 'bo', label='Training acc')  # bo:blue dot蓝点
        plt.plot(np.arange(0, len(val_acc_values)), val_acc_values, 'ro', label='Validation acc')  # b: blue蓝色
        # plt.plot(epochs, acc_values[ignore_num:], 'b', label='Training mae')#bo:blue dot蓝点
        # plt.plot(epochs, val_acc_values[ignore_num:], 'r-', label='Validation mae')#b: blue蓝色
        plt.title('Training and validation acc', fontsize=16)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('acc', fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        # plt.savefig(fig_name)
        if show == True:
            plt.show()
        plt.savefig(fig_name[:-4]+'acc.png')
        plt.close()

def Noise_DNet(W_l1RE, W_l2RE, shape, noisestddev=0.005):
    model = Sequential() # 16 32 64 128    256 64 1  
    model.add(GaussianNoise(stddev=noisestddev, input_shape=shape))
    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    
    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    #model.add(Activation('softmax'))
    model.add(Activation('linear'))
    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #model.summary()
    return model

def Noise_DropNet(W_l1RE, W_l2RE, shape, dropout_rate=0.005):
    model = Sequential() # 16 32 64 128    256 64 1  
    #model.add(GaussianNoise(stddev=noisestddev, input_shape=shape))
    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE), input_shape=shape))
    model.add(Activation('relu'))
    
    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Dense(1024, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    #model.add(Activation('softmax'))
    model.add(Activation('linear'))
    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    #model.summary()
    return model


def normalize_data(x_test, chanel_num):
    result=[]
    height = x_test.shape[1]
    for each_sam in x_test:
        new_sam = []
        for i in range(chanel_num):
            chanel = each_sam[:,:,i]
            chanel = (chanel - np.mean(chanel)) / (np.std(chanel)+0.01)
            if i==0:
                new_sam = chanel
            else:
                new_sam = np.concatenate((new_sam, chanel), axis =1)
               
        new_sam = new_sam.reshape((height,height,chanel_num),order='F')
        result.append(new_sam)
    result = np.array(result)
    return result

def train(model_name, EPOCHS, dataset, noisestddev, pretrained=False, idx=0, drop=False):
    batch_size = 128
    epochs = EPOCHS
    data_augmentation = False
    save_dir = os.path.join(os.getcwd(), 'result_model')

    W_l1RE = 0
    W_l2RE = 1e-3
    #W_l2RE = 1e-4

    assert  dataset.lower() == 'bostonhousing'
    #(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    all_data = np.array(np.loadtxt('data.txt'))
    x_train_data, y_train_data = all_data[:,:9], all_data[:,9]
    #ss=MinMaxScaler()
    #xx_data = ss.fit_transform(x_train_data)
    numd, samd = x_train_data.shape
    x_mean, x_std = np.mean(x_train_data[:int(numd*0.8)], 0), np.std(x_train_data[:int(numd*0.8)], 0)
    xx_data = (x_train_data - x_mean) / x_std
    x_train, x_test = xx_data[:int(numd*0.8)], xx_data[int(numd*0.8):]
    y_train, y_test = y_train_data[:int(numd*0.8)], y_train_data[int(numd*0.8):]
    
    if drop:
        model = Noise_DropNet(W_l1RE, W_l2RE, x_train.shape[1:], dropout_rate=noisestddev)
    else:
        model = Noise_DNet(W_l1RE, W_l2RE, x_train.shape[1:], noisestddev=noisestddev)


    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Sel_PostNet-'

        #model.load_weights('./NASA_model/weights-improvement-200-148.05.hdf5')
    if not data_augmentation:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=10, min_lr=1e-6)
        prefix = '{}:{}-{}-noisestddev-{}-{}-'.format(model_name, pretrained, dataset, noisestddev, idx)
        print(prefix)
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath="./Noise_NASA_model/"+prefix+\
                 "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period=50)
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=[reduce_lr, checkpoint])
    
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test Accuracy:', scores[1])
    model_name = model_name_pre + prefix +\
                 'Acc' + str(int(scores[1]*100)/100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    plot_history(history, model_path+"-"+str(int(scores[1]*10)/10.0)+".png", 100, acc=False) 
    print("Over!!!")
    return model_path


def evaluation_rotated(model_name, dataset, model_path, pretrained=False, times=50, noisestddev=0.01, idx=0):
    # classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    assert  dataset.lower() == 'bostonhousing'
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()

    classifymodel = Noise_DNet(W_l1RE, W_l2RE, x_train.shape[1:], noisestddev=noisestddev)

    classifymodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    loss_array = []

    print(f'test dataset size = {x_test.shape}')
    y_predict = np.zeros((y_test.shape[0], y_test.shape[1], times))

    for rotatedsita in tqdm(range(0, times)):
        test_data = x_test
        test_data = test_data + np.random.normal(0, noisestddev, test_data.shape)
        y_predict_classify = classifymodel.predict(test_data)

        y_predict[:, :, rotatedsita] = y_predict_classify
        loss = keras.losses.categorical_crossentropy(y_predict_classify, y_test)
        loss_array.append(loss)

    y_predict_mean = np.mean(y_predict, axis=-1)
    y_predict_var = np.sum(np.var(y_predict, axis=-1), axis=-1)
    #print("Total - rotated blend accuracy: " + str(rmse))
    
    var_y = np.reshape(y_predict_var, y_predict_var.shape[0])

    prefix = '{}:{}-{}-noisestddev-{}-{}-'.format(model_name, pretrained, dataset, noisestddev, idx)
    for idx, v_ in enumerate(var_y):
        with open("./cifar_noise/" + prefix + "uncertainty_rotate9_2.txt", "a+") as f:
            f.write('{:.9f}, class {}\n'.format(v_, np.argmax(y_test[idx])))
    
    dy = keras.losses.categorical_crossentropy(y_predict_mean, y_test)
    for d_ in dy:
        with open("./cifar_noise/"+ prefix
                  + "dy_rotate9_2.txt", "a+") as f1:
            f1.write(str(np.array(d_)) + "\n")
    
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
        with open("./cifar_noise/"+prefix
                  +"loss_rotate9_2.txt", "a+") as f2:
            f2.write(str(np.mean(loss)) + "\n")
        with open("./cifar_noise/"+prefix
                  +"acc_rotate9_2.txt", "a+") as f2:
            f2.write(str(acc) + "\n")

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)

    # 平均输出的loss排序
    sorted_dy = np.array(dy)[sorted_index]
    epochs = np.arange(total_num)

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
    plt.figure(2)
    # 根据输出方差排序样本后，前n个样本输出均值的平均loss
    plt.plot((epochs + 1)/len(epochs), loss_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlabel('Coverage', fontsize=16)
    plt.ylabel('Risk(Loss)', fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./cifar_noise/'+prefix
                +'loss_rotate9_2.png',bbox_inches = 'tight')
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
    plt.clf()

    return y_predict, y_predict_var, dy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.parse_args()
    parser.add_argument('-D', "--dataset", default='bostonhousing', type=str)
    parser.add_argument('-STD', "--stddev", default=0.01, type=float)
    parser.add_argument("-M", "--model", type=str, default='dnet', help="model to train")
    parser.add_argument("-PR", "--pretrained", action="store_true", help="pretrained")
    parser.add_argument("-I", "--idx", default=0, type=int, help="index")

    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="../Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=50, type=int, help="epochs for trainning")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sita", default=10, type=int, help="rotated sita for blending")
    args = parser.parse_args()
    if args.test:
        model_path = ''
        prefix = '{}:{}-{}-noisestddev-{}-'.format(args.model, args.pretrained, args.dataset, args.stddev)
        for item in os.listdir('./result_model'):
            if (prefix in item) and ('png' not in item) :
                model_path = os.path.join('./result_model/', item)
                break
        print(model_path)
        #model_path = './result_model/Sel_PostNet-vgg_False-cifar-noisestddev-0.01-2-Acc0.9.h5'
        evaluation_rotated(args.model, args.dataset, model_path, pretrained=args.pretrained,
                           times=args.sita, noisestddev=args.stddev, idx=args.idx)
    else:
        # for idx in range(5):
        #     for sdev in [0, 0.005, 0.01, 0.02, 0.04]: #[0, 0.005, 0.01, 0.02, 0.04]:
        #         args.stddev = sdev
        #         print("************ Gaussian stddev ************************", sdev)
        #         model_path = train(args.model, args.epoch, args.dataset, args.stddev, args.pretrained, idx=idx)

        for idx in range(10):
            for dropp in [0.1]: #[0, 0.02, 0.05, 0.1, 0.2]:
                args.stddev = dropp
                print("************ Dropout rate ************************")
                model_path = train(args.model, args.epoch, args.dataset, args.stddev, args.pretrained, idx=idx, drop=True)


        #evaluation_rotated(args.model, args.dataset, model_path, args.pretrained
        #                   , times=args.sita, noisestddev=args.stddev, idx=args.idx)
