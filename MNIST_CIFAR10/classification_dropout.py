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

def rotate_by_channel(data, sita, length=2):
    newdata = []
    chanel_num = data.shape[3]
    height = data.shape[1]
    if length > 1:
        for index, singal in enumerate(data):
            new_sam = np.array([])
            for i in range(chanel_num):
                channel = singal[:,:,i]
                img = Image.fromarray(channel)
                new_img = img.rotate(sita[index])
                new_channel = np.asarray(new_img)
                if i==0:
                    new_sam = new_channel
                else:
                    new_sam = np.concatenate((new_sam, new_channel), axis = 1) 
            new_sam = new_sam.reshape((height,height,chanel_num),order='F')
            newdata.append(new_sam)
    else:
        print("Error! data length = 1...")
    return np.array(newdata)

def Noise_AlexNet(W_l1RE, W_l2RE, shape, noisestddev=0.005):
    model = Sequential() # 16 32 64 128    256 64 1  
    model.add(GaussianNoise(stddev=noisestddev, input_shape=shape))
    model.add(Conv2D(16, (4, 4), strides = 2, padding='same',
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    #model.add(AveragePooling2D((2, 2), strides = 1))
    model.add(Conv2D(32, (3, 3), strides = 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(64, (3, 3), strides = 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3) , strides = 2, padding='same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(10, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    #model.add(Activation('softmax'))
    model.add(Activation('softmax'))

    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

def vgg(shape, noisestddev=0.005, pre_trained=False):
    if not pre_trained:
        model = Sequential()
        weight_decay = 1e-4

        model.add(GaussianNoise(stddev=noisestddev, input_shape=shape))
        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (1, 1), padding='same'))
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))

    else:
        model = Sequential()
        model.add(GaussianNoise(stddev=noisestddev, input_shape=shape))
        model.add(vgg16.VGG16(include_top=False, weights='imagenet'))
        model.add(Flatten())
        
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.Adam(lr=0.001)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    model.summary()
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

def train(model_name, EPOCHS, dataset, noisestddev, pretrained=False, idx=0):
    W_l1RE = 1e-5 # 5e-4 is best
    W_l2RE = 1e-5
    batch_size = 64
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'result_model')

    #W_l1RE = 5e-4
    W_l1RE = 0
    W_l2RE = 1e-4

    (x_train, y_train), (x_test, y_test) = mnist.load_data() if dataset.lower() == 'mnist' else cifar10.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.
    if dataset.lower() == 'mnist':
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, 10) if dataset.lower() == 'mnist' \
        else keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10) if dataset.lower() == 'mnist' \
        else keras.utils.to_categorical(y_test, 10)

    if model_name.lower() == 'vgg':
        model = vgg(x_train.shape[1:], noisestddev=noisestddev, pre_trained=pretrained)
    else:
        model = Noise_AlexNet(W_l1RE, W_l2RE, x_train.shape[1:], noisestddev=noisestddev)

    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Sel_PostNet-'

        #model.load_weights('./NASA_model/weights-improvement-200-148.05.hdf5')
    if not data_augmentation:
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
    
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=20, min_lr=1e-6)
        prefix = '{}:{}-{}-noisestddev-{}-{}-'.format(model_name, pretrained, dataset, noisestddev, idx)
        print(prefix)
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath="./Noise_NASA_model/"+prefix+\
                 "weights-improvement-{epoch:03d}-{val_loss:.3f}.hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period=25)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=0,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None)#,
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),#Mygen(x_train, y_train, batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            shuffle=True,
                            steps_per_epoch=int(x_train.shape[0]/batch_size)+1,
                            callbacks=[reduce_lr, checkpoint])

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test Accuracy:', scores[1])
    model_name = model_name_pre + prefix +\
                 'Acc' + str(int(scores[1]*10000)/100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    #plot_history(history, model_path+"-"+str(int(np.sqrt(scores[0])*100)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    plot_history(history, model_path+"-"+str(int(scores[1]*10000)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    print("Over!!!")
    return model_path

def evaluation(test_data, y_test, BATCH_SIZE):
    classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
    regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')

    Rotated_Max_Sita = 45
    y_predict = np.zeros(y_test.shape)
    y_class_predict = np.zeros((y_test.shape[0], 8))

    for rotatedsita in  range(0, 360, Rotated_Max_Sita):
        testx = rotate_by_channel(test_data, np.ones(test_data.shape[0])*rotatedsita, 2)
        testx = testx[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        y_predict_regress = regressmodel.predict(testx, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
        print("Test data rotated sita: ", rotatedsita)
        y_predict = y_predict + y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

        y_class_predict_tmp = classmodel.predict(testx, batch_size=32, verbose=0)
        if(len(y_class_predict_tmp.shape)==3):
            y_class_predict_tmp = y_class_predict_tmp.reshape(-1)
        y_class_predict = y_class_predict + y_class_predict_tmp
    
    y_predict = y_predict / (360/Rotated_Max_Sita)
    rmse = np.sqrt(np.mean((y_predict-y_test) * (y_predict-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    y_class_predict = y_class_predict / (360/Rotated_Max_Sita)
    dy = y_predict - y_test
    y_class = intensity2class(y_test)

    #np.savetxt("y_class_predict.csv", y_class_predict, delimiter=',')
    #np.savetxt("y_class_predict_maxindex.csv", y_class_predict.argmax(axis=-1), delimiter=',')
    #np.savetxt("y_class.csv", y_class, delimiter=',')
    #np.savetxt("dy.csv", dy, delimiter=',')
    return y_class_predict, y_class, dy

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
        #for midx in range(2):
        #    args.idx = midx
        for sdev in [0]:
            args.stddev = sdev
            print("************ Gaussian stddev ************************", sdev)
            model_path = train(args.model, args.epoch, args.dataset, args.stddev, args.pretrained, idx=args.idx)

        #evaluation_rotated(args.model, args.dataset, model_path, args.pretrained
        #                   , times=args.sita, noisestddev=args.stddev, idx=args.idx)
