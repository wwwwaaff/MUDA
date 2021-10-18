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

#import pandas as pd
import numpy as np
#import h5py
import math
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append("../tools/")



#regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2))
#regressmodel.load_weights('../result_model/weightsV2-improvement-450.hdf5')
#regressmodel.load_weights('./NASA_model/weights-improvement-250-131.65.hdf5')

def plot_history(history, fig_name, ignore_num=0, show = False):
    import matplotlib.pyplot as plt
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    acc_values = history_dict['mae']
    val_acc_values = history_dict['val_mae']

    epochs = range(1, len(loss_values) + 1 -ignore_num)

    plt.plot(epochs, loss_values[ignore_num:], 'bo', label='Training loss')#bo:blue dot蓝点
    plt.plot(epochs, val_loss_values[ignore_num:], 'ro', label='Validation loss')#b: blue蓝色
    #plt.plot(epochs, acc_values[ignore_num:], 'b', label='Training mae')#bo:blue dot蓝点
    #plt.plot(epochs, val_acc_values[ignore_num:], 'r-', label='Validation mae')#b: blue蓝色
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig = plt.gcf() # plt.savefig(fig_name)
    #if show == True:
    #    plt.show()
    fig.savefig(fig_name, dpi=100)
    #plt.close()

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

def AlexNet(W_l1RE, W_l2RE, shape, dropout_net=0):
    model = Sequential() # 16 32 64 128    256 64 1
    '''
    model.add(Conv2D(16, (4, 4), strides = 2, padding='valid',
                     input_shape=shape, 
                     kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                     kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    #model.add(AveragePooling2D((2, 2), strides = 1))
    model.add(Conv2D(32, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Conv2D(64, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3) , strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), 
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=3))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_net)) if dropout_net > 0 else None
    model.add(Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    model.add(Activation('relu'))

    model.add(Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
        kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE)))
    #model.add(Activation('softmax'))
    model.add(Activation('linear'))
    '''
    inputs = Input(shape=shape)
    if dropout_net > 0:
        x = Dropout(dropout_net)(inputs, training=True)
        x = Conv2D(16, (4, 4), strides = 2, padding='valid',
                         input_shape=shape,
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    else:
        x = Conv2D(16, (4, 4), strides = 2, padding='valid',
                         input_shape=shape,
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(inputs)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
            kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(64, (3, 3), strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
            kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3) , strides = 2, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
            kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=3)(x)

    x = Flatten()(x)
    x = Dense(256, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dense(64, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
          kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    x = Activation('relu')(x)
    x = Dense(1, kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    kernel_regularizer=l1_l2(l1=W_l1RE, l2=W_l2RE))(x)
    output = Activation('linear')(x)
    model = Model(inputs=[inputs], outputs=[output])

    #opt = keras.optimizers.rmsprop(lr=0.001)
    opt = keras.optimizers.RMSprop(lr=0.005)

    # Let's train the model using RMSprop
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
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


def generator(dataflow, dropout):
    for x, y in dataflow:
        bi = np.random.normal(0, dropout, x.shape)
        img = x + bi
        label = y
        yield (img, label) 


def train_AlexNet(EPOCHS, trainset_xpath, trainset_ypath, testset_xpath, testset_ypath, dropout_data=0, dropout_net=0):
    W_l1RE = 1e-5 # 5e-4 is best
    W_l2RE = 1e-5
    batch_size = 64
    epochs = EPOCHS
    data_augmentation = True
    save_dir = os.path.join(os.getcwd(), 'result_model')
    #model_name = 'AlexNetSmallRo' + str(SITA) + '.h5'

    #W_l1RE = 5e-4
    W_l1RE = 0
    W_l2RE = 1e-4
    #W_l2RE = 1e-5
    #x_train = np.load("../data/ATLN_2003_2014_data_x_101.npy").astype('float32')
    #y_train = np.load("../data/ATLN_2003_2014_data_y_201.npy").astype('float32')
    #x_test  = np.load("../data/ATLN_2015_2016_data_x_101.npy").astype('float32')
    #y_test  = np.load("../data/ATLN_2015_2016_data_y_201.npy").astype('float32')
    
    x_train = np.load(trainset_xpath).astype('float32')
    y_train = np.load(trainset_ypath).astype('float32')
    x_test  = np.load(testset_xpath).astype('float32')
    y_test  = np.load(testset_ypath).astype('float32')

    x_test = x_test[y_test<=180,:,:,:]
    y_test = y_test[y_test<=180]
    x_train = x_train[:, 18:83, 18:83, :]   # 18:83 = 65
    x_train = normalize_data(x_train, x_train.shape[3])
    x_test = x_test[:, 18:83, 18:83, :]   # 18:82 = 64
    x_test = normalize_data(x_test, x_test.shape[3])

    print("the shape of train set and test set: ", x_train.shape, x_test.shape)
    model_name_pre = 'Sel_PostNet-'
    model = AlexNet(W_l1RE, W_l2RE, x_train.shape[1:], dropout_net)

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
        #tb = TensorBoard(log_dir='./tmp/log', histogram_freq=10)
        filepath="./NASA_model/dropout_data={}_dropout_net={}".format(
            dropout_data, dropout_net)+"-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint= ModelCheckpoint(filepath, monitor='val_loss', verbose=1,  period=30, save_weights_only=True)
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=0,  # epsilon for ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
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
        history = model.fit_generator(generator(datagen.flow(x_train, y_train, batch_size=batch_size), dropout_data),#Mygen(x_train, y_train, batch_size=batch_size),
                                      epochs=epochs,
                                      validation_data=(x_test, y_test),
                                      shuffle=True,
                                      steps_per_epoch=int(x_train.shape[0]/batch_size)+1,
                                      workers=1,
                                      use_multiprocessing=False,
                                      callbacks=[reduce_lr, checkpoint])

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test RMSE:', np.sqrt(scores[0]))
    print('Test MAE:', scores[1])
    model_name = model_name_pre + '_dropout_data={}_dropout_net={}_'.format(dropout_data,
        dropout_net) + 'rotate-RMSE' + str(int(np.sqrt(scores[0])*100)/100.0) + '.h5'
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    # Score trained model.
    #plot_history(history, model_path+"-"+str(int(np.sqrt(scores[0])*100)/100.0)+".png", 100) # plot_history(history=history, ignore_num=5)
    #plot_history(history, model_path+"-"+str(int(scores[1]*10)/10.0)+".png", 100) # plot_history(history=history, ignore_num=5)
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

def evaluation_rotated(model_path, test_data_path, y_test_path, times=50, dropout_data=0, dropout_net=0):

    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2), dropout_net=dropout_net)
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    test_data = np.load(test_data_path).astype('float32')
    y_test = np.load(y_test_path).astype('float32')

    y_predict = np.zeros((y_test.shape[0], times))

    for rotatedsita in tqdm(range(0, times)):

        testx = test_data[:, 18:83, 18:83, :]
        testx = normalize_data(testx, testx.shape[3])

        if dropout_data > 0:
            drop_test = np.random.uniform(0, 1, size=testx.shape)
            drop_test[drop_test >= dropout_data] = 1
            drop_test[drop_test < dropout_data] = 0
            testx = testx * drop_test

        y_predict_regress = regressmodel.predict(testx).reshape(-1)

        y_predict[:, rotatedsita] = y_predict_regress
        rmse = np.sqrt(np.mean((y_predict_regress-y_test) * (y_predict_regress-y_test)))
        #print(str(rotatedsita/Rotated_Max_Sita+1) + "- rotated blend RMSE: " + str(rmse))

    y_predict_mean = np.mean(y_predict, axis = -1)
    y_predict_var = np.var(y_predict, axis =-1)
    rmse = np.sqrt(np.mean((y_predict_mean-y_test) * (y_predict_mean-y_test)))
    print("Total - rotated blend RMSE: " + str(rmse))

    var_y = np.reshape(y_predict_var, y_predict_var.shape[0])
    for v_ in var_y:
        with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
                .format(dropout_data, dropout_net)+"uncertainty_rotate2.txt", "a+") as f:
            f.write(str(v_) + "\n")

    dy_list = []
    dy = y_predict_mean - y_test
    for d_ in dy:
        dy_list.append(d_)
        with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
                .format(dropout_data, dropout_net)+"dy_rotate2.txt", "a+") as f1:
            f1.write(str(d_) + "\n")

    sorted_index = np.argsort(y_predict_var)
    total_num = len(sorted_index)
    y_pre_array = np.array(y_predict_mean)
    #sorted_x_data = test_data[sorted_index,:,:,:]
    sorted_y_pred = y_pre_array[sorted_index]
    sorted_y_data = y_test[sorted_index]
    rmse_list = []

    for i in range(1, total_num + 1):
        sorted_y_pred1 = sorted_y_pred[0:i]
        sorted_y_data1 = sorted_y_data[0:i]
        rmse = np.sqrt(np.mean((sorted_y_pred1 - sorted_y_data1) *
                               (sorted_y_pred1 - sorted_y_data1)))
        rmse_list.append(rmse)
        with open("./dropout/TC/"+'dropout_data_{}_dropout_net_{}_'
                .format(dropout_data, dropout_net)+"rmse_rotate2.txt", "a+") as f2:
            f2.write(str(rmse) + "\n")
    sorted_dy = dy[sorted_index]
    epochs = np.arange(total_num)

    plt.figure(1)
    plt.plot(epochs, sorted_dy, 'bo', label='Training loss')  # bo:blue dot蓝点
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('f(x)-y',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+'dropout_data_{}_dropout_net_{}_'
                .format(dropout_data, dropout_net)+'dy_rotate2.png',bbox_inches = 'tight')
    plt.clf()
    plt.figure(2)
    plt.plot((epochs + 1) / total_num, rmse_list, 'b-', marker="x")  # bo:blue dot蓝点
    plt.xlim(0.05, 1)
    plt.xlabel('Coverage',fontsize=16)
    plt.ylabel('Risk(RMSE)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('./dropout/TC/'+'dropout_data_{}_dropout_net_{}_'
                .format(dropout_data, dropout_net)+'rmse_rotate2.png',bbox_inches = 'tight')
    plt.clf()

    return y_predict, y_predict_var, dy


def sample_hist(model_path, test_data_path, dropout_data, dropout_net, test_label_path):
    #classmodel = load_model('./NASA_model/AlexNet0-180-0-0.5.h5')
    test_data = np.load(test_data_path).astype('float32')
    test_label = np.load(test_label_path).astype('float32')
    regressmodel = AlexNet(W_l1RE=0, W_l2RE=1e-4, shape=(65,65,2), dropout_net=dropout_net)
    regressmodel.load_weights(model_path)
    keras.backend.set_learning_phase(1)

    testx = test_data[:, 18:83, 18:83, :]
    testx = normalize_data(testx, testx.shape[3])

    for sample in range(100, 200, 10):
        pred_list = []
        for i in range(100):
            x_sample = np.expand_dims(testx[sample], 0)
            if dropout_data > 0:
                drop_test = np.random.uniform(0, 1, size=x_sample.shape)
                drop_test[drop_test >= dropout_data] = 1
                drop_test[drop_test < dropout_data] = 0
                x_sample = x_sample * drop_test

            y_predict_regress = regressmodel.predict(x_sample).reshape(-1)
            pred_list.append(y_predict_regress)

        pred_list = np.array(pred_list).reshape(-1)

        if dropout_data > 0:
            plt.figure()
            plt.hist(pred_list, alpha=0.8)
            plt.savefig('./re_output_0_01_2/dropout_data_{}_hist-{}.png'.format(dropout_data, sample))

            np.save('./re_output_0_01_2/dropout_data_{}_hist-{}.npy'.format(dropout_data, sample), pred_list)
        else:
            plt.figure()
            plt.hist(pred_list, alpha=0.8)
            plt.savefig('./re_output_0_01_2/dropout_net_{}_hist-{}.png'.format(dropout_net, sample))

            np.save('./re_output_0_01_2/dropout_net_{}_hist-{}.npy'.format(dropout_net, sample), pred_list)

        print(np.mean(pred_list), np.std(pred_list), test_label[sample])
    print('done')


def merge_fig(dropout_data, dropout_net):
    for sample in range(100, 600, 10):
        array1 = np.load('./re_output_0_01_2/dropout_data_{}_hist-{}.npy'.format(dropout_data, sample))
        array2 = np.load('./re_output_0_01_2/dropout_net_{}_hist-{}.npy'.format(dropout_net, sample))

        plt.figure(figsize=(10, 7))
        plt.hist(array1, alpha=0.8)
        plt.hist(array2, alpha=0.8)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(['dropout with data', 'dropout with net'], fontsize=16)
        plt.savefig('./re_output_0_01_2/dist_dropout_data_{}_dropout_net_{}_hist-{}.png'.
                    format(dropout_data, dropout_net, sample))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-DROP1', "--dropout_data", default=0, type=float)
    parser.add_argument('-DROP2', "--dropout_net", default=0, type=float)

    parser.add_argument("-P", "--datapath", default="../Data/TCIR-ATLN_EPAC_WPAC.h5", help="the TCIR dataset file path")
    parser.add_argument("-Tx", "--trainset_xpath", default="../Data/ATLN_2003_2014_data_x_101.npy", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="../Data/ATLN_2003_2014_data_y_101.npy", help="the trainning set y file path")

    parser.add_argument("-Tex", "--testset_xpath", default="../Data/ATLN_2015_2016_data_x_101.npy", help="the test set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="../Data/ATLN_2015_2016_data_y_101.npy", help="the test set y file path")

    parser.add_argument("-E", "--epoch", default=1, type=int, help="epochs for trainning")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sita", default=10, type=int, help="rotated sita for blending")
    args = parser.parse_args()
    
    if (args.dropout_data > 0) and (args.dropout_net > 0):
        merge_fig(dropout_data=args.dropout_data, dropout_net=args.dropout_net)
        sys.exit()

    if args.test:
        model_path = ''
        for item in os.listdir('./result_model'):
            if args.dropout_data == 0:
                args.dropout_data = int(args.dropout_data)
            if ('dropout_data={}_dropout_net={}_'.format(args.dropout_data,
        args.dropout_net) in item) and ('png' not in item) and ('RMSE' in item):
                print(item)
                model_path = os.path.join('./result_model/', item)

        evaluation_rotated(model_path, args.testset_xpath, args.testset_ypath, times=args.sita,
                           dropout_data=args.dropout_data, dropout_net=args.dropout_net)
        #sample_hist(model_path, args.testset_xpath, dropout_data=args.dropout_data, dropout_net=args.dropout_net,
        #            test_label_path=args.testset_ypath)
    else:
        for drop in [0, 0.005, 0.01, 0.02, 0.04]:
            args.dropout_data = drop
            print("************ Gaussian stddev ************************", drop)
            model_path = train_AlexNet(args.epoch, args.trainset_xpath, args.trainset_ypath, args.testset_xpath,
                                    args.testset_ypath, args.dropout_data, args.dropout_net)
            #evaluation_rotated(model_path, args.testset_xpath, args.testset_ypath, times=args.sita,
            #               dropout_data=args.dropout_data, dropout_net=args.dropout_net)
            #sample_hist(model_path, args.testset_xpath, dropout_data=args.dropout_data, dropout_net=args.dropout_net,
            #            test_label_path=args.testset_ypath)
