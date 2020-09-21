# coding=utf-8
import os
import scipy.io as sio
from keras.utils import to_categorical
import numpy as np
from config import *

def load_one_sub(sub,chan,dataformat,datafile):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    if band_pass:
        print('Bandpass: load data and label from \033[1;32;m{}\033[0m'.format(datafile))
        data_file = sio.loadmat(datafile)#['TotalCell']
        data = data_file['Data']#[0,0]  # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 886)
        label = data_file['Label']#[0,0]  # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 886)
        # data = file['Data']  # 以前的数据的处理方式 (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 887)
    else:
        file = sio.loadmat(datafile)#['TotalCell']
        print('load data and label from \033[1;32;m{}\033[0m'.format(datafile))
        data = file['Data']#[0,0] # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 887)
        label = file['Label']#[0,0] # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 887)

    # extract test data
    tmp = data[0, sub]  # (200, 62, 1198)
    tmp = tmp[:,chan,:] # (200, chans, 1198)
    test_x = np.expand_dims(np.transpose(tmp,(2,0,1)),axis=1) #(1198,1,200, chans)
    test_y = to_categorical(label[0, sub].reshape((tmp.shape[2], 1)))

    # extract training data
    train_x = None
    train_y = None
    train_sub_range = [i for i in range(0,sub)] + [j for j in range(sub+1,total_sub_num)]
    for i in train_sub_range:
        tmp = data[0,i]
        tmp = tmp[:,chan,:]
        tmp_l = label[0,i]
        if train_x is None:
            train_x = tmp
            train_y = tmp_l
        else:
            train_x = np.concatenate([train_x,tmp],axis=2)
            train_y = np.concatenate([train_y,tmp_l],axis=1)
    train_x = np.transpose(np.expand_dims(train_x,axis=0),(3,0,1,2))
    train_y = to_categorical(np.squeeze(train_y,axis=0))

    # 打乱数据 (1,200,chans_num)
    rand_inx = np.random.permutation(range(train_x.shape[0]))  
    train_x = train_x[rand_inx]
    train_y = train_y[rand_inx]
    model_input1 = (train_x, train_y), (test_x, test_y)

    if dataformat == '3D':
        return model_input1

    # change shape from (1,200,chans_num) to (chans_num，200)
    train_x = np.swapaxes(np.squeeze(train_x,axis=1), 1, 2)
    test_x = np.swapaxes(np.squeeze(test_x,axis=1), 1, 2)
    model_input2 = (train_x, train_y), (test_x, test_y)
    if dataformat == 'true_2D':
        return model_input2

    # change shape from (chans_num，200) to (chans_num，200，1)
    train_x = np.expand_dims(train_x, axis=3)  
    test_x = np.expand_dims(test_x, axis=3)
    model_input3 = (train_x, train_y), (test_x, test_y)
    if dataformat == '2D':
        return model_input3



def load_sub(sub):
    # 每个分支的输入组成一个列表返回
    Train_x =  []
    Test_x = []
    Train_y = []
    Test_y = []
    
    if band_pass:
        # 先提取出每个bandpass的数据，组成一个列表：5*（8090，9，200，1）
        for i in range(len(data_file_list)):
            (train_x, train_y), (test_x, test_y) = load_one_sub(sub,chans_index[i],dataformat_list[i],data_file_list[i])
            print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
            Train_x.append(train_x)
            Train_y= train_y
            Test_x.append(test_x)
            Test_y = test_y
    else:
        for i,chan in enumerate(chans_index):
            (train_x, train_y), (test_x, test_y) = load_one_sub(sub,chan,dataformat_list[i],data_file_list[0])
            print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
            Train_x.append(train_x)
            Train_y= train_y
            Test_x.append(test_x)
            Test_y = test_y
    # 若5个一起输入
    if len(model_names)==1:
        if dataformat_list[0]=='2D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=1)]
        if dataformat_list[0]=='3D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=-1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=-1)]
        print('shape after together')
        print(Train_x[0].shape, Test_x[0].shape)
    return (Train_x,Train_y), (Test_x,Test_y)




