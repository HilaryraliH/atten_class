# coding=utf-8
import os
import scipy.io as sio
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from config import *

def load_one_sub(sub,chan,dataformat,datafile):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    if band_pass:
        label_file = sio.loadmat(data_dir+'62.mat')
        data_file = sio.loadmat(datafile)
        data = data_file['total_data']  # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 886)
        label = label_file['Label']  # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 886)
    else:
        file = sio.loadmat(data_dir+'62.mat')
        data = file['Data']  # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 886)
        label = file['Label']

    # extract test data
    tmp = data[0, sub]  # (200, 62, 1198)
    tmp = tmp[:,chan,:] # (200, chans, 1198)
    nums = tmp.shape[2]
    test_x = np.zeros((nums, 1, tmp.shape[0], tmp.shape[1]))
    for j in range(nums):
        test_x[j, 0, :, :] = tmp[:, :, j]
    test_y = label[0, sub].reshape((tmp.shape[2], 1))
    test_y = to_categorical(test_y)

    # extract training data
    total_nums = 0
    chans_num = 0
    for j in range(total_sub_num):
        tmp_train = data[0, j]
        tmp_train = tmp_train[:,chan,:]  # (200, chans, 1198)
        chans_num = tmp_train.shape[1]
        if j != sub:
            total_nums += tmp_train.shape[2]
    train_x = np.zeros((total_nums, 1, 200, chans_num))
    train_y = np.zeros((total_nums,))
    indx = 0
    for j in range(total_sub_num):
        tmp_train = data[0, j]
        tmp_train = tmp_train[:,chan,:]  # (200, chans, 1198)
        if j != sub:
            tmp_x = tmp_train  # (200, chans, 1198)
            tmp_y = label[0, j]  # (1, 1198)
            nums = tmp_x.shape[2]
            for k in range(nums):
                train_x[indx, 0, :, :] = tmp_x[:, :, k]
                train_y[indx] = tmp_y[0, k]
                indx += 1
    train_y = train_y.reshape((total_nums, 1))
    train_y = to_categorical(train_y)

    # 打乱数据 (1,200,chans_num)
    rand_inx = np.random.permutation(range(train_x.shape[0]))  
    train_x = train_x[rand_inx]
    train_y = train_y[rand_inx]
    model_input1 = (train_x, train_y), (test_x, test_y)

    if dataformat == '3D':
        return model_input1

    # change shape from (1,200,chans_num) to (chans_num，200)
    train_x = np.swapaxes(np.squeeze(train_x), 1, 2)  
    test_x = np.swapaxes(np.squeeze(test_x), 1, 2)
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
    
    if band_pass:
        # 先提取出每个bandpass的数据，组成一个列表：5*（8090，9，200，1）
        for i in range(len(data_file_list)):
            (train_x, train_y), (test_x, test_y) = load_one_sub(sub,chans_index[i],dataformat_list[i],data_file_list[i])
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
    else:
        # 如果多个分支都是一样的, 通道选择和数据
        if len(set(select_chan_way))==1 and len(set(model_names))!=1:
            (train_x, train_y), (test_x, test_y) = load_one_sub(sub,chans_index[0],dataformat_list[0],data_file_list[0])
            Train_x = [train_x]*len(select_chan_way)
            Test_x = [test_x]*len(select_chan_way)
        # 如果多个分支，每个分支不一样
        else:
            for i,chan in enumerate(chans_index):
                (train_x, train_y), (test_x, test_y) = load_one_sub(sub,chan,dataformat_list[i],data_file_list[i])
                print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
                Train_x.append(train_x)
                Train_y= train_y
                Test_x.append(test_x)
                Test_y = test_y
        # 若为一个分支，则按实际来加，但也作为list返回
        if len(model_names) == 1:
            if dataformat_list[0]=='2D':
                Train_x = [np.concatenate([valu for valu in Train_x], axis=1)]
                Test_x = [np.concatenate([valu for valu in Test_x], axis=1)]
            if dataformat_list[0]=='3D':
                Train_x = [np.concatenate([valu for valu in Train_x], axis=-1)]
                Test_x = [np.concatenate([valu for valu in Test_x], axis=-1)]
            print('shape after together')
            print(Train_x[0].shape, Test_x[0].shape)
        return (Train_x,Train_y), (Test_x,Test_y)



def check_path(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('make dir error')
            return


def mk_save_dir(once):
    save_dir = None
    root_dir = None
    if band_pass:
        is_band_pass = 'bandpass'
    else:
        is_band_pass = None
    root_dir = 'results\\' + str(model_names) + str(select_chan_way)+input_way+ '_'+str(is_band_pass)+'\\'
    save_dir = root_dir + '第{}次'.format(once) + '\\'
    check_path(save_dir)
    check_path(root_dir)
    return root_dir,save_dir
