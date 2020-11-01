# coding=utf-8
import os
import scipy.io as sio
from keras.utils import to_categorical
import numpy as np
from config import *


def load_one_sub(sub, chan, dataformat, datafile):
    # '2D':eletrodes_to_chans  (1,200,9)
    # '3D':eletrodes_to_high   (9,200,1)

    # load total data
    if band_pass:
        print(
            'Bandpass: load data and label from \033[1;32;m{}\033[0m'.format(datafile))
        data_file = sio.loadmat(datafile)  # ['TotalCell']
        # [0,0]  # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 886)
        data = data_file['Data']
        # [0,0]  # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 886)
        label = data_file['Label']
        # data = file['Data']  # 以前的数据的处理方式 (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 887)
    else:
        file = sio.loadmat(datafile)  # ['TotalCell']
        print('load data and label from \033[1;32;m{}\033[0m'.format(datafile))
        # [0,0] # (1, 8) (200, 62, 1198) 下标为[0，4]的数据：(200, 62, 887)
        data = file['Data']
        label = file['Label']  # [0,0] # (1, 8) (1, 1198) 下标为[0，4]的数据：(1, 887)

    # extract test data
    tmp = data[0, sub]  # (200, 62, 1198)
    tmp = tmp[:, chan, :]  # (200, chans, 1198)
    test_x = np.expand_dims(np.transpose(tmp, (2, 0, 1)),
                            axis=1)  # (1198,1,200, chans)
    test_y = to_categorical(label[0, sub].reshape((tmp.shape[2], 1)))

    # extract training data
    train_x = None
    train_y = None
    train_sub_range = [i for i in range(
        0, sub)] + [j for j in range(sub+1, total_sub_num)]
    for i in train_sub_range:
        tmp = data[0, i]
        tmp = tmp[:, chan, :]
        tmp_l = label[0, i]
        if train_x is None:
            train_x = tmp
            train_y = tmp_l
        else:
            train_x = np.concatenate([train_x, tmp], axis=2)
            train_y = np.concatenate([train_y, tmp_l], axis=1)
    train_x = np.transpose(np.expand_dims(train_x, axis=0), (3, 0, 1, 2))
    train_y = to_categorical(np.squeeze(train_y, axis=0))

    # 打乱数据 (1,200,chans_num)
    rand_inx = np.random.permutation(range(train_x.shape[0]))
    train_x = train_x[rand_inx]
    train_y = train_y[rand_inx]
    model_input1 = (train_x, train_y), (test_x, test_y)

    if dataformat == '3D':
        return model_input1

    # change shape from (1,200,chans_num) to (chans_num，200)
    train_x = np.swapaxes(np.squeeze(train_x, axis=1), 1, 2)
    test_x = np.swapaxes(np.squeeze(test_x, axis=1), 1, 2)
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
    Train_x = []
    Test_x = []
    Train_y = []
    Test_y = []

    if band_pass:
        # 先提取出每个bandpass的数据，组成一个列表：5*（8090，9，200，1）
        for i in range(len(data_file_list)):
            (train_x, train_y), (test_x, test_y) = load_one_sub(
                sub, chans_index[i], dataformat_list[i], data_file_list[i])
            print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            Train_x.append(train_x)
            Train_y = train_y
            Test_x.append(test_x)
            Test_y = test_y
    else:
        for i, chan in enumerate(chans_index):
            (train_x, train_y), (test_x, test_y) = load_one_sub(
                sub, chan, dataformat_list[i], data_file_list[0])
            print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
            Train_x.append(train_x)
            Train_y = train_y
            Test_x.append(test_x)
            Test_y = test_y
    # 若5个一起输入
    if len(model_names) == 1:
        if dataformat_list[0] == '2D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=1)]
        if dataformat_list[0] == '3D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=-1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=-1)]
        print('shape after together')
        print(Train_x[0].shape, Test_x[0].shape)
    return (Train_x, Train_y), (Test_x, Test_y)


def load_sub_3D(sub):
    Train_x = []
    Test_x = []
    Train_y = []
    Test_y = []

    for i, chan in enumerate(chans_index):
        (train_x, train_y), (test_x, test_y) = load_one_sub(sub, chan, dataformat_list[i], data_file_list[0])
        print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        Train_x.append(train_x)
        Train_y = train_y
        Test_x.append(test_x)
        Test_y = test_y
    # 若5个一起输入，把样本数量拼起来
    if len(model_names) == 1:
        if dataformat_list[0] == '2D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=1)]
        if dataformat_list[0] == '3D':
            Train_x = [np.concatenate([valu for valu in Train_x], axis=-1)]
            Test_x = [np.concatenate([valu for valu in Test_x], axis=-1)]
        print('shape after together')
        print(Train_x[0].shape, Test_x[0].shape)
    Train_x = np.squeeze(Train_x)
    Test_x = np.squeeze(Test_x)
    Train_X = np.zeros(shape=(Train_x.shape[0], 9, 9, 400,1))
    Test_X = np.zeros(shape=(Test_x.shape[0], 9, 9, 400,1))
    index_to_locate = {
        1: [0, 3],
        2: [1, 2],
        3: [1, 4],
        4: [2, 3],
        5: [3, 2],
        6: [3, 3],
        7: [4, 0],
        8: [4, 2],
        9: [4, 4],
        10: [5, 1],
        11: [5, 3],
        12: [6, 0],
        13: [6, 2],
        14: [6, 4],
        15: [7, 4],
        16: [8, 3],
        17: [0, 5],
        18: [1, 6],
        19: [2, 5],
        20: [3, 5],
        21: [3, 7],
        22: [4, 6],
        23: [4, 8],
        24: [5, 5],
        25: [5, 7],
        26: [6, 6],
        27: [6, 8],
        28: [8, 6]
    }
    for sample in range(Train_x.shape[0]):
        for i in range(28):
            locate = index_to_locate[i+1]
            Train_X[sample,locate[0],locate[1],:,0] = Train_x[sample,i,:]
    
    for sample in range(Test_x.shape[0]):
        for i in range(28):
            locate = index_to_locate[i+1]
            Test_X[sample,locate[0],locate[1],:,0] = Test_x[sample,i,:]

    print('shape after 3D transform')
    print(Train_X.shape, Test_X.shape)
    print(Train_y.shape, Test_y.shape)

    
    if is_interpolate:
        from scipy import interpolate
        import matplotlib as mpl
        import pylab as pl


        # 对每一个样本都进行插值
        for i in range(Train_X.shape[0]):
            # 对每一个 9*9 都进行插值，共400个
            for j in range(Train_X.shape[-2]):
                # 生成二维坐标
                x = np.array([val[0] for val in index_to_locate.values()])
                y = np.array([val[1] for val in index_to_locate.values()])
                fvals_99 = Train_X[i,:,:,j,0] # 9*9矩阵
                fvals = Train_x[i,:,j] # 28的向量

                # # 画出原来的9*9矩阵
                # pl.subplot(121)
                # im1=pl.imshow(fvals_99, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")#pl.cm.jet
                # pl.colorbar(im1)

                # 进行3次插值
                # newfunc = interpolate.interp2d(x, y, fvals, kind='cubic') # 返回的是一个函数
                # 因为原来的电极数据不是规则的网格，用这种方法无法对其进行插值
                fnew = interp2d_station_to_grid(x, y,fvals)#仅仅是y值   100*100的值
                Train_X[i,:,:,j,0] = fnew

                # # 画出查之后的9*9矩阵
                # pl.subplot(122)
                # im2=pl.imshow(fnew, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
                # pl.colorbar(im2)
                # pl.show()

        # 处理测试样本的插值
        for i in range(Test_X.shape[0]):
            # 对每一个 9*9 都进行插值，共400个
            for j in range(Test_X.shape[-2]):
                # 生成二维坐标
                x = np.array([val[0] for val in index_to_locate.values()])
                y = np.array([val[1] for val in index_to_locate.values()])
                fvals_99 = Test_X[i,:,:,j,0] # 9*9矩阵
                fvals = Test_x[i,:,j] # 28的向量

                # # 调试时画出原来的9*9矩阵
                # pl.subplot(121)
                # im1=pl.imshow(fvals_99, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")#pl.cm.jet
                # pl.colorbar(im1)

                # 进行3次插值
                # newfunc = interpolate.interp2d(x, y, fvals, kind='cubic') # 返回的是一个函数
                # 因为原来的电极数据不是规则的网格，用这种方法无法对其进行插值
                fnew = interp2d_station_to_grid(x, y,fvals)#仅仅是y值   100*100的值
                Test_X[i,:,:,j,0] = fnew

                # # 调试时画出查之后的9*9矩阵
                # pl.subplot(122)
                # im2=pl.imshow(fnew, extent=[-1,1,-1,1], cmap=mpl.cm.hot, interpolation='nearest', origin="lower")
                # pl.colorbar(im2)
                # pl.show()
  

    print('shape after interpolate')
    print(Train_X.shape, Train_X.shape)
    print(Train_y.shape, Test_y.shape)

    return (Train_X, Train_y), (Test_X, Test_y)


# 改编自 https://blog.csdn.net/weixin_43718675/article/details/103497930
def interp2d_station_to_grid(lon,lat,data,method = 'cubic'):
    '''
    func : 将站点数据插值到等经纬度格点
    inputs:
        lon: 站点的经度
        lat: 站点的纬度
        data: 对应经纬度站点的 气象要素值
        method: 所选插值方法，默认 0.125
    return:
        
        [lon_grid,lat_grid,data_grid]
    '''
    from scipy.interpolate import griddata

    #step1: 先将 lon,lat,data转换成 n*1 的array数组
    lon = np.array(lon).reshape(-1,1)
    lat = np.array(lat).reshape(-1,1)
    data = np.array(data).reshape(-1,1)
    
    #shape = [n,2]
    points = np.concatenate([lon,lat],axis = 1)
    lon_grid, lat_grid = np.mgrid[0:9,0:9]
    
    #step3:进行网格插值
    grid_data = griddata(points,data,(lon_grid,lat_grid),method = method)
    grid_data = grid_data[:,:,0]
    
    # #保证纬度从上到下是递减的
    # if lat_grid[0,0]<lat_grid[1,0]:
    #     lat_grid = lat_grid[-1::-1]
    #     grid_data = grid_data[-1::-1]

    print('grid_data.shape',grid_data.shape)
    
    return grid_data
