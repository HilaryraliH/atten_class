# coding=utf-8
import numpy as np
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.constraints import max_norm
from config import *
from keras.engine.topology import Layer
from atten_layer import AttentionLayer, attach_attention_module

Samples = 200


class SeBlock(Layer):
    def __init__(self, reduction=4, **kwargs):
        super(SeBlock, self).__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):  # 构建layer时需要实现
        pass

    def call(self, inputs):
        print(inputs.shape)
        x = GlobalAveragePooling2D()(inputs)
        print('after GlobalAveragePooling2D, x.shape', x.shape)
        x = Dense(int(x.shape[-1]) // self.reduction,
                  use_bias=False, activation='relu')(x)
        x = Dense(int(inputs.shape[-1]), use_bias=False,
                  activation='hard_sigmoid')(x)
        return Multiply()([inputs, x])  # 给通道加权重
        # return inputs*x

# %% 输入为3D（1，200，chans）的模型
def Convention_2D(model_input, Chans, nb_classes=2):
    # article: Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI
    # # remain unchanged
    data = Conv2D(32, (1, 20), strides=(1, 2), activation='relu')(model_input)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = BatchNormalization()(data)
    data = Conv2D(16, (1, 10), activation='relu')(data)
    data = BatchNormalization()(data)
    data = Conv2D(8, (1, 5), activation='relu')(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(50, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def JNE_CNN(model_input, Chans, nb_classes=2):
    # article: Inter-subject transfer learning with an end-to-end deep convolutional neural network for EEG-based BCI
    # # remain unchanged
    data = Conv2D(60, (1, 4), strides=(1, 2), activation='relu')(model_input)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = Conv2D(40, (1, 3), activation='relu')(data)
    data = Conv2D(20, (1, 2), activation='relu')(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(100, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def JNE_CNN_SEBlock(model_input, chans, nb_classes=2):
    data = Conv2D(60, (1, 4), strides=(1, 2), activation='relu')(model_input)
    data = MaxPooling2D(pool_size=(1, 2))(data)
    data = Conv2D(40, (1, 3), activation='relu')(data)
    data = SeBlock()(data)
    data = Conv2D(20, (1, 2), activation='relu')(data)
    data = SeBlock()(data)
    data = Flatten()(data)
    data = Dropout(0.2)(data)
    data = Dense(100, activation='relu')(data)
    data = Dropout(0.3)(data)
    data = Dense(nb_classes, activation='softmax')(data)
    model = Model(model_input, data)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def Proposed_Conv(model_input, Chans, nb_classes=2):
    dropoutRate = 0.5
    norm_rate = 0.25

    block0 = Conv2D(8, (1, 5), padding='same', use_bias=False)(model_input)
    block0 = BatchNormalization()(block0)

    block1 = DepthwiseConv2D(
        (1, 20), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.))(block0)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(16, (1, 5), use_bias=False, padding='same')(
        block1)  # it's（1，16）before
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)  # it's（1，8）before
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def Proposed_Conv_R(model_input, Chans, nb_classes=2):
    dropoutRate = 0.5
    norm_rate = 0.25
    '''input1   = Input(shape = (1, Chans, Samples))'''

    block1 = Conv2D(8, (1, 5), padding='same', use_bias=False)(model_input)
    block1 = BatchNormalization(axis=-1)(block1)

    block1 = DepthwiseConv2D(
        (1, 20), use_bias=False, depth_multiplier=2, depthwise_constraint=max_norm(1.))(block1)
    # but when I use axis=1 before, it worked
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(
        16, (1, 16), use_bias=False, padding='same')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)

    # if use LSTM after Dropout, it will be confused by the order
    # but the AveragePooling2D may be worked, then I will check
    ''' block2 = AveragePooling2D((1, 4))(block2)# it's（1，8）before
    block2 = Dropout(dropoutRate)(block2)'''
    print(block2.shape)
    # block3 = Reshape((48, 16))(block2)
    block3 = Reshape((int(block2.shape[-2]), int(block2.shape[-1])))(block2)

    l_lstm_sent = LSTM(32, return_sequences=True)(block3)
    l_lstm_sent = LSTM(8, return_sequences=True)(l_lstm_sent)

    flatten = Flatten()(l_lstm_sent)
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    preds = Activation('softmax')(dense)

    # preds = Dense(nb_classes, name='dense', activation='softmax')(flatten)

    return Model(inputs=model_input, outputs=preds)

# %% 输入为2D（chans，200，1）的模型
def EEGNet_share_part(model_input, Chans, nb_classes=2, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(block1) 
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)
    flatten = Flatten()(block2)
    return Model(inputs=model_input, outputs=flatten)

def EEGNet(model_input, Chans, nb_classes=2, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(
        Chans, Samples, 1), use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding='same')(
        block1) 
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropoutRate)(block2)

    flatten = Flatten()(block2)
    dense = Dense(nb_classes,  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def DeepConvNet(model_input, Chans, nb_classes=2, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(25, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)  # it's channel first before

    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(
        epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def Smaller_DeepConvNet(model_input, Chans, nb_classes=2, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(8, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)  # it's channel first before

    block1 = Conv2D(8, (Chans, 1), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(
        epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(16, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(32, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(64, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def ShallowConvNet(model_input, Chans, nb_classes=2, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(40, (1, 25),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(model_input)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def Transpose_Net(model_input, Chans, nb_classes=2, dropoutRate=0.5):

    block1 = Conv2D(20, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)
    block1 = Conv2D(5, (Chans, 1), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(
        epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)
    print('block1.shape:', block1.shape)

    block1 = Reshape((int(block1.shape[-2]), int(block1.shape[-1]), 1))(block1)
    block1 = Permute((2, 1, 3))(block1)

    #block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    #block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(40, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(60, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(80, (1, 5), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))

# %% 建立多个融合的模型


def get_model_input(dataformat, chan_num):
    if dataformat == '2D':
        model_input = Input(shape=(chan_num, sample_points, 1))
    elif dataformat == '3D':
        model_input = Input(shape=(1, sample_points, chan_num))
    elif dataformat == 'true_2D':
        model_input = Input(shape=(chan_num, sample_points))
    return model_input


def erect_share_model():
    model_input = [None]*len(chans_num)
    for i in range(len(chans_num)):
        model_input[i] = get_model_input(dataformat_list[i], chans_num[i])
    # 因每个输入的channal不同，所以当模型中只有卷积的时候，才能共享，若包含全连接，则不能共享
    # 此处可以将EEGNet的卷积部分提出来共享，后面再接上相同的结构，再融合
    # 建立EEGNet，注意此时的输入，在channel的维度为none（但这样就不能建立EEGNet了呀，是需要chan这个具体参数的）
    # 所以最终还是对输入的几个分支形状做了调整，少了相应数目的channel，就可以直接共享了
    '''输入 channal 不一样时的 share 思路'''
    # 从EEGNet中提取出卷积部分，封装为一个共享层的模式
    # 共享层接受多个输入
    # 将多个输出进行连接，并进行分类
    '''最终的 share-model'''
    # 建立 share-model
    share_model = EEGNet_share_part(model_input[0], chans_num[0])
    # 左右输入
    out_list = [None]*len(chans_num)
    for i in range(len(chans_num)):
        out_list[i] = share_model(model_input[i])

    my_concatenate = Concatenate()(out_list)
    dense = Dense(2,  kernel_constraint=max_norm(0.25))(my_concatenate)
    
    return Model(model_input, dense)





def erect_single_model():
    chan_num = np.sum(np.array(chans_num))
    model_input = get_model_input(dataformat_list[0], chan_num)
    model = eval(model_names[0])(model_input, chan_num)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def erect_n_branch_model():
    model_input = [None]*len(model_names)
    model = [None]*len(model_names)
    for i in range(len(model_names)):
        model_input[i] = get_model_input(dataformat_list[i], chans_num[i])
    for i in range(len(model_names)):
        model[i] = eval(model_names[i])(model_input[i], chans_num[i])

    my_concatenate = Concatenate()(
        [model[i].layers[-3].output for i in range(len(model_names))])
    # my_concatenate = Dense(100)(my_concatenate)
    pre = Dense(2, activation='softmax')(my_concatenate)
    model = Model(model_input, pre)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def erect_n_branch_model_with_attention():
    model_input = [None]*len(model_names)
    model = [None]*len(model_names)
    concaten = [None]*len(model_names)
    for i in range(len(model_names)):
        model_input[i] = get_model_input(dataformat_list[i], chans_num[i])
    for i in range(len(model_names)):
        model[i] = eval(model_names[i])(model_input[i], chans_num[i])

    # 注意力及之前先进行维度扩张, model[i].layers[-3].output_shape: (1600,)
    # 若用JNE中的注意力机制，则要变为（1，1600）再拼起来； 若接在RNN后的注意力机制（CRAM中的那个），则要变为（1600，1）再拼起来，但这里没有用到
    # for i in range(len(model_names)):
    #     concaten[i] = Reshape(
    #         (1, model[i].layers[-3].output_shape[-1]))(model[i].layers[-3].output)
    # my_concatenate = Concatenate(
    #     axis=-2)([concaten[i] for i in range(len(model_names))])
    # my_concatenate = AttentionLayer()(my_concatenate)  # out: (none, 1600)

    # 若用SENet或者CBAM，则要变为（1，1600，1）,再在-1这一维拼起来
    # 其输出和 （1，1600，5）一样，所以接下来应该如何设计网络呢,用conv2D试试
    for i in range(len(model_names)):
        concaten[i] = Reshape(
            (1, model[i].layers[-3].output_shape[-1], 1))(model[i].layers[-3].output)
    my_concatenate = Concatenate(
        axis=-1)([concaten[i] for i in range(len(model_names))])
    my_concatenate = attach_attention_module(
        my_concatenate, 'cbam_block')  # se_block or cbam_block
    # my_concatenate = Conv2D(1, (1, 20),padding='same', activation='relu')(my_concatenate)
    # my_concatenate = MaxPool2D((1,3))(my_concatenate)
    my_concatenate = Flatten()(my_concatenate)

    # my_concatenate = Dense(100)(my_concatenate)
    pre = Dense(2, activation='softmax')(my_concatenate)
    model = Model(model_input, pre)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
