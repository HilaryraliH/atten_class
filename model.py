# coding=utf-8
import numpy as np
import keras.backend as K
from keras.layers import *
from keras.models import Model
from keras.constraints import max_norm
from config import *
from keras.engine.topology import Layer
from atten_layer import AttentionLayer, attach_attention_module
from keras.optimizers import Adam

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
def EEGNet_share_part(model_input, Chans, nb_classes=2, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16,
                      norm_rate=0.25):
    block1 = Conv2D(F1, (1, kernLength), padding='same', input_shape=(Chans, Samples, 1), use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)  # I'm not sure the axis, axis=1 before

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=D, depthwise_constraint=max_norm(1.))(block1)
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
    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


def EEGNet_smaller(model_input, Chans, nb_classes=2, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    block1 = Conv2D(8, (Chans, 1), use_bias=False)(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)

    block1 = Conv2D(16, (1, 3))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 3))(block1)

    block1 = Conv2D(32, (1, 3))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 3))(block1)

    block1 = Conv2D(32, (1, 3))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 3))(block1)

    flatten = Flatten()(block1)

    dense = Dense(nb_classes, kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


# 保持原来的参数
def DeepConvNet(model_input, Chans, nb_classes=2, dropoutRate=0.5):
    # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
    # changed as the comments
    block1 = Conv2D(25, (1, 10), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
        model_input)  # it's channel first before

    block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(
        2., axis=(0, 1, 3)))(block1)
    block1 = BatchNormalization(
        epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
    block1 = Activation('elu')(block1)

    block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 10), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)

    block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 10), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)

    block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 10), kernel_constraint=max_norm(
        2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)

    block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=model_input, outputs=softmax)


# def DeepConvNet(model_input, Chans, nb_classes=2, dropoutRate=0.5):
#     # article: EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces
#     # changed as the comments
#     block1 = Conv2D(25, (1, 5), input_shape=(Chans, Samples, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(
#         model_input)  # it's channel first before
#
#     block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(
#         2., axis=(0, 1, 2)))(block1)
#     block1 = BatchNormalization(
#         epsilon=1e-05, momentum=0.1)(block1)  # it's axis=1 before
#     block1 = Activation('elu')(block1)
#
#     block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
#     block1 = Dropout(dropoutRate)(block1)
#
#     block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(
#         2., axis=(0, 1, 2)))(block1)
#     block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
#     block2 = Activation('elu')(block2)
#
#     block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
#     block2 = Dropout(dropoutRate)(block2)
#
#     block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(
#         2., axis=(0, 1, 2)))(block2)
#     block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
#     block3 = Activation('elu')(block3)
#
#     block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
#     block3 = Dropout(dropoutRate)(block3)
#
#     block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(
#         2., axis=(0, 1, 2)))(block3)
#     block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
#     block4 = Activation('elu')(block4)
#
#     block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
#     block4 = Dropout(dropoutRate)(block4)
#
#     flatten = Flatten()(block4)
#     dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
#     softmax = Activation('softmax')(dense)
#
#     return Model(inputs=model_input, outputs=softmax)


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

    # block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    # block1 = Dropout(dropoutRate)(block1)

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
    if is_3D:
        model_input = Input(shape=(9, 9, sample_points, 1))
    elif dataformat == '2D':
        model_input = Input(shape=(chan_num, sample_points, 1))
    elif dataformat == '3D':
        model_input = Input(shape=(1, sample_points, chan_num))
    elif dataformat == 'true_2D':
        model_input = Input(shape=(chan_num, sample_points))

    return model_input


# 因每个输入的channal不同，所以当模型中只有卷积的时候，才能共享，若包含全连接，则不能共享
# 此处可以将EEGNet的卷积部分提出来共享，后面再接上相同的结构，再融合
# 建立EEGNet，注意此时的输入，在channel的维度为none（但这样就不能建立EEGNet了呀，是需要chan这个具体参数的）
# 所以最终还是对输入的几个分支形状做了调整，少了相应数目的channel，就可以直接共享了
def erect_share_model():
    model_input = [None] * 3
    for i in range(3):
        model_input[i] = Input(shape=(chans_num[0], 200, 1))
    # 建立 share-model
    share_model = EEGNet_share_part(model_input[0], chans_num[0])
    # 分支输入
    out_list = [None] * len(chans_num)
    for i in range(len(chans_num)):
        out_list[i] = share_model(model_input[i])

    my_concatenate = Concatenate()(out_list)
    dense = Dense(2, kernel_constraint=max_norm(0.25))(my_concatenate)
    model = Model(model_input, dense)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def erect_single_model():
    chan_num = np.sum(np.array(chans_num))
    model_input = get_model_input(dataformat_list[0], chan_num)
    model = eval(model_names[0])(model_input, chan_num)
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def erect_n_branch_model():
    model_input = [None] * len(model_names)
    model = [None] * len(model_names)
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
    model_input = [None] * len(model_names)
    model = [None] * len(model_names)
    concaten = [None] * len(model_names)
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


def Time_Small_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 1), strides=(2, 2, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 1), strides=(2, 2, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Large_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_branches(model_input, chan_num):
    # smalll
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 1), strides=(2, 2, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 1), strides=(2, 2, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put_small = Dense(2, activation='softmax')(dense2)

    # mid
    block4 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense4 = Dense(32)(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4)

    out_put_mid = Dense(2, activation='softmax')(dense4)

    # big
    block7 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    block9 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block8)
    block9 = BatchNormalization()(block9)
    block9 = Activation('relu')(block9)

    flattened = Flatten()(block9)
    dense5 = Dense(32)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense6 = Dense(32)(dense5)
    dense6 = BatchNormalization()(dense6)
    dense6 = Activation('relu')(dense6)

    out_put_big = Dense(2, activation='softmax')(dense6)

    # add and classification
    dense_out = Add()([out_put_small, out_put_mid, out_put_big])
    classifiction = Softmax()(dense_out)
    model = Model(model_input, classifiction)
    return model


def Time_branches_with_attention(model_input, chan_num):
    # smalll
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 1), strides=(2, 2, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 1), strides=(2, 2, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put_small = Dense(2, activation='softmax')(dense2)

    # mid
    block4 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense4 = Dense(32)(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4)

    out_put_mid = Dense(2, activation='softmax')(dense4)

    # big
    block7 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    block9 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block8)
    block9 = BatchNormalization()(block9)
    block9 = Activation('relu')(block9)

    flattened = Flatten()(block9)
    dense5 = Dense(32)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense6 = Dense(32)(dense5)
    dense6 = BatchNormalization()(dense6)
    dense6 = Activation('relu')(dense6)

    out_put_big = Dense(2, activation='softmax')(dense6)

    # Attention block and classification
    out_put_small = Reshape((1, out_put_small.shape[-1], 1))(out_put_small)
    out_put_mid = Reshape((1, out_put_mid.shape[-1], 1))(out_put_mid)
    out_put_big = Reshape((1, out_put_big.shape[-1], 1))(out_put_big)

    my_concatenate = Concatenate(axis=-1)([out_put_small, out_put_mid, out_put_big])
    my_concatenate = attach_attention_module(my_concatenate, 'se_block')  # se_block or cbam_block
    my_concatenate = Flatten()(my_concatenate)
    pre = Dense(2, activation='softmax')(my_concatenate)
    model = Model(model_input, pre)
    return model


def Time_branches_feature_concat(model_input, chan_num):
    # smalll
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 1), strides=(2, 2, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 1), strides=(2, 2, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense_small = Dense(32)(dense1)

    # mid
    block4 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense_mid = Dense(32)(dense3)

    # big
    block7 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    block9 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block8)
    block9 = BatchNormalization()(block9)
    block9 = Activation('relu')(block9)

    flattened = Flatten()(block9)
    dense5 = Dense(32)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense_big = Dense(32)(dense5)

    # add and classification
    dense_cat = Concatenate(axis=1)([dense_small, dense_mid, dense_big])
    dense_out = Dense(2, activation='softmax')(dense_cat)
    model = Model(model_input, dense_out)
    return model


def Time_branches_feature_concat_attention(model_input, chan_num):
    # smalll
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 1), strides=(2, 2, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 1), strides=(2, 2, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense_small = Dense(32)(dense1)

    # mid
    block4 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense_mid = Dense(32)(dense3)

    # big
    block7 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    block9 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block8)
    block9 = BatchNormalization()(block9)
    block9 = Activation('relu')(block9)

    flattened = Flatten()(block9)
    dense5 = Dense(32)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense_big = Dense(32)(dense5)

    # Attention block and classification
    out_put_small = Reshape((1, dense_small.shape[-1], 1))(dense_small)
    out_put_mid = Reshape((1, dense_mid.shape[-1], 1))(dense_mid)
    out_put_big = Reshape((1, dense_big.shape[-1], 1))(dense_big)

    my_concatenate = Concatenate(axis=-1)([out_put_small, out_put_mid, out_put_big])
    my_concatenate = attach_attention_module(my_concatenate, 'cbam_block')  # se_block or cbam_block
    my_concatenate = Flatten()(my_concatenate)
    pre = Dense(2, activation='softmax')(my_concatenate)
    model = Model(model_input, pre)

    return model


def Spatial_Small_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (2, 2, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Spatial_Mid_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(1, 1, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (3, 3, 3), strides=(1, 1, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (3, 3, 3), strides=(1, 1, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Spatial_Large_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (4, 4, 5), strides=(1, 1, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (4, 4, 3), strides=(1, 1, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # block3 = Conv3D(64,(4,4,3),strides=(1,1,2))(block2)
    # block3 = BatchNormalization()(block3)
    # block3 = Activation('relu')(block3)

    flattened = Flatten()(block2)
    dense1 = Dense(100)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Spatial_branches(model_input, chan_num):
    # smalll
    block1 = Conv3D(16, (2, 2, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put_small = Dense(2, activation='softmax')(dense2)

    # mid
    block4 = Conv3D(16, (3, 3, 5), strides=(1, 1, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (3, 3, 3), strides=(1, 1, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (3, 3, 3), strides=(1, 1, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense4 = Dense(32)(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4)

    out_put_mid = Dense(2, activation='softmax')(dense4)

    # big
    block7 = Conv3D(16, (4, 4, 5), strides=(1, 1, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (4, 4, 3), strides=(1, 1, 2))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    # block9 = Conv3D(64,(4,4,3),strides=(1,1,2))(block8)
    # block9 = BatchNormalization()(block9)
    # block9 = Activation('relu')(block9)

    flattened = Flatten()(block8)
    dense5 = Dense(100)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense6 = Dense(32)(dense5)
    dense6 = BatchNormalization()(dense6)
    dense6 = Activation('relu')(dense6)

    out_put_big = Dense(2, activation='softmax')(dense6)

    # add and classification
    dense_out = Add()([out_put_small, out_put_mid, out_put_big])
    classifiction = Softmax()(dense_out)
    model = Model(model_input, classifiction)
    return model


def Time_Spatial_branches(model_input, chan_num):
    # Only time (no spatial)
    block1 = Conv3D(16, (1, 1, 5), strides=(1, 1, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (1, 1, 3), strides=(1, 1, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (1, 1, 3), strides=(1, 1, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put_no_Spatial = Dense(2, activation='softmax')(dense2)

    # mid Spatial
    # block4 = Conv3D(16, (3, 3, 5), strides=(1, 1, 4))(model_input)
    # block4 = BatchNormalization()(block4)
    # block4 = Activation('relu')(block4)
    #
    # block5 = Conv3D(32, (3, 3, 3), strides=(1, 1, 2))(block4)
    # block5 = BatchNormalization()(block5)
    # block5 = Activation('relu')(block5)
    #
    # block6 = Conv3D(64, (3, 3, 3), strides=(1, 1, 2))(block5)
    # block6 = BatchNormalization()(block6)
    # block6 = Activation('relu')(block6)
    #
    # flattened = Flatten()(block6)
    # dense3 = Dense(32)(flattened)
    # dense3 = BatchNormalization()(dense3)
    # dense3 = Activation('relu')(dense3)
    #
    # dense4 = Dense(32)(dense3)
    # dense4 = BatchNormalization()(dense4)
    # dense4 = Activation('relu')(dense4)
    #
    # out_put_mid_Spatial = Dense(2, activation='softmax')(dense4)

    # big Spatial
    block7 = Conv3D(16, (4, 4, 5), strides=(1, 1, 4))(model_input)
    block7 = BatchNormalization()(block7)
    block7 = Activation('relu')(block7)

    block8 = Conv3D(32, (4, 4, 3), strides=(1, 1, 2))(block7)
    block8 = BatchNormalization()(block8)
    block8 = Activation('relu')(block8)

    flattened = Flatten()(block8)
    dense5 = Dense(100)(flattened)
    dense5 = BatchNormalization()(dense5)
    dense5 = Activation('relu')(dense5)

    dense6 = Dense(32)(dense5)
    dense6 = BatchNormalization()(dense6)
    dense6 = Activation('relu')(dense6)

    out_put_big_Spatial = Dense(2, activation='softmax')(dense6)

    # no Time
    block1 = Conv3D(16, (3, 3, 5), strides=(1, 1, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (3, 3, 1), strides=(1, 1, 1))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (3, 3, 1), strides=(1, 1, 1))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put_no_time = Dense(2, activation='softmax')(dense2)

    # mid Time
    block4 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    block5 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(block4)
    block5 = BatchNormalization()(block5)
    block5 = Activation('relu')(block5)

    block6 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block5)
    block6 = BatchNormalization()(block6)
    block6 = Activation('relu')(block6)

    flattened = Flatten()(block6)
    dense3 = Dense(32)(flattened)
    dense3 = BatchNormalization()(dense3)
    dense3 = Activation('relu')(dense3)

    dense4 = Dense(32)(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Activation('relu')(dense4)

    out_put_mid_time = Dense(2, activation='softmax')(dense4)

    # # big Time
    # block7 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    # block7 = BatchNormalization()(block7)
    # block7 = Activation('relu')(block7)
    #
    # block8 = Conv3D(32, (2, 2, 5), strides=(2, 2, 4))(block7)
    # block8 = BatchNormalization()(block8)
    # block8 = Activation('relu')(block8)
    #
    # block9 = Conv3D(64, (2, 2, 5), strides=(2, 2, 4))(block8)
    # block9 = BatchNormalization()(block9)
    # block9 = Activation('relu')(block9)
    #
    # flattened = Flatten()(block9)
    # dense5 = Dense(32)(flattened)
    # dense5 = BatchNormalization()(dense5)
    # dense5 = Activation('relu')(dense5)
    #
    # dense6 = Dense(32)(dense5)
    # dense6 = BatchNormalization()(dense6)
    # dense6 = Activation('relu')(dense6)
    #
    # out_put_big_time = Dense(2, activation='softmax')(dense6)

    # add and classification
    dense_out = Add()([out_put_no_Spatial, out_put_mid_time, out_put_big_Spatial, out_put_no_time])
    classifiction = Softmax()(dense_out)
    model = Model(model_input, classifiction)
    return model


def Big_Three_D_model(model_input, chan_num):
    block1 = Conv3D(16, (5, 5, 10), strides=(1, 1, 5))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (3, 3, 5), strides=(1, 1, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_0line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_0line_RNN(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    rnn_outdb6 = Activation('relu')(db6)

    # flatten and classification
    print('db6.shape',db6.shape) # 1,1,24,64

    rnn_input = Reshape((db6.shape[-1],db6.shape[-2]))(db6)
    rnn_out = GRU(32)(rnn_input)
    print(rnn_out.shape)
    # flattened = Flatten()(rnn_out)
    #
    # dense1 = Dense(32)(flattened)
    # dense1 = BatchNormalization()(dense1)
    # dense1 = Activation('relu')(dense1)

    out_put = Dense(2, activation='softmax')(rnn_out)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_1line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    conca = Concatenate(axis=-1)([db1, db2])

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(conca)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    conca = Concatenate(axis=-1)([db3, db4])

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(conca)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    conca = Concatenate(axis=-1)([db5, db6])

    # flatten and classification
    flattened = Flatten()(conca)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    conca_block1_db1 = Concatenate(axis=-1)([block1, db1])

    db2 = Conv3D(32, (2, 2, 3), padding='same')(conca_block1_db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    conca_block1_db1_db2 = Concatenate(axis=-1)([block1, db1, db2])

    # transition bolock 1 : inputsize: 4,4,99,64
    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(conca_block1_db1_db2)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)
    conca_block2_db3 = Concatenate(axis=-1)([block2, db3])

    db4 = Conv3D(32, (2, 2, 3), padding='same')(conca_block2_db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    conca_block2_db3_db4 = Concatenate(axis=-1)([block2, db3, db4])

    # transition bolock 2 : inputsize: 4,4,99,64
    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(conca_block2_db3_db4)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)
    conca_block3_db5 = Concatenate(axis=-1)([block3, db5])

    db6 = Conv3D(64, (2, 2, 3), padding='same')(conca_block3_db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    conca_block3_db5_db6 = Concatenate(axis=-1)([block3, db5, db6])

    # flatten and classification
    flattened = Flatten()(conca_block3_db5_db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_resnet(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    add_block1_db2 = Add()([block1, db2])

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(add_block1_db2)
    db3 = BatchNormalization()(db3)
    block2 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    add_block2_db4 = Add()([block2, db4])

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(add_block2_db4)
    db5 = BatchNormalization()(db5)
    block3 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    add_block3_db6 = Add()([block3, db6])

    # flatten and classification
    flattened = Flatten()(add_block3_db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3Conv_0line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3Conv_0line_one_denseLayer(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    # dense2 = Dense(32)(dense1)
    # dense2 = BatchNormalization()(dense2)
    # dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense1)

    model = Model(model_input, out_put)
    return model


def No_spatial_3D(model_input, chan_num):
    # Only time (no spatial)
    block1 = Conv3D(16, (1, 1, 5), strides=(1, 1, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    block2 = Conv3D(32, (1, 1, 3), strides=(1, 1, 2))(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    block3 = Conv3D(64, (1, 1, 3), strides=(1, 1, 2))(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    flattened = Flatten()(block3)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    classifiction = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, classifiction)
    return model


def Time_Mid_3D_densenet_3Conv_small_Resnet(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    add_block1_db2 = Add()([block1, db2])

    # transition bolock 2 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(add_block1_db2)
    db3 = BatchNormalization()(db3)
    block2 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    add_block2_db4 = Add()([block2, db4])

    # transition bolock 3 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(add_block2_db4)
    db5 = BatchNormalization()(db5)
    block3 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    add_block3_db6 = Add()([block3, db6])

    # flatten and classification
    flattened = Flatten()(add_block3_db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3Conv_6small_Resnet(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)
    db1 = Add()([block1,db1])

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)
    db2 = Add()([db1,db2])

    db3 = Conv3D(16, (2, 2, 3), padding='same')(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    add_block1_db2 = Add()([db2, db3])

    # transition bolock 2 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(add_block1_db2)
    db3 = BatchNormalization()(db3)
    block2 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)
    db3 = Add()([block2,db3])

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)
    db4 = Add()([db3,db4])

    db5 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    add_block2_db4 = Add()([db4, db5])

    # transition bolock 3 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(add_block2_db4)
    db5 = BatchNormalization()(db5)
    block3 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)
    db5 = Add()([block3,db5])

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)
    db6 = Add()([db5,db6])

    db7 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db7 = BatchNormalization()(db7)
    db7 = Activation('relu')(db7)

    add_block3_db6 = Add()([db6, db7])

    # flatten and classification
    flattened = Flatten()(add_block3_db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_Big_Resnet(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    print('db6.shape', db6.shape)
    print('block1.shape', block1.shape)

    db7 = Conv3D(64, (4, 4, 7), strides=(1, 1, 4))(block1)
    print('db7.shape', db7.shape)

    big_res = Add()([db6, db7])

    # flatten and classification
    flattened = Flatten()(big_res)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3Conv_Big_Resnet(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    print('db6.shape', db6.shape)
    print('block1.shape', block1.shape)

    db7 = Conv3D(64, (4, 4, 7), strides=(1, 1, 4))(block1)
    print('db7.shape', db7.shape)

    big_res = Add()([db6, db7])

    # flatten and classification
    flattened = Flatten()(big_res)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_3Conv_line_between_block(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(16, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db2 = Conv3D(32, (1, 1, 1))(db2)  # 改变输出channel，方便之后相加
    db2 = AveragePooling3D((2,2,2))(db2)
    db4 = Add()([db2,db4])

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db4 = Conv3D(64,(1,1,1))(db4) # 改变输出channel，方便之后相加
    db4 = AveragePooling3D((2, 2, 2))(db4)
    db6 = Add()([db4,db6])

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_4Conv_0line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_4Conv_1line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db11 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db11 = BatchNormalization()(db11)
    db11 = Activation('relu')(db11)

    db12 = Conv3D(16, (2, 2, 3), padding='same')(db11)
    db12 = BatchNormalization()(db12)
    db12 = Activation('relu')(db12)

    db13 = Conv3D(16, (2, 2, 3), padding='same')(db12)
    db13 = BatchNormalization()(db13)
    db13 = Activation('relu')(db13)

    db14 = Conv3D(16, (2, 2, 3), padding='same')(db13)
    db14 = BatchNormalization()(db14)
    db14 = Activation('relu')(db14)

    db1_out = Add()([block1, db14])

    # transition bolock 1 : inputsize: 4,4,99,64
    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db1_out)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # dense block 2: inputsize: 4,4,99,64
    db21 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db21 = BatchNormalization()(db21)
    db21 = Activation('relu')(db21)

    db22 = Conv3D(32, (2, 2, 3), padding='same')(db21)
    db22 = BatchNormalization()(db22)
    db22 = Activation('relu')(db22)

    db23 = Conv3D(32, (2, 2, 3), padding='same')(db22)
    db23 = BatchNormalization()(db23)
    db23 = Activation('relu')(db23)

    db24 = Conv3D(32, (2, 2, 3), padding='same')(db23)
    db24 = BatchNormalization()(db24)
    db24 = Activation('relu')(db24)

    db2_out = Add()([block2, db24])

    # transition bolock 2 : inputsize: 4,4,99,64
    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db2_out)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # dense block 3: inputsize: 4,4,99,64
    db31 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db31 = BatchNormalization()(db31)
    db31 = Activation('relu')(db31)

    db32 = Conv3D(64, (2, 2, 3), padding='same')(db31)
    db32 = BatchNormalization()(db32)
    db32 = Activation('relu')(db32)

    db33 = Conv3D(64, (2, 2, 3), padding='same')(db32)
    db33 = BatchNormalization()(db33)
    db33 = Activation('relu')(db33)

    db34 = Conv3D(64, (2, 2, 3), padding='same')(db33)
    db34 = BatchNormalization()(db34)
    db34 = Activation('relu')(db34)

    db3_out = Add()([block3, db34])

    # flatten and classification
    flattened = Flatten()(db3_out)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_4Conv_6line_4transition(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db11 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db11 = BatchNormalization()(db11)
    db11 = Activation('relu')(db11)

    db12 = Conv3D(16, (2, 2, 3), padding='same')(db11)
    db12 = BatchNormalization()(db12)
    db12 = Activation('relu')(db12)
    db12 = Concatenate(axis=-1)([db11, db12])

    db13 = Conv3D(16, (2, 2, 3), padding='same')(db12)
    db13 = BatchNormalization()(db13)
    db13 = Activation('relu')(db13)
    db13 = Concatenate(axis=-1)([db12, db13])

    db14 = Conv3D(16, (2, 2, 3), padding='same')(db13)
    db14 = BatchNormalization()(db14)
    db14 = Activation('relu')(db14)
    db14 = Concatenate(axis=-1)([db13, db14])

    # transition bolock 1 : inputsize: 4,4,99,64
    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db14)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # dense block 2: inputsize: 4,4,99,64
    db21 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db21 = BatchNormalization()(db21)
    db21 = Activation('relu')(db21)

    db22 = Conv3D(32, (2, 2, 3), padding='same')(db21)
    db22 = BatchNormalization()(db22)
    db22 = Activation('relu')(db22)
    db22 = Concatenate(axis=-1)([db21, db22])

    db23 = Conv3D(32, (2, 2, 3), padding='same')(db22)
    db23 = BatchNormalization()(db23)
    db23 = Activation('relu')(db23)
    db23 = Concatenate(axis=-1)([db22, db23])

    db24 = Conv3D(32, (2, 2, 3), padding='same')(db23)
    db24 = BatchNormalization()(db24)
    db24 = Activation('relu')(db24)
    db24 = Concatenate(axis=-1)([db23, db24])

    # transition bolock 2 : inputsize: 4,4,99,64
    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db24)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # dense block 3: inputsize: 4,4,99,64
    db31 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db31 = BatchNormalization()(db31)
    db31 = Activation('relu')(db31)

    db32 = Conv3D(64, (2, 2, 3), padding='same')(db31)
    db32 = BatchNormalization()(db32)
    db32 = Activation('relu')(db32)
    db32 = Concatenate(axis=-1)([db31, db32])

    db33 = Conv3D(64, (2, 2, 3), padding='same')(db32)
    db33 = BatchNormalization()(db33)
    db33 = Activation('relu')(db33)
    db33 = Concatenate(axis=-1)([db32, db33])

    db34 = Conv3D(64, (2, 2, 3), padding='same')(db33)
    db34 = BatchNormalization()(db34)
    db34 = Activation('relu')(db34)
    db34 = Concatenate(axis=-1)([db33, db34])

    print('db34.shape',db34.shape)

    # transition bolock 3 : inputsize: 1,1,24,256
    block4 = Conv3D(128, (1, 1, 2), strides=(1, 1, 2))(db34)
    block4 = BatchNormalization()(block4)
    block4 = Activation('relu')(block4)

    # dense block 3: inputsize: 4,4,99,64
    db41 = Conv3D(128, (1, 1, 2), padding='same')(block4)
    db41 = BatchNormalization()(db41)
    db41 = Activation('relu')(db41)

    db42 = Conv3D(128, (1, 1, 2), padding='same')(db41)
    db42 = BatchNormalization()(db42)
    db42 = Activation('relu')(db42)
    db42 = Concatenate(axis=-1)([db41, db42])

    db43 = Conv3D(128, (1, 1, 2), padding='same')(db42)
    db43 = BatchNormalization()(db43)
    db43 = Activation('relu')(db43)
    db43 = Concatenate(axis=-1)([db42, db43])

    db44 = Conv3D(128, (1, 1, 2), padding='same')(db43)
    db44 = BatchNormalization()(db44)
    db44 = Activation('relu')(db44)
    db44 = Concatenate(axis=-1)([db43, db44])

    # flatten and classification
    flattened = Flatten()(db44)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_4Conv_6line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db11 = Conv3D(16, (2, 2, 3), padding='same')(block1)
    db11 = BatchNormalization()(db11)
    db11 = Activation('relu')(db11)

    db12 = Conv3D(16, (2, 2, 3), padding='same')(db11)
    db12 = BatchNormalization()(db12)
    db12 = Activation('relu')(db12)
    db12 = Concatenate(axis=-1)([db11, db12])

    db13 = Conv3D(16, (2, 2, 3), padding='same')(db12)
    db13 = BatchNormalization()(db13)
    db13 = Activation('relu')(db13)
    db13 = Concatenate(axis=-1)([db12, db13])

    db14 = Conv3D(16, (2, 2, 3), padding='same')(db13)
    db14 = BatchNormalization()(db14)
    db14 = Activation('relu')(db14)
    db14 = Concatenate(axis=-1)([db13, db14])

    # transition bolock 1 : inputsize: 4,4,99,64
    block2 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db14)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # dense block 2: inputsize: 4,4,99,64
    db21 = Conv3D(32, (2, 2, 3), padding='same')(block2)
    db21 = BatchNormalization()(db21)
    db21 = Activation('relu')(db21)

    db22 = Conv3D(32, (2, 2, 3), padding='same')(db21)
    db22 = BatchNormalization()(db22)
    db22 = Activation('relu')(db22)
    db22 = Concatenate(axis=-1)([db21, db22])

    db23 = Conv3D(32, (2, 2, 3), padding='same')(db22)
    db23 = BatchNormalization()(db23)
    db23 = Activation('relu')(db23)
    db23 = Concatenate(axis=-1)([db22, db23])

    db24 = Conv3D(32, (2, 2, 3), padding='same')(db23)
    db24 = BatchNormalization()(db24)
    db24 = Activation('relu')(db24)
    db24 = Concatenate(axis=-1)([db23, db24])

    # transition bolock 2 : inputsize: 4,4,99,64
    block3 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db24)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # dense block 3: inputsize: 4,4,99,64
    db31 = Conv3D(64, (2, 2, 3), padding='same')(block3)
    db31 = BatchNormalization()(db31)
    db31 = Activation('relu')(db31)

    db32 = Conv3D(64, (2, 2, 3), padding='same')(db31)
    db32 = BatchNormalization()(db32)
    db32 = Activation('relu')(db32)
    db32 = Concatenate(axis=-1)([db31, db32])

    db33 = Conv3D(64, (2, 2, 3), padding='same')(db32)
    db33 = BatchNormalization()(db33)
    db33 = Activation('relu')(db33)
    db33 = Concatenate(axis=-1)([db32, db33])

    db34 = Conv3D(64, (2, 2, 3), padding='same')(db33)
    db34 = BatchNormalization()(db34)
    db34 = Activation('relu')(db34)
    db34 = Concatenate(axis=-1)([db33, db34])

    # flatten and classification
    flattened = Flatten()(db34)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model


def Time_Mid_3D_densenet_5Conv_0line(model_input, chan_num):
    block1 = Conv3D(16, (3, 3, 5), strides=(2, 2, 4))(model_input)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # dense block 1 (db1) : inputsize: 4,4,99,16
    db1 = Conv3D(32, (2, 2, 3), padding='same')(block1)
    db1 = BatchNormalization()(db1)
    db1 = Activation('relu')(db1)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db1)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    db2 = Conv3D(32, (2, 2, 3), padding='same')(db2)
    db2 = BatchNormalization()(db2)
    db2 = Activation('relu')(db2)

    # transition bolock 1 : inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), strides=(2, 2, 2))(db2)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    # dense block 2: inputsize: 4,4,99,64
    db3 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db3 = BatchNormalization()(db3)
    db3 = Activation('relu')(db3)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db3)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    db4 = Conv3D(32, (2, 2, 3), padding='same')(db4)
    db4 = BatchNormalization()(db4)
    db4 = Activation('relu')(db4)

    # transition bolock 2 : inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), strides=(2, 2, 2))(db4)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    # dense block 3: inputsize: 4,4,99,64
    db5 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db5 = BatchNormalization()(db5)
    db5 = Activation('relu')(db5)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db5)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    db6 = Conv3D(64, (2, 2, 3), padding='same')(db6)
    db6 = BatchNormalization()(db6)
    db6 = Activation('relu')(db6)

    # flatten and classification
    flattened = Flatten()(db6)
    dense1 = Dense(32)(flattened)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)

    dense2 = Dense(32)(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)

    out_put = Dense(2, activation='softmax')(dense2)

    model = Model(model_input, out_put)
    return model
