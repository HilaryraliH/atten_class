from sklearn.metrics import confusion_matrix
from keras.layers import Input
from keras.callbacks import LearningRateScheduler
import numpy as np
from config import *
import matplotlib.pyplot as plt


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]

        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")
        plt.savefig('lr_changes.png')
        plt.close()


class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.001, factor=0.4,dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)


def fit_model(model, X_train, Y_train, X_test, Y_test, save_model_dir):
    schedule = StepDecay(initAlpha=0.001,factor=0.4,dropEvery=10) # 构造阶梯型学习率衰减
    schedule.plot([i for i in range(30)])
    call_backs = [LearningRateScheduler(schedule)]
    hist = model.fit(X_train, Y_train,
                     batch_size=batch_size, epochs=epochs, verbose=2, callbacks=call_backs,
                     validation_data=(X_test, Y_test))
    model.save(save_model_dir)
    return hist


def evaluate_model(model, X_test, Y_test):
    probs = model.predict(X_test)
    preds = probs.argmax(axis=-1)
    true_label = Y_test.argmax(axis=-1)
    acc = np.mean(preds == true_label)
    confu_mat = confusion_matrix(true_label, preds, labels=[0, 1])
    return acc, confu_mat


#另一种改变学习率的方法
# def scheduler(epoch):
#     # 以前的变化：start：0.01  >10epoch：0.001 >20epoch：0.0005
#     # 现在：每10个epoch降低0.25倍的学习率
#     init_lr = 0.001
#     factor = 0.4
#     dropEvery = 10
#     exp = np.floor((1 + epoch) / dropEvery)
#     lr = init_lr * (factor ** exp)
#     return lr
#
# change_lr = LearningRateScheduler(scheduler, verbose=1) # LearningRateScheduler接收一个函数作为一个封装
# model.compile(....callbacks=[change_lr])