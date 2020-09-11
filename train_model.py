from sklearn.metrics import confusion_matrix
from keras.layers import Input
from keras.callbacks import LearningRateScheduler
import numpy as np
from config import *


def fit_model(model,X_train, Y_train, X_test, Y_test, save_model_dir):
    hist=None

    # def scheduler(epoch):
    #     # 以前的变化：start：0.01  >10epoch：0.001 >20epoch：0.0005
    #     # 现在：每10个epoch降低0.25倍的学习率
    #     init_lr = 0.001
    #     factor = 0.25
    #     dropEvery = 10
    #     exp = np.floor((1 + epoch) / dropEvery)
    #     lr = init_lr * (factor ** exp)
    #     return lr

    #change_lr = LearningRateScheduler(scheduler,verbose=1)
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs, verbose=2,# callbacks = [change_lr],
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
