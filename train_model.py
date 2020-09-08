from sklearn.metrics import confusion_matrix
from keras.layers import Input
from keras.callbacks import LearningRateScheduler
import numpy as np
from config import *


def fit_model(model,X_train, Y_train, X_test, Y_test, save_model_dir):
    hist=None

    # def scheduler(epoch):
    #     lr=None
    #     if epoch >20:
    #         lr=0.0005
    #     elif epoch >10:
    #         lr=0.001
    #     else:
    #         lr=0.01
    #     return lr

    # callbacks = [change_lr],

    # change_lr = LearningRateScheduler(scheduler,verbose=1)
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size, epochs=epochs, verbose=2,
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
