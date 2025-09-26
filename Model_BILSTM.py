from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import numpy as np
import cv2 as cv
from Evaluation_nrml import evaluation


def LSTM_Bi_train(train_data, train_target, test_data, sol):
    # define problem properties
    n_timesteps = 10
    # define LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(sol[0], return_sequences=True), input_shape=(train_target.shape[1], 1)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # # train BI-LSTM
    #     # fit model for one epoch on this sequence
    for i in range(train_data.shape[0]):
        d=cv.resize(train_data[i],[1, train_target.shape[1]]).ravel()
        data = d.reshape((1, d.shape[0], 1))
        tar = train_target[i].reshape((1, train_target.shape[1], 1))
        model.fit(data, tar, epochs=1, batch_size=1, verbose=2)
    # # evaluate BI-LSTM
    predict = np.zeros((test_data.shape[0], train_target.shape[1]))#.astype('int')
    for i in range(test_data.shape[0]):
        d = cv.resize(test_data[i], [1, train_target.shape[1]]).ravel()
        data = d.reshape((1, d.shape[0], 1))
        predict[i] = model.predict(data, verbose=0).ravel()
    return predict, model

def Model_BILSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 5]
    out, model = LSTM_Bi_train(train_data, train_target, test_data, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred.astype('int'), test_target)
    return np.asarray(Eval).ravel()

