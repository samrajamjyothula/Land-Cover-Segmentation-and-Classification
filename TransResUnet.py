from keras.layers import *
from keras.optimizers import Adam
import random as rn
import sys
import warnings
import matplotlib
import numpy as np
matplotlib.use('agg')
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2 as cv





def Unet(sol, pretrained_weights=None, input_size=(128, 128, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(sol, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(sol, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def TransResUnet(train_data, train_target, test_data=None, sol=None):
    if test_data is None :
        test_data = train_data
    if sol is None :
        sol = [5, 5, 300]
    # Set some parameters
    IMG_SIZE = 256

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    rn.seed = seed
    np.random.seed = seed

    X_train = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    Y_train = np.zeros((train_target.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    X_test = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    # Y_test = np.zeros((test_target.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_train[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))

    for i in range(train_target.shape[0]):
        Temp = cv.resize(train_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                if Temp[j, k] < 0.5:
                    Temp[j, k] = 0
                else:
                    Temp[j, k] = 1
        Y_train[i, :, :, :] = Temp

    for i in range(test_data.shape[0]):
        Temp = cv.resize(test_data[i, :], (IMG_SIZE, IMG_SIZE))
        X_test[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))

    '''for i in range(test_target.shape[0]):
        Temp = cv.resize(test_target[i, :], (IMG_SIZE, IMG_SIZE))
        Temp = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))
        for j in range(Temp.shape[0]):
            for k in range(Temp.shape[1]):
                if Temp[j, k] < 0.5:
                    Temp[j, k] = 0
                elif Temp[j, k] >= 0.5:
                    Temp[j, k] = 1
        Y_test[i, :, :, :] = Temp'''
    sys.stdout.flush()

    # Fit model

    model = Unet(sol[0])
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=sol[1],steps_per_epoch=sol[2],
                        callbacks=[earlystopper, checkpointer])
    pred_img = model.predict(X_test)
    ret_img = pred_img[:, :, :, 0]

    return ret_img


def Test_TransResUnet(data):
    # Set some parameters
    IMG_SIZE = 256

    warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
    seed = 42
    rn.seed = seed
    np.random.seed = seed

    X_test = np.zeros((data.shape[0], IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i in range(data.shape[0]):
        Temp = cv.resize(data[i, :], (IMG_SIZE, IMG_SIZE))
        X_test[i, :, :, :] = Temp.reshape((IMG_SIZE, IMG_SIZE, 1))

    sys.stdout.flush()

    model = tf.keras.models.load_model('model_1.h5')
    pred_img = model.predict(X_test)
    ret_img = pred_img[:, :, :, 0]
    return ret_img






class automaticmaplabelling():
    def __init__(self, modelPath, full_chq, X_test, width, height, channels, model):
        self.modelPath = modelPath
        self.full_chq = full_chq
        self.X_test = X_test
        self.IMG_WIDTH = width
        self.IMG_HEIGHT = height
        self.IMG_CHANNELS = channels
        self.model = model

    def mean_iou(self, y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def prediction(self):
        X_test = self.X_test
        Y_Pred = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        preds_test = self.model.predict(X_test, verbose=1)
        preds_test = (preds_test > 0.5).astype(np.uint8)
        for i in range(preds_test.shape[0]):
            mask = preds_test[i]
            for j in range(mask.shape[0]):
                for k in range(mask.shape[1]):
                    if mask[j][k] >= 1:
                        mask[j][k] = 255
                    else:
                        mask[j][k] = 0
            Y_Pred[i] = mask
        return Y_Pred




