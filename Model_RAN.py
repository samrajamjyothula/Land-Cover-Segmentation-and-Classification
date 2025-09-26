import warnings
from Evaluation_nrml import evaluation
warnings.filterwarnings("ignore")
from ResidualAttentionNetwork import ResidualAttentionNetwork
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras import optimizers
import numpy as np
import cv2 as cv

def Model_RAN(train_data,train_target,test_data,test_target):
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32
    IMAGE_CHANNELS = 3
    IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    X_train = np.zeros((train_target.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)
    X_test = np.zeros((test_target.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)
    for i in range(train_data.shape[0]):
        Temp = cv.resize(train_data[i, :], (IMAGE_WIDTH, IMAGE_HEIGHT))
        Temp1 = cv.resize(test_data[i, :], (IMAGE_WIDTH, IMAGE_HEIGHT))
        X_train[i, :, :, :] = Temp.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
        X_test[i, :, :, :] = Temp1.reshape((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    model_path = "/pylon5/cc5614p/deopha32/Saved_Models/cvd-model.h5"
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)
    csv_logger = CSVLogger("/pylon5/cc5614p/deopha32/Saved_Models/cvd-model-history.csv", append=True)
    callbacks = [checkpoint, csv_logger]
    # Model Training
    with tf.device('/gpu:0'):
        model = ResidualAttentionNetwork(
            input_shape=IMAGE_SHAPE,
            n_classes=2,
            activation='softmax').build_model()

        model.compile(optimizer=optimizers.RMSprop(lr=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, steps_per_epoch=100, verbose=0, callbacks=callbacks,epochs=2, use_multiprocessing=True, workers=40)
        predcit  = model.predict(test_data)
        Eval = evaluation(predcit, test_target)
        return Eval




