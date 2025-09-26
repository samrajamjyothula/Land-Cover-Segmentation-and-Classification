from keras.models import *
from keras.layers import *
import numpy as np
from keras import optimizers

def FCN8(sol,nClasses, input_height=128, input_width=128):
    VGG_Weights_path = "./Keypoint/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ## input_height and width must be devisible by 32 because maxpooling with filter size = (2,2) is operated 5 times,
    ## which makes the input_height and width 2^5 = 32 times smaller
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    IMAGE_ORDERING = "channels_last"

    img_input = Input(shape=(input_height, input_width, 3))  ## Assume 224,224,3

    ## Block 1
    x = Conv2D(sol[0], (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING)(x)
    pool3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 14, 14, 512)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(pool4)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(
        x)  ## (None, 7, 7, 512)


    n = 4096
    o = (Conv2D(n, (7, 7), activation='relu', padding='same', name="conv6", data_format=IMAGE_ORDERING))(pool5)
    conv7 = (Conv2D(n, (1, 1), activation='relu', padding='same', name="conv7", data_format=IMAGE_ORDERING))(o)

    ## 4 times upsamping for pool4 layer
    conv7_4 = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format=IMAGE_ORDERING)(
        conv7)
    ## (None, 224, 224, 10)
    ## 2 times upsampling for pool411
    pool411 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool4_11", data_format=IMAGE_ORDERING))(pool4)
    pool411_2 = (
        Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format=IMAGE_ORDERING))(
        pool411)


    pool311 = (
        Conv2D(nClasses, (1, 1), activation='relu', padding='same', name="pool3_11", data_format=IMAGE_ORDERING))(pool3)

    o = Add(name="add")([pool411_2, pool311, conv7_4])
    o = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format=IMAGE_ORDERING)(o)
    o = (Activation('softmax'))(o)

    model = Model(img_input, o)

    return model


def FCN(X,Y,Z,sol):
    an = 1
    if an == 1:
        model = FCN8(sol=sol,nClasses=3, input_height=128, input_width=128)
        model.summary()
        from sklearn.utils import shuffle
        train_rate = 0.85
        index_train = np.random.choice(X.shape[0], int(X.shape[0] * train_rate), replace=False)
        index_test = list(set(range(X.shape[0])) - set(index_train))
        X, Y = shuffle(X, Y)
        X_train, y_train = X[index_train], Y[index_train]
        X_test, y_test = X[index_test], Y[index_test]
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        sgd = optimizers.SGD(lr=1E-2, decay=5 ** (-4), momentum=0.04, nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=32, epochs=sol[1], verbose=2)
        model.save('model_2.h5')
    else:
        model = load_model('model_2.h5')
        pred = model.predict(Z)
        Ab_Seg = np.zeros((pred.shape)).astype('uint8')
        uni, count = np.unique(pred, return_counts=True)
        if len(count) == 2:
            ind = np.where(count == np.sort(count)[0])
            index = np.where(pred == uni[ind[0][0]])
            Ab_Seg[index[0], index[1]] = 255
        elif len(count) == 1 or (count == []):
            Ab_Seg = Ab_Seg
        else:
            ind = np.where(count == np.sort(count)[1])  # np.min(count))
            index = np.where(pred == uni[ind[0][0]])
            Ab_Seg[index[0], index[1]] = 255
        return Ab_Seg

