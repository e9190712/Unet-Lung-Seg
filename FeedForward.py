from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
import matplotlib.pyplot as plt


working_path = "E:/CT_OUTPUT/FOR_LUNACT/"
K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)

    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train_and_predict(use_existing):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    imgs_train = np.load(working_path + "trainImages_LUNAseg.npy").astype(np.float32)
    imgs_mask_train = np.load(working_path + "trainMasks_LUNAseg.npy").astype(np.float32)

    imgs_test = np.load(working_path + "testImages_LUNAseg.npy").astype(np.float32)
    imgs_mask_test_true = np.load(working_path + "testMasks_LUNAseg.npy").astype(np.float32)
    print(len(imgs_mask_train), len(imgs_mask_test_true))
    fig, ax = plt.subplots(2, 2, figsize=[12, 12])
    ax[0, 0].imshow(imgs_train[0,0,:,:])
    ax[0, 1].imshow(imgs_mask_train[0,0,:,:])
    ax[1, 0].imshow(imgs_test[0,0,:,:])
    ax[1, 1].imshow(imgs_mask_test_true[0,0,:,:])


    plt.show()

    '''
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean  # images should already be standardized, but just in case
    imgs_train /= std
    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test -= mean  # images should already be standardized, but just in case
    imgs_test /= std
    print(imgs_train.shape)
    '''


    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_unet()

    tb_cb = TensorBoard(log_dir='./logs_LUNA_relu', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None)

    print('-' * 30)
    print('Fitting model...')
    print('-' * 30)
    model.fit(x=imgs_train, y=imgs_mask_train, batch_size=2, nb_epoch=100, verbose=1,
              shuffle=True
              , callbacks=[tb_cb])


    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.save('unet_LUNAseg_relu.hdf5')
    model.load_weights('unet_LUNAseg_relu.hdf5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    num_test = len(imgs_test)
    imgs_mask_test = np.ndarray([num_test, 1, 512, 512], dtype=np.float32)
    for i in range(num_test):
        imgs_mask_test[i] = model.predict([imgs_test[i:i + 1]], verbose=0)[0]
    np.save(working_path+'masksTestPredicted_LUNAseg_relu.npy', imgs_mask_test)
    mean = 0.0
    for i in range(num_test):
        mean += dice_coef_np(imgs_mask_test_true[i, 0], imgs_mask_test[i, 0])
    print(mean)
    mean /= num_test
    print("Mean Dice Coeff : ", mean)
    for i in range(num_test):

        fig, ax = plt.subplots(2, 2, figsize=[12, 12])
        ax[0, 0].imshow(imgs_mask_test[i,0], cmap='gray')
        ax[0, 1].imshow(imgs_test[i,0], cmap='gray')

        plt.show()

        print(mean)

    plt.show()
if __name__ == '__main__':
    train_and_predict(False)
