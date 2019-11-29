import keras
from keras.models import Model, load_model
from keras.layers import Concatenate,Dense,Input,Dropout,BatchNormalization,Activation,Add,Lambda, InputLayer, UpSampling2D,ZeroPadding2D

from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard,EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras_radam import RAdam

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

# from data import df_gen
from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import efficientnet.keras as efn
from keras.layers import ReLU, LeakyReLU


def UpSampling2DBilinear(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        return tf.image.resize_bilinear(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)


def get_model(label_counts = 4, input_shape=(256,256,3)):
    K.clear_session()
    base_model = efn.EfficientNetB2(weights='imagenet',include_top=False, input_shape= input_shape)
    base_model.trainable = False
    base_out = base_model.output
    conv1 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (base_out) # (8, 16, 16)
    up = UpSampling2DBilinear(8 )(conv1) # (8, 128, 128)
    conv2 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same') (up) # (1, 256, 256)
    conv3 = Conv2D(label_counts, (1, 1))(conv2)
    conv4 = Activation('sigmoid')(conv3)
    model = Model(input=base_model.input, output=conv4)
    return model





