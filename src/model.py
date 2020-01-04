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
# from keras_radam import RAdam
import efficientnet.keras as efn
from keras.layers import ReLU,LeakyReLU

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2

# from data import df_gen
from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import efficientnet.keras as efn
from keras.layers import ReLU, LeakyReLU


# def UpSampling2DBilinear(stride, **kwargs):
#     def layer(x):
#         input_shape = K.int_shape(x)
#         output_shape = (stride * input_shape[1], stride * input_shape[2])
#         return tf.image.resize_bilinear(x, output_shape, align_corners=True)
#     return Lambda(layer, **kwargs)


# def get_model(label_counts = 4, input_shape=(256,256,3)):
#     K.clear_session()
#     base_model = efn.EfficientNetB2(weights='imagenet',include_top=False, input_shape= input_shape)
#     base_model.trainable = False
#     base_out = base_model.output
#     conv1 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (base_out) # (8, 16, 16)
#     up = UpSampling2DBilinear(8 )(conv1) # (8, 128, 128)
#     conv2 = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same') (up) # (1, 256, 256)
#     conv3 = Conv2D(label_counts, (1, 1))(conv2)
#     conv4 = Activation('sigmoid')(conv3)
#     model = Model(input=base_model.input, output=conv4)
#     return model


    
    # ACTIVATION = "relu"
def H(lst, name, use_gn=False):
    #     if use_gn:
    #         norm = GroupNormalization(groups=1, name=name+'_gn')
    #     else:
    norm = BatchNormalization(name=name+'_bn')

    x = concatenate(lst)
    num_filters = int(x.shape.as_list()[-1]/2)

    x = Conv2D(num_filters, (2, 2), padding='same', name=name)(x)
    x = norm(x)
    x = LeakyReLU(alpha = 0.1, name=name+'_activation')(x)

    return x

def U(x, use_gn=False):
    #     if use_gn:
    #         norm = GroupNormalization(groups=1)
    #     else:
    norm = BatchNormalization()

    num_filters = int(x.shape.as_list()[-1]/2)

    x = Conv2DTranspose(num_filters, (3, 3), strides=(2, 2), padding='same')(x)
    x = norm(x)
    x = LeakyReLU(alpha = 0.1 )(x)

    return x


def get_model(label_counts = 4 , input_shape = (256,256,3)):

    
    base_model =  efn.EfficientNetB4(weights=None, include_top=False, input_shape=input_shape)
    input = base_model.input
    x00 = base_model.input  # (256, 512, 3)
    x10 = base_model.get_layer('stem_activation').output  # (128, 256, 4)
    x20 = base_model.get_layer('block2d_add').output  # (64, 128, 32)
    x30 = base_model.get_layer('block3d_add').output  # (32, 64, 56)
    x40 = base_model.get_layer('block5f_add').output  # (16, 32, 160)
    x50 = base_model.get_layer('block7b_add').output  # (8, 16, 448)
    
    x01 = H([x00, U(x10)], 'X01')
    x11 = H([x10, U(x20)], 'X11')
    x21 = H([x20, U(x30)], 'X21')
    x31 = H([x30, U(x40)], 'X31')
    x41 = H([x40, U(x50)], 'X41')
    
    x02 = H([x00, x01, U(x11)], 'X02')
    x12 = H([x11, U(x21)], 'X12')
    x22 = H([x21, U(x31)], 'X22')
    x32 = H([x31, U(x41)], 'X32')
    
    x03 = H([x00, x01, x02, U(x12)], 'X03')
    x13 = H([x12, U(x22)], 'X13')
    x23 = H([x22, U(x32)], 'X23')
    
    x04 = H([x00, x01, x02, x03, U(x13)], 'X04')
    x14 = H([x13, U(x23)], 'X14')
    
    x05 = H([x00, x01, x02, x03, x04, U(x14)], 'X05')
    
    x_out = Concatenate(name='bridge')([x01, x02, x03, x04, x05])
    x_out = Conv2D(label_counts, (3,3), padding="same", name='final_output', activation="sigmoid")(x_out)
    
    return Model(inputs=input, outputs=x_out)
    
    
    

