import keras
import math
import numpy as np
import argparse
import os
# from keras.datasets import cifar10
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D, Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers

growth_rate        = 12 
depth              = 100
compression        = 0.5

img_rows, img_cols = 32, 32
img_channels       = 3
# num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 50
iterations         = 782       
weight_decay       = 1e-4

# tensorlog_dir = './graph/image_classification/multi_class'
# train_data_dir = './dataset/cifar-10-python/train_set'

# mean = [125.307, 122.95, 113.865]
# std  = [62.9932, 62.0887, 66.7048]

from keras import backend as K
if('tensorflow' == K.backend()):
    import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    
def createFolder(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        

def arguments():
    parser = argparse.ArgumentParser(description = "binary image classification")
    
    parser.add_argument('--train_path' , 
                        required = True, 
                        help = 'train data directory path')
    parser.add_argument('--resize_x', 
                        required = False,
                        type = int,
                        default = 32,
                       help = 'resizing x')
    parser.add_argument('--epoch',
                       required = False,
                       type= int,
                        default = 50,
                       help = 'epoch size')
    parser.add_argument('--sample_count',
                       required = False,
                        type = int,
                        default = 5000,
                       help = 'tarining sampe count')
    args  = parser.parse_args()
    
    return args

    
def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

def image_generator(train_data_dir, resize_shape ,batch_size):
    train_datagen =ImageDataGenerator(horizontal_flip=True,\
                                      width_shift_range=0.125,
                                      height_shift_range=0.125,
                                      fill_mode='constant',
                                      cval=0.,
                                     rescale = 1./255,
                                     validation_split= 0.2)
    train_generator = train_datagen.flow_from_directory(train_data_dir,\
                                                       target_size = resize_shape,
                                                        batch_size = batch_size,
                                                        class_mode = 'categorical',
                                                        subset = 'training'
                                                       )
    
    validation_generator = train_datagen.flow_from_directory(train_data_dir,\
                                                       target_size = resize_shape,
                                                        batch_size = 1,
                                                        class_mode = 'categorical',
                                                        subset = 'validation'
                                                       )
    return train_generator, validation_generator

def densenet(img_input,classes_num):
    def conv(x, out_filters, k_size):
        return Conv2D(filters=out_filters,
                      kernel_size=k_size,
                      strides=(1,1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      use_bias=False)(x)

    def dense_layer(x):
        return Dense(units=classes_num,
                     activation='softmax',
                     kernel_initializer='he_normal',
                     kernel_regularizer=regularizers.l2(weight_decay))(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = conv(x, channels, (1,1))
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def single(x):
        x = bn_relu(x)
        x = conv(x, growth_rate, (3,3))
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = conv(x, outchannels, (1,1))
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2


    x = conv(img_input, nchannels, (3,3))
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


if __name__ == '__main__':

#     # load data
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_test  = keras.utils.to_categorical(y_test, num_classes)
#     x_train = x_train.astype('float32')
#     x_test  = x_test.astype('float32')
    
#     # - mean / std
#     for i in range(3):
#         x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
#         x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    args = arguments()
    createFolder('./graph')
    createFolder('./weight')
    num_classes = len(os.listdir(args.train_path))
    
    
    img_input = Input(shape=(args.resize_x,args.resize_x,img_channels))
    output    = densenet(img_input,num_classes)
    model     = Model(img_input, output)
    
    # model.load_weights('ckpt.h5')

    print(model.summary())

    # from keras.utils import plot_model
    # plot_model(model, show_shapes=True, to_file='model.png')

    # set optimizer
    # sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # set callback
    tb_cb     = TensorBoard(log_dir = './graph', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint('./graph/ckpt.h5', 
                                save_best_only=True, 
                                mode='auto', period=10)
    cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    
    resize_shape = (args.resize_x, args.resize_x)
    
    train_datagen, validation_datagen = image_generator(args.train_path, resize_shape ,batch_size)
    
    
    # datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

#     datagen.fit(x_train)

    # start training
    model.fit_generator(train_datagen,
                       steps_per_epoch=iterations,
                        epochs=args.epoch, 
                        callbacks=cbks,
                        validation_data=validation_datagen,
                       workers =os.cpu_count(),
                       use_multiprocessing = True,
                        validation_steps=validation_set.samples
                       )
    
    # model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_test, y_test))
    model.save('./weight/model_weight.h5')