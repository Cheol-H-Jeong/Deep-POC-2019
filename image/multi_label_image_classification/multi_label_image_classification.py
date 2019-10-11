import keras
import math
import numpy as np
import pandas as pd
import argparse
import os
# from keras.datasets import cifar10
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Input, add, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from sklearn.metrics import classification_report
np.random.seed(42)

# growth_rate        = 12 
# depth              = 100
# compression        = 0.5

filters = 32
kernel_size = (3,3)
strides = (1,1)

# img_rows, img_cols = 32, 32
img_channels       = 3
# num_classes        = 10
batch_size         = 64         # 64 or 32 or other
epochs             = 50
iterations         = 782       
weight_decay       = 1e-4


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
    parser.add_argument('--label_path' , 
                        required = True, 
                        help = 'label data file path')
    parser.add_argument('--resize_x', 
                        required = False,
                        type = int,
                        default = 100,
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

def image_generator(train_data_dir, label_file_path, resize_shape, batch_size):
    df = pd.read_csv(label_file_path)
    columns = df.columns[1:].tolist()
    indices = np.random.permutation(df.shape[0])
    training_idx, valid_idx = indices[:int(indices.shape[0]*0.8)], indices[int(indices.shape[0]*0.8):]
    

    
    datagen = ImageDataGenerator(horizontal_flip=True,\
                                      width_shift_range=0.125,
                                      height_shift_range=0.125,
                                      fill_mode='constant',
                                      cval=0.,
                                     rescale = 1./255,)
    test_datagen = ImageDataGenerator(rescale = 1./255,)
    
    
    train_generator = datagen.flow_from_dataframe(dataframe = df.loc[training_idx],
                                                 directory = train_data_dir,
                                                 x_col = df.columns[0],
                                                 y_col = columns,
                                                 batch_size  = batch_size,
                                                 seed = 42,
                                                 shuffle = True,
                                                 class_mode = 'other',
                                                 target_size = resize_shape,)
    valid_generator = test_datagen.flow_from_dataframe(dataframe = df.loc[valid_idx],
                                                 directory = train_data_dir,
                                                 x_col = df.columns[0],
                                                 y_col = columns,
                                                 batch_size  = 1,
                                                 seed = 42,
                                                 shuffle = False,
                                                 class_mode = 'other',
                                                 target_size = resize_shape)
    return train_generator, valid_generator
    
def working_model(input_dim, filters , kernel_size , strides, class_num):
    
    inputs = Input(shape = (input_dim, input_dim, 3))
    x = Conv2D(filters = filters, 
               kernel_size =kernel_size, 
               # strides = strides, 
               padding = 'same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters = filters, 
               kernel_size =kernel_size, 
               # strides = strides,
              )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(filters = 64, 
               kernel_size =kernel_size, 
               # strides = strides,
               padding = 'same'
              )(x)
    # print(x.shape)
    x = Activation('relu')(x)
    # print(x.shape)
    x = Conv2D(filters = 64, 
               kernel_size =kernel_size, 
               # strides = strides,
              )(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(class_num, activation ='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

def train(train_set, validation_set, model, epoch, sample_size):
    model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    tb_cb = TensorBoard(log_dir = './graph/', histogram_freq=0, 
                        write_graph=True, write_images=True)
    
    ckpt = ModelCheckpoint('./weight/ckpt.h5', 
                           save_best_only = True, mode = 'auto', period = 10)
    # change_lr = LearningRateScheduler(scheduler)
    cbks = [tb_cb, ckpt]
    
    STEP_SIZE_TRAIN=train_set.n//train_set.batch_size
    STEP_SIZE_VALID=validation_set.n//validation_set.batch_size
    
    model.fit_generator(train_set,
                       # samples_per_epoch = sample_size,
                       epochs  = epoch,
                       steps_per_epoch = STEP_SIZE_TRAIN,
                       validation_data = validation_set,
                        validation_steps=STEP_SIZE_VALID,
                       # verbose = 2,
                       callbacks = cbks,
                       workers = os.cpu_count(),
                       use_multiprocessing = True)
    return model

if __name__ == '__main__':
    
    args = arguments()
    createFolder('./graph')
    createFolder('./weight')
    
    
    
    print('Data argumentation for multilabel classification')
    resize_shape = (args.resize_x, args.resize_x)
    train_datagen, validation_datagen = image_generator(args.train_path,
                                                        args.label_path,
                                                        resize_shape, 
                                                        batch_size)
    class_num= train_datagen.labels.shape[1]
    print(class_num)
    model = working_model(args.resize_x, filters, kernel_size, strides, class_num)
    print(model.summary())
    
    mdoel = train(train_datagen,
                  validation_datagen,
                 model,
                 args.epoch,
                 args.sample_count,)
    model.save('./weight/model_weight.h5')
    
    STEP_SIZE_TRAIN=train_datagen.n//train_datagen.batch_size
    STEP_SIZE_VALID=validation_datagen.n//validation_datagen.batch_size
    
    
    res = model.predict_generator(validation_datagen, verbose = 2
                                  , steps = STEP_SIZE_VALID)
    
    res = (res>0.5).astype(int)

    
