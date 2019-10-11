import numpy as np
import pandas as pd
import os
import argparse
from keras.models import Model
from keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report


# input_shape = 64
filters = 32
kernel_size = 3
strides = (3,3)
iterations = 782
# resize_shape = (64,64)

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



def working_model(input_dim, filters , kernel_size , strides):
    inputs = Input(shape = (input_dim, input_dim, 3))
    cnn = Conv2D(filters = filters, kernel_size=kernel_size, strides=strides, activation = 'relu')(inputs)
    cnn = MaxPooling2D(pool_size=(2,2))(cnn)
    cnn = Flatten()(cnn)
    cnn = Dense(128, activation='relu')(cnn)
    output = Dense(1, activation='sigmoid')(cnn)
    classifier = Model(inputs,output)
    
    return classifier


def image_generator(train_path,  resize_shape):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range = 0.2,
                                      zoom_range = 0.2,
                                      horizontal_flip = True,
                                      validation_split = 0.2)
    # test_datagen = ImageDataGenerator(rescale= 1./255)
    training_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = resize_shape,
                                                    batch_size = 32,
                                                    class_mode = 'binary',
                                                    subset = 'training')
    validation_set = train_datagen.flow_from_directory(train_path,
                                                    target_size = resize_shape,
                                                    batch_size = 1,
                                                    class_mode = 'binary',
                                                     subset = 'validation')
    
    # test_set = test_datagen.flow_from_directory(test_path,
    #                                                 target_size = resize_shape,
    #                                                 batch_size = 32,
    #                                                 class_mode = 'binary')
    return training_set, validation_set


def train(train_set,validation_set ,model, epoch, sample_size):
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
#     tb_hist = TensorBoard(log_dir='./graph/image_classification/binary', 
#                           histogram_freq=0, write_graph=True, write_images=True)
    
    tb_cb = TensorBoard(log_dir = './graph/', histogram_freq=0, 
                        write_graph=True, write_images=True)
    
    ckpt = ModelCheckpoint('./weight/ckpt.h5', 
                           save_best_only = True, mode = 'auto', period = 10)
    cbks = [tb_cb, ckpt]
    
    
    model.fit_generator(train_set,
#                         steps_per_epoch = 15,
#                         epochs = 5,
                       samples_per_epoch = sample_size,
                       epochs = epoch,
                        # steps_per_epoch = iterations,
                        steps_per_epoch = train_set.samples// 32,
                       validation_data = validation_set,
                        # validation_steps=1,
                        validation_steps=validation_set.samples,
                       # nb_val_samples = 2000,
                       verbose = 2,
                       callbacks = cbks,
                       workers = os.cpu_count(),
                       use_multiprocessing = True)
    return model

def score(validation_set, model):
    predict_res = model.predict_generator(validation_set,
                                         verbose = 2,
                                         workers = os.cpu_count(),
                                         use_multiprocessing = True,
                                         steps = validation_set.samples)
    print(validation_set.class_indices)
    def label(x):
        if x>0.5:
            return 1
        else:
            return 0
    label_encode = np.vectorize(label)
    res = label_encode(predict_res)
    # print(np.unique(res))
    # print(res.shape)
    # print(confusion_matrix(validation_set.classes,res))
    print(classification_report(validation_set.classes, res))
    

if __name__ == '__main__':
    
    args = arguments()
    createFolder('./graph')
    createFolder('./weight')
    
    
    
    
    
    model = working_model(args.resize_x, filters, kernel_size, strides)
    print(model.summary())
    
    print('Data argumentation for binary classification')
    resize_shape = (args.resize_x, args.resize_x)
    train_datagen, validation_datagen = image_generator(args.train_path, resize_shape)
    model = train(train_datagen , 
                  validation_datagen, 
                  model, 
                  args.epoch, 
                  args.sample_count)
    model.save('./weight/model_weight.h5')
    
    # predict_res = model.predict_generator()
    
    score(validation_datagen, model)
    