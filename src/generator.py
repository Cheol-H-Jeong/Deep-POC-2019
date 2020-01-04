import warnings
warnings.filterwarnings(action='ignore')
import os
import sys
import keras
from utils import rle2mask, img_preprocess
from model import *
# from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import numpy as np
import tensorflow as tf
import sys
import argparse
import time
from data import df_gen
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import ast

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)
import ast


class SegmentDataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size = 16 ,subset ='train', shuffle = False, preprocess = None, info={}, augmentation = None, resize_shape = (256,256), train_path=None, test_path=None, ):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.augmentation = augmentation
        self.resize_shape = resize_shape
        
        if self.subset =='train':
            # self.data_path = path +'train_images/'
            self.data_path = train_path + '/'
        elif self.subset =='test':
            # self.data_path = path + 'test_images/'
            self.data_path = test_path + '/'
        self.on_epoch_end()
        
    def __len__(self):
        
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __random_transform__(self, img, masks):
        composed = self.augmentation(image = img, mask = masks)
        aug_img = composed['image']
        aug_masks = composed['mask']
        
        return aug_img, aug_masks
    def __getitem__(self,index):
        x = np.empty((self.batch_size, self.resize_shape[0], self.resize_shape[1], 3), dtype=np.float32)
        y = np.empty((self.batch_size, self.resize_shape[0], self.resize_shape[1], len(self.df.columns[3:].to_list())), dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size + i] =f 
#             x[i,]=Image.open(self.data_path + f).resize((256,256), resample = Image.)
            
            # print (self.data_path + f)
            img_value=cv2.imread(self.data_path + f)
            # color BGR => RGB
            img_value = cv2.cvtColor(img_value, cv2.COLOR_BGR2RGB)
            x[i,] = cv2.resize(img_value,\
                               (self.resize_shape[1], self.resize_shape[0]),\
                               interpolation=cv2.INTER_AREA)
            
            
            
            
#             x[i,] = cv2.imread(self.data_path + f)(img_value, (256,256),interpolation=cv2.INTER_AREA)
            if self.subset =='train':
                # print(self.df.columns[2:].to_list())
                for j,label in enumerate(self.df.columns[3:].to_list()):
                    try:
                        rle = ast.literal_eval(self.df[label].iloc[indexes[i]])
                    except:
                        rle = self.df[label].iloc[indexes[i]]
#                     print(rle)
                    try:
                        shape = ast.literal_eval(self.df['size'].iloc[indexes[i]])
                    except:
                        shape = self.df['size'].iloc[indexes[i]]
                    # print(shape)
                    # print(rle)
                    y[i,:,:,j] = rle2mask(rle, shape, resize_shape = (self.resize_shape[1], self.resize_shape[0]) )
                    
            if not self.augmentation is None:
                x[i,], y[i,] = self.__random_transform__(x[i,], y[i,])
        if self.preprocess !=None : x= self.preprocess(x)
        if self.subset == 'train' : return x,y
        else: return x

        
def augumentation():
    augument = albu.Compose([albu.VerticalFlip(), 
                         albu.HorizontalFlip(), 
                         albu.Rotate(limit= 20),
                         albu.GridDistortion(),
#                     albu.RandomSizedCrop((128,174), 256, 384,interpolation =cv2.INTER_LINEAR )
                        ],p=1.0)
    return augument

