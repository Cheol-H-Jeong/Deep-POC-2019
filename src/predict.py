import warnings
warnings.filterwarnings(action='ignore')
import os
import sys
import keras
from utils import rle2mask, img_preprocess
from keras.models import load_model
from model import *
from generator import * 
from utils import *
from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
import sys
import argparse
import time
from data import df_gen
import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import gc
import ast
from PIL import Image

from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)




FLAGS = None
# print('!!!')
# print(FLAGS)
# print(FLAGS.proba_thres)
# label_proba = [float(k) for k in FLAGS.proba_thres]
# print(label_proba)
# size_thres = [int(k) for k in FLAGS.size_thres]
# pad_thres = [int(k) for k in FLAGS.pad_thres]
# convex_hull = [bool(k) for k in FLAGS.convex_hull]



    
# class TestPredictor:

def predict_df(img_path, colors, label_name=['FISH', 'FLOWER','SUGAR', 'GRAVEL']):
    
    img_set = np.array([k for k in os.listdir(img_path) if k.split('.')[-1]==('jpg' or 'png' or 'jpeg')]).reshape(-1,1)
    
    img_size = np.array([str(list(Image.open(os.path.join(img_path, k)).size)) for k in tqdm(img_set .reshape(-1,))]).reshape(-1,1)
    print('img_size loaded')
    
    colors = np.array([str(colors.tolist()) for _ in tqdm(range(len(img_set.reshape(-1,))))]).reshape(-1,1)
    print('colors loaded')
    
    df = pd.DataFrame(img_set, columns=['ImageId'])
    
    lable_nan_arr = np.empty((df.shape[0], len(label_name))) 
    lable_nan_arr[:] = np.nan
    df_label = pd.DataFrame(lable_nan_arr, columns = label_name)
    
    # origin_size_nam_arr = np.empty((df.shape[0], 1))
    # origin_size_nam_arr[:] = np.nan
    df_origin_size = pd.DataFrame(img_size, columns = ['size'])
    
    # colors_nan = np.empty((df.shape[0],1))
    # colors_nan[:] = np.nan
    df_colors = pd.DataFrame(colors, columns = ['colors'])
    
    df = pd.concat([df,df_origin_size,df_colors,df_label],axis=1)
    
    return df


def batch_predict(model, test_df,label_names ,pred_batch_size = 10,proba = [0.5,0.5], pad_size=[3,3], reduce_size = [3000,3000], convex= [True,True] , resize_shape=[512,256]):
    batch_res_df = []
    
    
    
    
    for i in tqdm(range(0, test_df.shape[0], pred_batch_size)):
        batch_idx = list(range(i, min(test_df.shape[0], i + pred_batch_size)))
        print(batch_idx)
        
        # (512,256,4)
        batch_generator  = SegmentDataGenerator( test_df.iloc[batch_idx], subset='test', batch_size = 1,
                                               shuffle=False, preprocess = img_preprocess,
                                               augmentation = None, resize_shape = resize_shape,
                                               test_path = FLAGS.img_path)
        
        # (512,256,4)
        batch_preds = model.predict_generator(batch_generator, verbose = 1)
        print(batch_preds.shape)
        
        # (512,256,4)
        batch_preds_re = predict_resize(batch_preds, 
                                       proba=proba, 
                                       pad_size=pad_size, 
                                       reduce_size = reduce_size, 
                                       convex= convex,
                                        label_names = label_names
                                     )
        print(batch_preds_re.shape)
        np.save('../test.npy',batch_preds_re)
#         sys.exit()
        
        for j, b in enumerate(batch_idx):
            filename = test_df['ImageId'].iloc[b]
            image_df = test_df[test_df['ImageId'] == filename].copy()
            # print(image_df)
            preds_mask = batch_preds_re[j]
            
            # print(origin_size)
            # image_df['size'] = [[origin_size[0], origin_size[1]]]
            # image_df['colors'] = [colors]
            
            # [2100,1400]
            origin_shape = ast.literal_eval(image_df['size'].values[0])
            print(origin_shape)
            # print(image_df)
            # sys.exit()
            for l_idx, label in enumerate(label_names):
                l_mask = preds_mask[:,:,l_idx].astype(np.uint8)
                print(l_mask.shape)
                l_mask = cv2.resize(l_mask, (origin_shape[1], origin_shape[0]))
                label_rle = mask2rle(l_mask)
                # print(label_rle)

                image_df[label] = label_rle['counts']
            batch_res_df.append(image_df)
            gc.collect()
        gc.collect()
    batch_res_df = pd.concat(batch_res_df)
    
    print(batch_res_df)
    # sys.exit()
    batch_res_df.replace(to_replace=[None], value = np.nan, inplace = True)
    print(batch_res_df)
    print(batch_res_df.shape)
    print('Batch predict end, Images : {}, labels : {}'.format(batch_res_df.shape[0], batch_res_df.shape[1]-3))
    
    
    
    return batch_res_df


    




def predict_main():
    # label_proba = [float(k) for k in FLAGS.proba_thres]
    # size_thres = [int(k) for k in FLAGS.size_thres]
    # pad_thres = [int(k) for k in FLAGS.pad_thres]
    # convex_hull = [bool(k) for k in FLAGS.convex_hull]
    
     
    # print(label_proba)
    # sys.exit()
    print(FLAGS.model_path)
    valid_info_path = '/'.join(FLAGS.model_path.split('/')[:-2]) + '/valid/valid_post_grid.csv'
    # print(valid_info_path)
    # sys.exit()
    
    
    info_df = pd.read_csv(valid_info_path)
    colors = info_df['colors'].values
    labels = info_df['label'].values
    probas = info_df['proba'].values
    reduce_sizes = info_df['reduce_size'].values
    
    # (512,256)
    resize_shape = ast.literal_eval( info_df['size'].values[0])
    pads = info_df['pad'].values
    convexes = info_df['convex'].values
    print(colors)
    print(labels)
    print(probas)
    print(reduce_sizes)
    print(pads)
    print(convexes)
    print(resize_shape)
    # sys.exit()
    
    
    # origin_shape = ast.literal_eval(df.iloc[0]['size'])
    
    # colors = ast.literal_eval(df.iloc[0]['colors'])
    # print(origin_shape)
    # print(colors)
    # sys.exit()
    
    
    # print(df.keys())
    # labels = df.keys()[3:].values.tolist()
    print(labels)
    
    
    # sys.exit()
    
    print(FLAGS.model_path)
    # sys.exit()
    K.clear_session()
    
    # In : (40, 512,256,3)
    # Out : (40, 512,256,4)
    loaded_model = load_model(FLAGS.model_path, custom_objects = {'tf' : tf, 
                                                           'bce_dice_loss' : bce_dice_loss,
                                                          'dice_coef' : dice_coef})
    print(loaded_model)
    # sys.exit()
    
    df_test = predict_df(FLAGS.img_path, colors = colors ,label_name=labels)
    print(df_test)
    # sys.exit()
    
    pred_test = batch_predict(loaded_model, df_test, label_names = labels ,pred_batch_size = FLAGS.batch_size, proba = probas, pad_size = pads , reduce_size = reduce_sizes , convex = convexes, resize_shape = resize_shape)
    
    pred_path = '/'.join(FLAGS.model_path.split('/')[:-2]) +  '/pred/'
    
    pred_test.to_csv(os.path.join(pred_path, 'batch_pred.csv'), index=False)
    print(pred_test)
    print(pred_path)
    # sys.exit()
    
    for r_idx in tqdm(range(pred_test.shape[0])):
        
        
        img = rle_mask2img(pred_test.iloc[r_idx], FLAGS.img_path)
        
        # print( pred_test.iloc[r_idx]['ImageId'].split('.')[0])
        # print(pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        # print('!!!!')
        print(pred_test.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        save_path = os.path.join(pred_path, pred_test.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        img.save(save_path)
        print ('Segment {} done !!! '.format(pred_test.iloc[r_idx]['ImageId']))
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'testing')
    parser.add_argument('--method', required=True, help = 'batch  or realtime')
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    # parser.add_argument('--json_path', required=True)
    parser.add_argument('--img_path', required=True)
    parser.add_argument('--model_path',required = True)
    # parser.add_argument('--proba_thres', nargs='+', type = float, help = 'Post - label proba', default = [0.5])
    # parser.add_argument('--size_thres', nargs='+', type = int, help = 'Post - label size', default = [3000])
    # parser.add_argument('--pad_thres', nargs='+', type = int, help = 'Post - label pad size', default = [3])
    # parser.add_argument('--convex_hull',  nargs='+',type = bool,  help = 'Post - label convex hull', default = [True])
    # parser.add_argument('--epoch', default = 50, type =int)
    # parser.add_argument('--batch_size', default = 16, type =int)
    
    # parser.add_argument('--resize_shape', default = (256,256))
    # parser.add_argument('--origin_shape', default = (1200,1200))
    parser.add_argument('--resize_shape', nargs='+', type=int, default = [256,256])
    # parser.add_argument('--origin_shape', nargs='+', type=int, default = [1200,1400])
    parser.add_argument('--batch_size', default = 10, type=int)
    # parser.add_argument('--augument',default = True)
    
    
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    # print(FLAGS.proba_thres)
    
    
    
    predict_main()
        
        
















# if __name__ == '__main__':
    
#     parser = argparse.ArgumentParser(description = 'predict')
#     # parser.add_argument('--target', required=True, help = 'train or predict')
#     # parser.add_argument('--train_path')
#     # parser.add_argument('--test_path')
#     parser.add_argument('--json_path', required=True)
#     parser.add_argument('--img_path', required=True)
#     parser.add_argument('--model_path',required=True)
#     # parser.add_argument('--epoch', default = 50, type =int)
#     # parser.add_argument('--batch_size', default = 16, type =int)
    
#     parser.add_argument('--resize_shape', default = (256,256))
#     # parser.add_argument('--augument',default = True)
    
    
#     FLAGS, unparsed = parser.parse_known_args()
    
    
#     main()