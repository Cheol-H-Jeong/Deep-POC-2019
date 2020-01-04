import pandas as pd
import numpy as np
from utils import *
from metric import *
from multiprocessing import Pool
import cv2
from tqdm import tqdm
from functools import partial
from generator import *
from model import *
from keras.models import load_model
import tensorflow as tf
import ast
from sklearn.model_selection import ParameterGrid
import gc
import os
# from predict import *


param_grid = {'proba' : np.round(np.array([0.25,0.5, 0.75],dtype =np.float64),2),
                   'pad' : np.array([0,5,10]),
                   'reduce_size' : np.array([0,5000,10000,20000,40000],dtype=np.int64),
              'convex': [True,False]
              
 }
# param_grid = {'proba' : np.round(np.array([0,0.25,0.50,0.75, 1.0],dtype =np.float64),2),
#                    'pad' : np.array([0,3,5,10]),
#                    'reduce_size' : np.arange(5000,20000,5000,dtype=np.int64),
# #               'convex': [True, False]
              
#  }
cores = os.cpu_count()

parameter = list(ParameterGrid(param_grid))
print (len(parameter))




def predict_post_grid(msks, proba = 0.9, pad = False, pad_size = 10, reduce = False, reduce_size =10000, convex = False, resize_shape=[512,256] ):
#     print(msks.shape)

    resized_msks = np.zeros((msks.shape[0],resize_shape[0], resize_shape[1]))
    for i in range(len(msks)):
#         cls = msks[1][i]
#         msk = msks[0][i,:,:] * cls
        msk = msks[i,:,:].copy()
        msk = np.array(msk >=proba, dtype = np.uint8)
        # msk = cv2.resize(msk,(resize_shape[1],resize_shape[0]), interpolation = cv2.INTER_LINEAR)

        msk = mask2pad(msk, pad_size)
        

        msk = masks_reduce(msk, reduce_size)
        msk = contour_convexHull(msk.astype(np.uint8))
        
        
        resized_msks[i, : , :] = msk
    return resized_msks




def paralleize_numpy(preds, func ,proba = 0.9, pad_size = 10, reduce_size = 10000, convex =False, resize_shape=[512,256], cores=2):
    np_split = np.array_split(preds, cores)
#     np_split_cls = np.array_split(preds_cls, cores)
    
#     value = list(zip(np_split, np_split_cls))
    
    pool = Pool(cores)
    res_np = np.concatenate(pool.map(partial(func, proba = proba,pad_size = pad_size, reduce_size = reduce_size , convex = convex, resize_shape = resize_shape), np_split ))
    pool.close()
    pool.join()
    return res_np





def post_optimi(valid_true, valid_pred, resize_shape=[512,256], labels =['fish', 'flower', 'gravel', 'sugar']):
    val_dice_max = []
    val_dice_max_np = []
    
    
    for l_idx, label in enumerate(labels):
        val_dice_tabel = pd.DataFrame(columns = ['label','size', 'proba', 'reduce_size','pad', 'convex', 'dice'])
        val_dice_max_np_temp = []
        for p_idx,para in tqdm(enumerate(parameter)):
            
            valid_preds_pp = paralleize_numpy(valid_pred[:,:,:,l_idx],
                                          predict_post_grid, 
                                          proba=para['proba'], 
                                          pad_size = para['pad'], 
                                          reduce_size = para['reduce_size'],
                                         cores = cores,
                                         convex=para['convex'],
                                             resize_shape = resize_shape)
            dice_score = dice_channel_label(valid_preds_pp, valid_true[:,:,:, l_idx])

            val_dice_tabel = val_dice_tabel.append({'label' : label,
                                                   'proba' : para['proba'],
                                                    'size' : str(resize_shape),
                                                   'reduce_size' : para['reduce_size'],
                                                    'pad' : para['pad'],
                                                    'convex': para['convex'],
                                                   'dice' : dice_score},
                                                   ignore_index=True)

            val_dice_max_np_temp.append(valid_preds_pp)    
            gc.collect()
        label_max_table = val_dice_tabel[val_dice_tabel['label']==label]
        print(label_max_table)
        print(label_max_table[label_max_table['dice'] == label_max_table['dice'].max()])
        label_max_table = label_max_table[label_max_table['dice'] == label_max_table['dice'].max()].iloc[0].to_frame().T
        print(label_max_table)
        label_max_idx = label_max_table.index[-1]
        print(label_max_idx)
        # sys.exit()
        val_dice_max.append(label_max_table)
        val_dice_max_np.append(np.expand_dims(val_dice_max_np_temp[label_max_idx],-1))
        
        gc.collect()
    val_dice_max = pd.concat(val_dice_max, axis=0)
    val_dice_max_np = np.concatenate(val_dice_max_np, axis =-1)
    print(val_dice_max.shape, val_dice_max_np.shape)
    return val_dice_max, val_dice_max_np


def post_batch(valid_df,post_arr, label_names ):
    batch_res_df = []
#     for i in tqdm(range(0, test_df.shape[0], pred_batch_size)):
#         batch_idx = list(range(i, min(test_df.shape[0], i + pred_batch_size)))
        
#         batch_generator  = SegmentDataGenerator( test_df.iloc[batch_idx], subset='test', batch_size = 1,
#                                                shuffle=False, preprocess = img_preprocess,
#                                                augmentation = None, resize_shape = FLAGS.resize_shape,
#                                                test_path = FLAGS.img_path)
        
        
#         batch_preds = model.predict_generator(batch_generator, verbose = 1)
#     batch_preds_re = predict_resize(valid_arr.copy(), 
#                                    proba=proba, 
#                                    pad_size=pad_size, 
#                                    reduce_size = reduce_size, 
#                                    convex= convex,
#                                     origin_img_size = (FLAGS.origin_shape[0],FLAGS.origin_shape[1])
#                                  )
# #         # np.save('../test.npy',batch_preds_re)
# #         # sys.exit()
        
    for j in tqdm(range(valid_df.shape[0])):
        filename = valid_df['ImageId'].iloc[j]
        image_df = valid_df[valid_df['ImageId'] == filename].copy()
        # print(image_df)
        preds_mask = post_arr[j, : ,:, :]
        
        # size [2100,1400]
#         print(image_df['size'])
#         print(image_df)
        # try:
            # origin_shape = ast.literal_eval(image_df['size'].values)
        # except ValueError:
        
        # origin shape : [2100,1400]
        origin_shape = ast.literal_eval(image_df['size'].values[0])
        

        # print(origin_size)
        
        
        # image_df['size'] = [[origin_size[0], origin_size[1]]]
        # image_df['colors'] = [colors]
        
        
        # print(image_df)
        # sys.exit()
#         print(origin_shape)
        for l_idx, label in enumerate(label_names):
            # (512, 256)
            l_mask = preds_mask[:,:,l_idx].astype(np.uint8)
            # (2100,1400)
            l_mask = cv2.resize(l_mask, (origin_shape[1], origin_shape[0]))
            # In : (2100,1400)
            # Out : rle (2100, 1400), [2100,1400]
            label_rle = mask2rle(l_mask)
#             print(label_rle)

            image_df[label] = label_rle['counts']
            
        batch_res_df.append(image_df)
        gc.collect()
        # gc.collect()
    batch_res_df = pd.concat(batch_res_df)
    
    print(batch_res_df)
    # sys.exit()
    batch_res_df.replace(to_replace=[None], value = np.nan, inplace = True)
    print(batch_res_df)
    print(batch_res_df.shape)
    print('Batch predict end, Images : {}, labels : {}'.format(batch_res_df.shape[0], batch_res_df.shape[1]-3))
    
    return batch_res_df
        
        


        
        
def post_process_main():
    # print('/'.join(FLAGS.valid_true_batch_path.split('/')[:-1]))
    # sys.exit()
    K.clear_session()
    # In : (512,256,3,4)
    # Out : (512,256,3,4)
    model_name = [ k for k in os.listdir(FLAGS.model_path) if k.split('.')[-1]=='h5'][0]
    model_path = os.path.join(FLAGS.model_path,model_name)
    loaded_model = load_model(model_path, custom_objects = {
                                                           'bce_dice_loss' : bce_dice_loss,
                                                          'dice_coef' : dice_coef})
    
    valid_df = pd.read_csv(FLAGS.valid_true_batch_path)
    labels = valid_df.columns.tolist()[3:]
    print(valid_df)
    print(labels)
    # print(FLAGS.resize_shape[0])
    # print(FLAGS.resize_shape[1])
    colors = np.array(ast.literal_eval( valid_df.loc[0,'colors']))
    
    
    
    # In : size(2100,1400) rle(2100,1400)
    # out :
    # Img  - resize_size(512,256 ) - > resize(256,512) -> (512,256)
    # Mask  - resize_size(512,256 ) - > resize(256,512) -> (512,256)
    valid_generator_true  = SegmentDataGenerator(valid_df, batch_size = 1, 
                                  subset='train', shuffle=False,
                                 preprocess = img_preprocess, augmentation = None,
                                 resize_shape = (FLAGS.resize_shape[0],FLAGS.resize_shape[1]),train_path = FLAGS.valid_img_path)
    
    # In : size(2100,1400) rle(2100,1400)
    # out :
    # Img  - resize_size(512,256 ) - > resize(256,512) -> (512,256)
    # Mask  - resize_size(512,256 ) - > resize(256,512) -> (512,256)
    valid_generator_pred  = SegmentDataGenerator(valid_df, batch_size = 1, 
                                  subset='train', shuffle=False,
                                 preprocess = img_preprocess, augmentation = None,
                                 resize_shape = (FLAGS.resize_shape[0],FLAGS.resize_shape[1]),train_path = FLAGS.valid_img_path)
    # print( valid_generator_true[0][1])
    
    print(valid_generator_pred[0][0].shape)
    print(valid_generator_pred[0][1].shape)
    
#     print(valid_generator_pred[0][0][0,:,:,0])
#     print('@@@@@')
#     print(np.where(valid_generator_pred[0][1][0,:,:,0]==1))
#     print(np.where(valid_generator_pred[0][1][0,:,:,1]==1))
#     print(np.where(valid_generator_pred[0][1][0,:,:,2]==1))
#     print(np.where(valid_generator_pred[0][1][0,:,:,3]==1))
#     sys.exit()
    
    # label shape = (40, 512,256,4)
    valid_true_img_resize = np.zeros((valid_df.shape[0], FLAGS.resize_shape[0],FLAGS.resize_shape[1],len(labels)))
    # print(valid_true_img_resize.shape)
    # print(valid_generator_true[0][1].shape)
    # print('!!!!')
    for idx in tqdm(range(valid_df.shape[0])):
        # print( valid_generator_true[idx][1])
        valid_true_img_resize[idx, :, :, :] = valid_generator_true[idx][1]
    
    print(valid_true_img_resize.shape)
    print(valid_true_img_resize[0].shape)
    print(valid_true_img_resize[0][:,:,0].shape)
    print(valid_true_img_resize)
    print(np.where(valid_true_img_resize==1))
#     sys.exit()
    
    # print('$$$$$')
#     np.save('../valid_true.npy',valid_true_img_resize)
#     print
    # In : (512,256,3)
    # Out : (512,256,4)
    valid_pred_img = loaded_model.predict_generator(valid_generator_true, verbose=1)
#     np.save('../valid_pred.npy',valid_pred_img)
    ### Need Check
    valid_pred_img_proba = np.array(valid_pred_img >0.5,dtype = np.uint8)
    
    print("valid_true : " + str(valid_true_img_resize.shape), 
          "valid_pred : " + str(valid_pred_img.shape))
    
    
    print(" validation dice coefficient scores ")
    valid_dice_scores = dice_channel_torch(valid_pred_img_proba, valid_true_img_resize)
    
    
    # sys.exit
    ### resized mask & resized optimization parameter
    
    # In : (40,512,256,4)
    # Out : (40,512,256,4)
    # Post Processing
    post_matrix, post_nps = post_optimi(valid_true_img_resize,valid_pred_img, labels = labels, resize_shape = (FLAGS.resize_shape[0],FLAGS.resize_shape[1]) )
    
#     np.save('../valid_pred_post.npy',post_nps)
    
    print(" Post_processed validation dice coefficient scores ")
    processed_valid_dice_scores = dice_channel_torch(post_nps, valid_true_img_resize)
    
    # print("Improve {}, {} % ".format((processed_valid_dice_scores - valid_dice_scores) ,
    #                                  ( (processed_valid_dice_scores - valid_dice_scores)/valid_dice_scores*100) ))
#     sys.exit()
    
    
    print(" Score data frame save")
    valid_dice_scores = np.append(False , valid_dice_scores)
    processed_valid_dice_scores = np.append(True , processed_valid_dice_scores)
    score_nps = np.vstack((valid_dice_scores, processed_valid_dice_scores))
    score_label = ['post_processed'] + ['total_score']+ labels
    score_df = pd.DataFrame(score_nps, columns= score_label)
    score_df_path = os.path.join('/'.join(FLAGS.valid_true_batch_path.split('/')[:-1]), 'valid_score.csv')
    score_df.to_csv(score_df_path, index= False)
    print(" Score data frame save done !! ")
    
    
    
    
    post_save_path = os.path.join('/'.join(FLAGS.valid_true_batch_path.split('/')[:-1]), 'valid_post_grid.csv')
    post_matrix['colors'] = colors.reshape(-1,1)
    post_matrix.to_csv(post_save_path, index=False)
    print("Post processing matrix saved to {}".format(post_save_path))
    
    # proba_max = post_matrix['proba'].tolist()
    # reduce_size_max = post_matrix['reduce_size'].tolist()
    # pad_max = post_matrix['pad'].tolist()
    # convex_max = post_matrix['convex'].tolist()
    print(valid_df)
    print(post_nps)
    print(post_nps.shape)
    
    # In : (40, 512, 256, 4)
    # Out : rle : (40,2100,1400,4 ) is same (40, original width, original height, 4)
    post_rle_df =post_batch(valid_df= valid_df,post_arr = post_nps,label_names = labels )
    rle_df_path = os.path.join('/'.join(FLAGS.valid_true_batch_path.split('/')[:-1]), 'valid_batch_pred.csv')
    post_rle_df.to_csv(rle_df_path, index=False)
    
    valid_path = '/'.join(model_path.split('/')[:-2]) +  '/valid/'
    
    print(valid_true_img_resize.shape)
#     sys.exit()
    
    for r_idx in tqdm(range(post_rle_df.shape[0])):
        
        
        img = rle_mask2img(post_rle_df.iloc[r_idx], FLAGS.valid_img_path, origin_mask_array = valid_true_img_resize[r_idx])
        # print(img.shape)
        # sys.exit()
#         print(valid_path)
        # print( pred_test.iloc[r_idx]['ImageId'].split('.')[0])
        # print(pred_test.iloc[r_idx]['ImageId'].split('.')[1])
        # print('!!!!')
        print(post_rle_df.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + post_rle_df.iloc[r_idx]['ImageId'].split('.')[1])
        save_path = os.path.join(valid_path, post_rle_df.iloc[r_idx]['ImageId'].split('.')[0] + '_segment.' + post_rle_df.iloc[r_idx]['ImageId'].split('.')[1])
        img.save(save_path)
        print(img.size)
        print ('Segment {} done !!! '.format(post_rle_df.iloc[r_idx]['ImageId']))
    
    
    
    return post_matrix
                                                 
    



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = 'testing')
#     parser.add_argument('--target', required=True, help = 'train or predict')
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    parser.add_argument('--valid_true_batch_path', required=True)
    parser.add_argument('--model_path',required = True)
    parser.add_argument('--valid_img_path', required = True)
    # parser.add_argument('--origin_shape', nargs='+', type=int, default = [1400,2100])
    parser.add_argument('--resize_shape', nargs='+', type=int, default = [256,256])
    
    FLAGS, unparsed = parser.parse_known_args()
    
    
    post_process_main()
    
    

    # val_dice_table_ensemble.to_csv('val_dice_table_ensemble_2.csv')
