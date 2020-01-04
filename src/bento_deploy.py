from keras.models import load_model
import os
import sys
import keras
import tensorflow as tf
from metric import *
from model import *
from keras import backend as K
from metric import bce_dice_loss,bce_logdice_loss,dice_coef,dice_loss
from model import *
import efficientnet.keras as efn
import argparse
import pandas as pd
from pprint import pprint



FLAGS = None

# print('model_path : ')
# print(FLAGS.model_path) 
# print('valid_post_path : ')
# print(FLAGS.valid_post_path)
# print('deploy_save_path : ')
# print(FLAGS.deploy_save_path)

    
    
    
#     return FLAGS.model_path, FLAGS.valid_post_path

def main():
    print( FLAGS.model_path )
    print(FLAGS.valid_post_path)
    post_grid_file = FLAGS.valid_post_path.split('/')[-1]
    valid_abs_path  = os.path.dirname(os.path.abspath(FLAGS.valid_post_path))
    post_grid_path = os.path.join(valid_abs_path, post_grid_file)
    print(valid_abs_path)
    print(post_grid_path)
    
#     deploy_valid_path = 'valid_path = ' + '{' + "'valid_post_grid'" +  ' : ' + "'{}'".format(post_grid_path) +  '}'
#     print(deploy_valid_path)
    
    post_df = pd.read_csv(post_grid_path)
    post_dict = post_df.to_dict(orient='list')
    
#     res = """valid_path = {'valid_post_grid' : {}.format(obs_path)}"""
#     sys.exit()
    print('#'*100)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(root_dir)
    
    print(os.path.join(root_dir, 'deploy_valid_grid.py'))
#     sys.exit()
#     with open(os.path.join(root_dir, 'deploy_valid_path.py'), 'w') as text:
#         text.write(deploy_valid_path)
#     sys.exit()
    with open(os.path.join(root_dir, 'deploy_valid_grid.py'), 'w') as text:
        text.write('grid_value = ')
        pprint(post_dict, stream = text)
        
    
# sys.exit()

    model_name = [ k for k in os.listdir(FLAGS.model_path) if k.split('.')[-1]=='h5'][0]
    model_path = os.path.join(FLAGS.model_path,model_name)

    loaded_model = load_model( model_path,
                             custom_objects = {
                                 'bce_dice_loss' : bce_dice_loss,
                                 'dice_coef' : dice_coef
                             }
                             )
    print("model loaded {}".format(str(loaded_model)))
    
    
    print('Done')
    
    from keras_segment import KerasSegmentationService

    segment_svc = KerasSegmentationService.pack(segmentation = loaded_model)
#     sys.exit()
    saved_path = segment_svc.save(base_path = FLAGS.deploy_save_path, version =
                                  '{}'.format(str(FLAGS.version_number)))
    print(saved_path)
    print("model_deploy saved to {}".format(saved_path))



if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description = 'deploying model')
#     parser.add_argument('--target', required=True, help = 'train or predict')
    # parser.add_argument('--train_path')
    # parser.add_argument('--test_path')
    
    parser.add_argument('--model_path', required=True, type = str)
    parser.add_argument('--valid_post_path', required=True, type = str)
    parser.add_argument('--deploy_save_path', required=True, type = str)
    parser.add_argument('--version_number', required=True, type = str)
    
    
#     parser.add_argument('--json_path', required=True)
#     parser.add_argument('--img_path', required=True)
#     parser.add_argument('--model_name',default= local_time)
#     parser.add_argument('--epoch', default = 50, type =int)
#     parser.add_argument('--batch_size', default = 16, type =int)
    
#     parser.add_argument('--resize_shape', nargs='+', type=int, default = [256,256])
#     parser.add_argument('--augument',default = True)
    
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
    