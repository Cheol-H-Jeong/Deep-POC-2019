import numpy as np
from PIL import Image
import pandas as pd
import cv2
import sys
import os
import json
print(os.path.dirname(os.path.realpath(__file__))) 
sys.path.append(os.path.dirname(os.path.realpath(__file__)) )
print(sys.path)

from bentoml import api, artifacts, env, BentoService
from bentoml.artifact import KerasModelArtifact
from bentoml.handlers import ImageHandler
# from metric import *
# from model import *

import metric, utils, model
import deploy_valid_grid

import efficientnet.keras as efn


# from utils import *
# from deploy_valid_path import valid_path
# from numpy.random import seed
# seed(42)
# from tensorflow import set_random_seed
# set_random_seed(42)

# FLAGS = None
# print(FLAGS.valid_post_path)

# valid_path = '../res/segment/test_20/valid/valid_post_grid.csv'
# with open('./deploy_valid_path.txt') as f:
#     valid

print(deploy_valid_grid.grid_value)
post_df = pd.DataFrame(deploy_valid_grid.grid_value)
#     print(deploy_valid_path.valid_path['valid_post_grid'])

#     post_df = pd.read_csv(str(deploy_valid_path.valid_path['valid_post_grid']))
print(post_df)
#     sys.exit()
print('Load dict file Post grid')
label_names = post_df['label'].values.tolist()
proba = post_df['proba'].values.tolist()
reduce_size = post_df['reduce_size'].values.tolist()
pad = post_df['pad'].values.tolist()
convex = post_df['convex'].values.tolist()
colors = post_df['colors'].values.tolist()
print(label_names)




@env(pip_dependencies=['keras==2.2.5', 'tensorflow-gpu==1.14.0', 'Pillow', 'numpy','opencv-python', 'efficientnet', 'pandas','cython','git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'])
@artifacts([KerasModelArtifact(name = 'segmentation', 
                               custom_objects = {
#                               'tf':tf,
                                   'bce_dice_loss' : metric.bce_dice_loss,
                                    'dice_coef' : metric.dice_coef
                               }
                         ) 
           ]
          )
class KerasSegmentationService(BentoService):
#     print(deploy_valid_grid.grid_value)
#     post_df = pd.DataFrame(deploy_valid_grid.grid_value)
# #     print(deploy_valid_path.valid_path['valid_post_grid'])
    
# #     post_df = pd.read_csv(str(deploy_valid_path.valid_path['valid_post_grid']))
#     print(post_df)
# #     sys.exit()
#     print('Load dict file Post grid')
#     label_names = post_df['label'].values.tolist()
#     proba = post_df['proba'].values.tolist()
#     reduce_size = post_df['reduce_size'].values.tolist()
#     pad = post_df['pad'].values.tolist()
#     convex = post_df['convex'].values.tolist()
#     colors = post_df['colors'].values.tolist()
#     print(label_names)
    
    @api(ImageHandler, pilmode='RGB')
    def predict(self, img):
        print(img.shape)
        img_width = img.shape[1]
        img_height = img.shape[0]
        img_channel = img.shape[2]
        
        # (1,256,256,3)
        img_resized = cv2.resize(img.astype(np.uint8), (256,256)).astype(np.float)
        print('#'*100)
        print(img_resized.shape)
        # img = Image.fromarray(img).resize((256, 256))
        img_resized /= 255.0
        print(img_resized.shape)
        img_resized = np.expand_dims(img_resized, axis = 0)
        # img = np.array(img.getdata()).reshape((1,256,256,3))
        
        # (1,256,256,4) ==> predcited
        predicted = self.artifacts.segmentation.predict(img_resized)
        predicted_post = utils.predict_resize(predicted,
                                       proba = proba,
                                       pad_size = pad,
                                       reduce_size = reduce_size,
                                       convex = convex,
                                       label_names = label_names)
        print(predicted_post.shape)
        rle_dict = {}
        rle_list = []
        for l_idx, label in enumerate(label_names):
            l_mask = predicted_post[0][:,:, l_idx].astype(np.uint8)
            print(l_mask.shape)
            # (2100,1400)
            l_mask = cv2.resize(l_mask, (img_height, img_width))
            print(l_mask.shape)
            label_rle = utils.mask2rle(l_mask)
            rle_list.append(label_rle)
        

#         img_masked = rle_mask2img_request(img, rle_list, label_names, colors)
#         img_str = img_masked.tobytes()
        
        rle_dict['rle'] = [{'size' : rle['size'], 'counts' : str(rle['counts'])} for rle in rle_list]
#         rle_dict['image'] = str(img_str)
        
        rle_json = json.dumps(rle_dict)
#         print(rle_list)
        
        
        
        return rle_json
    


# if __name__ == '__main__':
    
    
#     parser = argparse.ArgumentParser(description = 'deploying model')
# #     parser.add_argument('--target', required=True, help = 'train or predict')
#     # parser.add_argument('--train_path')
#     # parser.add_argument('--test_path')
    
# #     parser.add_argument('--model_path', required=True)
#     parser.add_argument('--valid_post_path', required=True)
# #     parser.add_argument('--deploy_save_path', required=True)
# #     parser.add_argument('--version_number', requried=True, type = int)
    
    
# #     parser.add_argument('--json_path', required=True)
# #     parser.add_argument('--img_path', required=True)
# #     parser.add_argument('--model_name',default= local_time)
# #     parser.add_argument('--epoch', default = 50, type =int)
# #     parser.add_argument('--batch_size', default = 16, type =int)
    
# #     parser.add_argument('--resize_shape', nargs='+', type=int, default = [256,256])
# #     parser.add_argument('--augument',default = True)
    
    
#     FLAGS, unparsed = parser.parse_known_args()
    
#     KerasSegmentationService