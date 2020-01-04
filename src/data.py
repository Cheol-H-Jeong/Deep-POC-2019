import pandas as pd
import numpy as np
import os
from pycocotools.coco import COCO
from pycocotools.mask import encode, decode, area, toBbox
import json
from tqdm import tqdm
import cv2

def df_gen(json_location):
    
    
    coco = COCO(json_location)
    cats = coco.loadCats(coco.getCatIds())
    catids = coco.getCatIds()
    nms=[cat['name'] for cat in cats]
    colors = [cat['color'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms_super = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms_super)))
    
    column_names = ['ImageId','size','colors']
#     print
    column_names = column_names + nms
    train_df = pd.DataFrame(columns=column_names)
    
    imgIds = coco.getImgIds()
    for img_idx, img_id in tqdm(enumerate(imgIds)):
        rles = dict.fromkeys(column_names)
        
        img = coco.loadImgs(img_id)
        rles['ImageId']  = img[0]['file_name']
        rles['size'] = [img[0]['width'], img[0]['height']] # 2100, 1400
        rles['colors'] = colors
        
        for catid in catids:
            
        
            annId = coco.getAnnIds(imgIds = img_id, catIds = catid)
            anns = coco.loadAnns(annId)
            
#             if anns ==[]:
#                 continue
            mask_label = np.zeros((img[0]['width'],img[0]['height']), dtype = np.uint8) # 2100, 1400
            for ann_idx, ann in enumerate(anns):
                try:
                    
                    mask_ = coco.annToMask(ann) # (1400, 2100)
                    mask_ = cv2.resize(mask_, (rles['size'][1], rles['size'][0])) # 2100, 1400
                    mask_label = np.maximum(mask_label, mask_) # 2100, 1400
                except:
                    pass
                
            mask_label = np.asfortranarray(mask_label)
            
            rle_ = encode(mask_label)
    #                 print(ann['category_id'])
            cat_name = list(filter(lambda x : x['id']==catid, cats))[0]['name']
    #                 print(cat_id)
            # rle (2100, 1400)
            rles[cat_name] = rle_['counts']
            
            
#         dict_keys = nms.append(column_names)
        
        
#         print(img[0])
#         print(anns)
#         break()

#         rles['ImageId']  = img[0]['file_name']
#         rles['size'] = [img[0]['width'], img[0]['height']] # 2100, 1400
#         rles['colors'] = colors
        
#         print(anns)
#         break
#         for ann_idx, ann in enumerate(anns):
#             try:
#                 # (1400, 2100)
#                 mask_ = coco.annToMask(ann)
#                 # (2100, 1400)
#                 mask_ = np.asfortranarray(cv2.resize(mask_, (rles['size'][1], rles['size'][0])))
#                 # rle (2100, 1400)
#                 rle_ = encode(mask_)
# #                 print(ann['category_id'])
#                 cat_id = list(filter(lambda x : x['id']==ann['category_id'], cats))[0]['name']
# #                 print(cat_id)
#                 rles[cat_id] = rle_['counts']
                
#             except:
#                 pass
        
        train_df = train_df.append(rles,
                                   ignore_index=True)
        train_df.replace(to_replace=[None], value=np.nan, inplace=True)    
            
            
    return train_df

def df_save(save_path, df):
    df.to_csv(save_path)
    print('{}_save'.format(save_path))

if __name__ == '__main__':
    coco_path = '../coco-annotator/datasets/cloud_segment_test/.exports/coco-1574824539.779883.json'
    df = df_gen(coco_path)
    print(df.head())