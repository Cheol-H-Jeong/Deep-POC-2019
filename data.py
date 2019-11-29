import pandas as pd
import numpy as np
import os
from pycocotools.coco import COCO
import json
from tqdm import tqdm

def df_gen(json_location):
    
    
    coco = COCO(json_location)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms_super = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms_super)))
    
    column_names = ['ImageId','size']
#     print
    column_names = column_names + nms
    train_df = pd.DataFrame(columns=column_names)
    
    imgIds = coco.getImgIds()
    for img_idx, img_id in tqdm(enumerate(imgIds)):
        img = coco.loadImgs(img_id)
        annId = coco.getAnnIds(imgIds = img_id)
        anns = coco.loadAnns(annId)
        
        dict_keys = nms.append(column_names)
        rles = dict.fromkeys(column_names)
        
#         print(img[0])
#         print(anns)
#         break()
        rles['ImageId']  = img[0]['file_name']
        rles['size'] = [img[0]['height'], img[0]['width']]
#         print(anns)
#         break
        for ann_idx, ann in enumerate(anns):
            try:
                rle_ = coco.annToRLE(ann)
#                 print(ann['category_id'])
                cat_id = list(filter(lambda x : x['id']==ann['category_id'], cats))[0]['name']
#                 print(cat_id)
                rles[cat_id] = rle_['counts']
                
            except:
                pass
        
        train_df = train_df.append(rles,
                                   ignore_index=True)
        train_df.replace(to_replace=[None], value=np.nan, inplace=True)    
            
            
    return train_df

def df_save(save_path, df):
    df.to_csv(save_path)
    print('{}_save'.format(save_path))

if __name__ == '__main__':
    coco_path = './coco-annotator/datasets/cloud_segment_test/'
    df = df_gen(coco_path)
    print(df.head())