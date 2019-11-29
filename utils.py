import cv2
import numpy as np
from pycocotools.mask import encode, decode, area, toBbox


def rle2mask(rle, input_shape, resize_shape):
    rle_dict = dict.fromkeys(['size', 'counts'])
    rle_dict['size'] = input_shape
    rle_dict['counts'] = rle
#     print(rle_dict)
    try:
        mask = decode(rle_dict)
    except:
        mask= np.zeros( input_shape ).astype(np.uint8)
#     print(resize_shape)
        
    mask = cv2.resize(mask, resize_shape)
    return mask




def img_preprocess(x):
    x = x.astype(np.float32)
    x = x/255
    return x


# def mask2rle(mask, input_shape, resize_shape):
    