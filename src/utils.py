import cv2
import numpy as np
from pycocotools.mask import encode, decode, area, toBbox
from PIL import Image
import os
import sys
import ast

def rle2mask(rle, input_shape, resize_shape = None):
    rle_dict = dict.fromkeys(['size', 'counts'])
    # print('%%%%%')
    # print(input_shape)
    # rle(1400,2100)
    rle_dict['size'] = [input_shape[0], input_shape[1]] # 2100, 1400
    rle_dict['counts'] = rle # 2100, 1400
    # print(rle_dict)
    try:
        # 2100, 1400 -> 2100,1400 decoding
        mask = decode(rle_dict)
        # print(mask.shape)
        # mask = cv2.resize(decode(rle_dict).astype(np.uint8), (input_shape[1], input_shape[0]))
        # print(mask)
        # print(mask)
    except:
        # print('????')
        # print(input_shape)
        
        # 2100, 1400
        mask= np.zeros( (int(input_shape[0]), int(input_shape[1])) ).astype(np.uint8)
        # print(mask)
#     print(resize_shape)
        
    if resize_shape:
        # 512, 256
        mask = cv2.resize(mask, resize_shape)
    return mask

def mask2rle(mask):
    rle =  encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle




def img_preprocess(x):
    x = x.astype(np.float32)
    x = x/255
    return x



def mask2pad(mask, pad=2):
    # ENLARGE MASK TO INCLUDE MORE SPACE AROUND DEFECT
    w = mask.shape[1]
    h = mask.shape[0]
    
    # MASK UP
    for k in range(1,pad,2):
        temp = np.concatenate([mask[k:,:],np.zeros((k,w))],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK DOWN
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((k,w)),mask[:-k,:]],axis=0)
        mask = np.logical_or(mask,temp)
    # MASK LEFT
    for k in range(1,pad,2):
        temp = np.concatenate([mask[:,k:],np.zeros((h,k))],axis=1)
        mask = np.logical_or(mask,temp)
    # MASK RIGHT
    for k in range(1,pad,2):
        temp = np.concatenate([np.zeros((h,k)),mask[:,:-k]],axis=1)
        mask = np.logical_or(mask,temp)
    
    return mask 


def mask2contour(mask, width=3):
    # CONVERT MASK TO ITS CONTOUR
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3) 




def contour_convexHull(predictions):
    contours, hierarchy = cv2.findContours(predictions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(predictions.shape,np.uint8)
    if len(contours)>0:
        for contour in contours:
            area = cv2.convexHull(contour)
            cv2.drawContours(mask, [area], 0,1,-1)
    return mask


def masks_reduce(masks, mask_thres):
    
#     masks_copy = masks.copy() 
#     print(masks_copy.shape)
#     for idx in range(masks.shape[-1]):
    label_num, labeled_mask = cv2.connectedComponents(masks.astype(np.uint8))
    reduced_mask = np.zeros(masks.shape[:2],np.float32)

    for label in range(1, label_num):
        single_label_mask = (labeled_mask == label)
        if single_label_mask.sum() > mask_thres:
            reduced_mask[single_label_mask] = 1

#     masks_copy[:,:,idx] = reduced_mask
#     print(masks.shape)
    return reduced_mask


def predict_resize(msks, proba=[0.9, 0.9, 0.9, 0.9], pad_size=[10,10,10,10], reduce_size = [10000,10000,10000,10000], convex = [False,False,False,False], resize_shape = (512,256), label_names =['Fish','Flower', 'Gravel', 'Sugar']):
    
    resized_msks = np.zeros((msks.shape[0],msks.shape[1],msks.shape[2],len(label_names)),dtype=np.int8)
    for i in range(len(msks)):
        for j, label in enumerate(label_names):
            msk = msks[i , : , : , j]
            msk = np.array(msk >=proba[j], dtype = np.uint8)
            # msk = cv2.resize(msk,(origin_img_size[1],origin_img_size[0]), interpolation = cv2.INTER_LINEAR)
            msk = mask2pad(msk, pad_size[j])
            
            msk = masks_reduce(msk, reduce_size[j])
            if convex[j]==True:
                msk = contour_convexHull(msk.astype(np.uint8))
            
            
            resized_msks[i, : , : , j] = msk
            
            
            
    return resized_msks



# def predict_resize(msks, proba=[0.9, 0.9, 0.9, 0.9], pad =False, pad_size=[10,10,10,10], reduce = False, reduce_size = [10000,10000,10000,10000], convex = [False,False,False,False]):
    
#     resized_msks = np.zeros((msks.shape[0],350,525,4),dtype=np.int8)
#     for i in range(len(msks)):
#         for j, label in enumerate(['Fish', 'Flower', 'Gravel', 'Sugar']):
#             msk = msks[i , : , : , j]
#             msk = np.array(msk >=proba[j], dtype = np.uint8)
#             msk = cv2.resize(msk,(525,350), interpolation = cv2.INTER_LINEAR)
#             if pad:
#                 msk = mask2pad(msk, pad_size[j])
#             if reduce:
#                 msk = masks_reduce(msk, reduce_size[j])
#             if convex[j]==True:
#                 msk = contour_convexHull(msk.astype(np.uint8))
#             resized_msks[i, : , : , j] = msk


def hex2rgb(hex_code):
    h = hex_code.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    return rgb


def rle_mask2img(df_row, img_path, origin_mask_array=None):
    img = os.path.join(img_path , df_row['ImageId'])
    # print(img)
    # sys.exit()
    colors = ast.literal_eval(df_row['colors'])
    origin_size = ast.literal_eval(df_row['size'])
    img = cv2.imread(img)
    # color BGR => RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # (1400,2100)
    img = cv2.resize(img, (origin_size[0], origin_size[1]))
    print(img.shape)
    # print(colors)
    
    for l_idx, label in enumerate(df_row.index[3:]):
        hex_code =colors[l_idx]
        rgb_code = list(hex2rgb(hex_code))
        # rgb_code = 
#         print(df_row[label])
        
        # 2100, 1400
        msk = rle2mask(df_row[label], input_shape =  origin_size)
        # 1400, 2100
        msk = cv2.resize(msk, (origin_size[0],origin_size[1]))
#         print('!!!!')
#         print(msk.shape)
        msk = mask2pad(msk, pad=2)
        msk = mask2contour(msk, width =2)
        # print(msk)
        # print(msk.shape)
        # print(img)
        # print(img.shape)
        
        
        origin_mask = origin_mask_array[:, :,l_idx].copy()
        origin_mask = origin_mask.astype(np.uint8)
        origin_mask = cv2.resize(origin_mask, (origin_size[0],origin_size[1]))
        print(origin_mask.shape)
        print(msk.shape)
        print(img.shape)
        
        for c_idx , color in enumerate(rgb_code):
            # print(color)
            img[msk==1, c_idx] = color
#             print(origin_mask_array[:,:,l_idx].shape)
            
            if not origin_mask_array is None :
                img[origin_mask == 1, c_idx] = color
    # 2100, 1400    
    img = Image.fromarray(img)
    print(img.size)
    return img

def rle_mask2img_request(img, rles, labels, colors):
    origin_size = rles[0]['size']
    width = origin_size[0]
    height = origin_size[1]
    # (1400,2100)
    img = cv2.resize(img, (width, height))
    print(img.shape)
    
    for l_idx, label in enumerate(labels):
        hex_code = colors[l_idx]
        rgb_code = list(hex2rgb(hex_code))
        print(label)
        
        # (2100, 1400)
        msk = rle2mask(rles[l_idx]['counts'], input_shape = [width,height])
        # (1400, 2100)
        msk = cv2.resize(msk, (width, height))
        print(msk.shape)
        msk = mask2pad(msk, pad=2)
        msk = mask2contour(msk, width=2)
        
        for c_idx, color in enumerate(rgb_code):
            img[msk==1,c_idx] = color
    
    # (2100, 1400)
    img = Image.fromarray(img)
    print(img.size)
    return img
    


        
    
    
    
    
    

# def mask2rle(mask, input_shape, resize_shape):
    