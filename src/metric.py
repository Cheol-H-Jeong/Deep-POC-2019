from keras import backend as K
from keras.losses import binary_crossentropy
import numpy as np
from tqdm import tqdm

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))



def dice_channel_label(probability, truth):
    batch_size = truth.shape[0]
    channel_num = truth.shape[-1]
    mean_dice_channel = 0.
    
#     channel_1 = 0.
#     channel_2 = 0.
#     channel_3 = 0.
#     channel_4 = 0.
    for i in range(batch_size):
#         for j in range(channel_num):
        channel_dice = dice_single_channel(probability[i, :,:], truth[i, :, :])
        mean_dice_channel += channel_dice/(batch_size)
#         mean_dice_channel += channel_dice/(batch_size)

#             mean_channels[j] += channel_dice/batch_size
#     print("Channel_1 : {}, Channel_2 : {}, Channel_3 : {},Channel_4 : {},".format(
#         round(mean_channels[0],5 ), 
#         round(mean_channels[1],5 ), 
#         round(mean_channels[2],5 ), 
#         round(mean_channels[3],5 )))
    return mean_dice_channel



def dice_channel_torch(probability, truth):
    batch_size = truth.shape[0]
    channel_num = truth.shape[-1]
    mean_dice_channel = 0.
    
    
    mean_channels = [0.]* channel_num

    for i in tqdm(range(batch_size)):
        for j in range(channel_num):
            channel_dice = dice_single_channel(probability[i, :,:,j], truth[i, :, :, j])
            mean_dice_channel += channel_dice/(batch_size * channel_num)
            
            mean_channels[j] += channel_dice/batch_size
#     print(channel_num)
    score_text = ' : {}, '.join(['channnel_{}'.format(k + 1) for k in range(channel_num)]) + ' : {}'
#     print(score_text)
    score = np.round(np.array(mean_channels), 5)
    total_score = np.round(np.append(mean_dice_channel, np.array(mean_channels)),5)
    score_text = score_text.format(*score)
    print("Mean_dice_channel : {} ".format(total_score[0]))
    print(score_text)

    return total_score


def dice_single_channel(probability, truth,  eps = 1E-9):
    p = probability.astype(np.float32)
    t = truth.astype(np.float32)
    dice = (2.0 * (p * t).sum() + eps)/ (p.sum() + t.sum() + eps)
    return dice