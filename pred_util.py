import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy import ndimage
from skimage import morphology
import os
import utils, data_generator, train_util, models, loss_util, data_prep
from keras.models import load_model
from tensorflow.python.keras.utils import to_categorical
import cv2

def _pred_img(image, model):
    '''
    image can be pre preprocessed image. (original numpy)
    '''
    
    #(256, 256)
    input_shape = model.layers[0].input_shape[1:3]
    # 3
    ch = model.layers[0].input_shape[3]
    
    original_shape = image.shape[:2]
    
    # preprocessing. This should be the same as data generator does
    image = cv2.resize(image, input_shape)
    image = data_generator._normalize(image)
    if ch == 3:
        image = np.stack((image, image, image), axis=-1)
    elif ch == 1:
        image = np.expand_dims(image, axis=-1)
    
    # bach 1
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)
    pred = np.squeeze(pred)
    
    # back to original shape
    pred = cv2.resize(pred, original_shape)
    
    return pred

def _load_model(model_path='/data/pneumo_log/val_1//2019_0805_0344/best_weights.hdf5'):
    zero_weight = 0.1
    one_weight = 0.9
    custom_objects = { 'weighted_binary_crossentropy':loss_util.create_weighted_binary_crossentropy(zero_weight, one_weight),
           'dice_coef':loss_util.create_dice_coef(),
           'dice_coef_flat':loss_util.dice_coef_flat,
           'bce_dice_loss':loss_util.bce_dice_loss,
           'dice_loss_flat':loss_util.dice_loss_flat,
           'my_dice_metric':loss_util.my_dice_metric}
    
    return load_model(model_path, custom_objects=custom_objects)

def save_preds_all_models(model_base_path='/data/pneumo_log/val_1/2019_0805_0344/', 
                          data_path='/data/pneumo/fold/1/',
                          tta=True):
    '''
    save all predictions as numpy with all models under the model_base_path
    '''

    model_path_list = glob(model_base_path + '/*[.h5, .hdf5]')

    for model_path in model_path_list:
        _save_preds(model_path=model_path, data_path=data_path, tta=tta)

def _save_preds(model_path='/data/pneumo_log/val_1/2019_0805_0344/best_weights.hdf5',
                data_path='/data/pneumo/fold/1/',
                tta=True):

    '''
    save prediction as numpy. if tta=True, save horizontal flip too. The saved 'aug_pred' mask has been fliped again.

    '''
    print('start pred with {}'.format(model_path))
    data_path_list = glob(data_path + '*.npy')
    
    #'/data/pneumo_log/val_1/'
    base_save_path = '/'.join(model_path.split('/')[:-2])
    #2019_0805_0344
    folder_name = model_path.split('/')[-2]
    # '/data/pneumo_log/val_1/val_predictions/2019_0805_0344/best_weights/'
    # extended for snapshots
    save_path = base_save_path + '/val_predictions/' + folder_name + '/' + model_path.split('/')[-1].split('.')[0] + '/'
    
    # loading model
    model = _load_model(model_path=model_path)
    
    print('saving dirs: {}'.format(save_path))
    data_prep._make_dir(save_path)
    
    for data_path in tqdm(data_path_list):
        # this includes .npy
        file_name = data_path.split('/')[-1]
        data = np.load(data_path)[()]
        img = data['img']
        data['pred'] = _pred_img(img, model)
        if tta:
            flip = np.flip(img, axis=-1)
            flip_pred = _pred_img(flip, model)
            # save after flip
            data['aug_pred'] = flip_pred
            data['mean_pred'] = 0.5*data['pred'] + 0.5*data['aug_pred']
        
        np.save(save_path + file_name, data)
    
    

    
    
    