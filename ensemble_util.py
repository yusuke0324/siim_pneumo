import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy import ndimage
from skimage import morphology
import os
from tensorflow.python.keras.utils import to_categorical
from multiprocessing import Pool
from os import path

def ensemble_dirs(binary_mask_path_list=[], cpu_num=16, save_path=''):
    '''
    save ensembled numpy at save_path. ensemble all of files under each folder in binary_mask_path_list
    '''

    image_id_list = [path.split('/')[-1].split('.npy')[0] for path in glob(binary_mask_path_list[0] + '/*.npy')]
    p = Pool(processes=cpu_num)

    job_args = [(image_id, save_path, binary_mask_path_list) for image_id in image_id_list]
    list(tqdm(p.imap(_wrap_ensemble_preds, job_args), total=len(job_args)))

def _wrap_ensemble_preds(image_id, save_path, binary_mask_path_list):

    for i, base_path in enumerate(binary_mask_path_list):
        data_path = base_path + '/' + image_id + '.npy'
        if i==0:
            if not path.exists(data_path):
                print('{} is not exist!!!}'.format(data_path))
            else:
                #for tta, 'pred' -> 'mean_pred'?
                pred = np.load(data_path)[()]['pred']
                pred = to_categorical(pred, num_classes=2)
        else:
            if not path.exists(data_path):
                print('{} is not exist!!!}'.format(data_path))
            else:
                pred_2 = np.load(data_path)[()]['pred']
                pred_2 = to_categorical(pred_2, num_classes=2)

                pred = np.add(pred, pred_a)
    # what if the number of models are even?
    pred = np.argmax(pred, axis=-1)

    np.save(save_path + '/' + image_id, pred)