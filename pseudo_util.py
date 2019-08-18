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
import data_prep
import pydicom

def _get_pseudo_label(pred_data_path, zero_max=0.005, one_min=0.8, test_data=True):

    # the data should have been saved by submission_util.make_submissions()
    if test_data:
        pred = np.load(pred_data_path)[()]['pred_row']
    else:
        # TODO: for TTA, the key should be mean_pred
        # this numpy should have been saved by pred_util._save_preds()
        pred = np.load(pred_data_path)[()]['pred']

    if np.max(pred) < zero_max:
        # this should be empty!
        pred[:] = 0
        flag=True
    elif np.max(pred) < one_min:
        # no confident label. Dispose this.
        flag=False
    else:
        # make label only strong confidences
        pred = np.where(pred>one_min, 1, 0)
        flag=True
    return pred, flag

def _wrap_save_pseudo_label(args):
    return _save_pseudo_label(*args)

def _save_pseudo_label(pred_data_path, save_path, zero_max=0.005, one_min=0.8, test_base_path='/data/pneumo/dicom-images-test/', test_data=True):

    pseudo_label, flag = _get_pseudo_label(pred_data_path, zero_max=zero_max, one_min=one_min, test_data=test_data)
    if flag:
        img_id = pred_data_path.split('/')[-1].split('.npy')[0]
        if test_data:
            test_data_path = glob(test_base_path+'/*/*/'+img_id+'*')[0]
            img = pydicom.dcmread(test_data_path).pixel_array
        else:
            img_path = test_base_path + '/{}.npy'.format(img_id)
            img = np.load(img_path)[()]['img']
        data = {'img':img, 'mask':pseudo_label}
        np.save(save_path+'/'+img_id, data)


def save_make_pseudo_data(pred_data_dir='/data/pneumo_log/val_1/2019_0815_1742/submission/snapshot_model_2/',
                          zero_max=0.005,
                          one_min=0.8, cpu_num=16, test_base_path='/data/pneumo/dicom-images-test/', test_data=True):

    '''
    save pseudo label as dictionary {'img':, 'mask'} under pred_data_dir+'/pseudo/'
    This can be applied to train data (fold) too. set test_data=False
    '''

    if test_data:
        save_path = pred_data_dir + '/pseudo/'
    else:
        save_path = pred_data_dir + '/pseudo_train_fold/'
    data_prep._make_dir(save_path)
    print('start to make pseudo label under {}'.format(save_path))
    pred_data_path_list = glob(pred_data_dir + '/*.npy')

    p = Pool(processes=cpu_num)
    job_args = [(pred_data_path, save_path, zero_max, one_min, test_base_path, test_data) for pred_data_path in pred_data_path_list]
    list(tqdm(p.imap(_wrap_save_pseudo_label, job_args), total=len(job_args)))


    








