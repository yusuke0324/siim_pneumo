import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import utils
from tensorflow.python.keras.utils import to_categorical
from multiprocessing import Pool



def eval_all_pred(pred_data_dir='/data/pneumo_log/val_1/val_predictions/2019_0805_0344/', thresh_list=None, cpu_num=16, save=True):
    '''
    Return all scores based on thresh in thresh_list for all files under pred_data_dir. pred_data_dir should be from pred_util._save_preds()
    '''
    
    if thresh_list is None:
        thresh_list = np.linspace(0, 1, 21)#0, 0.05, 0.1,,, so on
    data_path_list = glob(pred_data_dir + '/*.npy')
    
    p = Pool(processes=cpu_num)
    
    job_args = [(data_path, thresh_list) for data_path in data_path_list]
    
    result_list = list(tqdm(p.imap(_wrap_load_eval_w_thresh_list, job_args), total=len(job_args)))
    result_list_flat = list(np.array(result_list).flatten())
    df =  pd.DataFrame(result_list_flat)

    if save:
        df.to_csv(pred_data_dir + '/thresh_evaluation.csv', index=False)
    return df

def _wrap_load_eval_w_thresh_list(args):
    return _load_eval_w_thresh_list(*args)

def _load_eval_w_thresh_list(data_path, thresh_list=[0.5]):
    
    data = np.load(data_path)[()]
    mask = data['mask']
    pred = data['pred']
    image_id = data_path.split('/')[-1].split('.npy')[0]
    if 'aug_pred' in data.keys():
        # TTA
        aug_pred = data['aug_pred']
        mean_pred = data['mean_pred']
    
    result_list = []
    for thresh in thresh_list:
        result = {}
        dice = evaluation(mask, pred,  thresh)
        result['score'] = dice
        if 'aug_pred' in data.keys():
            aug_dice = evaluation(mask, aug_pred,  thresh)
            mean_dice = evaluation(mask, mean_pred,  thresh)
            result['aug_score'] = aug_dice
            result['mean_score'] = mean_dice
            
        result['image_id'] = image_id
        result['thresh'] = thresh
        
        result_list.append(result)

    return result_list
    

def evaluation(gt_mask, pred, thresh=0.5):
    '''
    evaluate dice. pred and gt_mask should be (h, w) and pred values are 0-1. if the values are not binary, use thesh to make it binary.
    '''
    
    if gt_mask.shape != pred.shape:
        print('gt_mask and pred should be the same shape')
        return
    elif len(gt_mask.shape) > 4:
        print('too many rank. only accept 3 or 4')
        return 
    elif len(gt_mask.shape) == 3:
        print('squeeze to make rank 2')
        gt_mask = np.squeeze(gt_mask)
        pred = np.squeeze(pred)
       
    # if the pred is 0-1, make it binary with the thresh
    pred = np.where(pred>thresh, 1, 0)
#     try:
        # Compute Dice
    gt = np.greater(gt_mask, 0)
    pd = np.greater(pred, 0)
    # if the ground truth and prediction have no mask, return 1
    if (gt.sum() == 0) and (pd.sum() == 0):
        return 1.0
    # if the ground truth has no mask but prediction has masks, return 0
    elif (gt.sum() == 0) and (pd.sum() > 0):
        return 0.0
    # if the ground truth has mask but prediction has no masks, ofcause return 0
    elif (gt.sum() > 0) and (pd.sum() == 0):
        return 0.0
        
    else:
        dice = 2*np.logical_and(pd, gt).sum()/(
        pd.sum() + gt.sum()
    )
        return dice
        

    
    
    
    
        