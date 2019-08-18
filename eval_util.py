import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import data_prep, ensemble_util
import utils
from tensorflow.python.keras.utils import to_categorical
from multiprocessing import Pool


def eval_all_models_pred(pred_data_dir_base='/data/pneumo_log/val_1/val_predictions/2019_0805_0344/',
                         ensemble=True, thresh_list=None, cpu_num=16, save=True, score_column_name='score'):
    
    # got only folders
    pred_data_dir_list = glob(pred_data_dir_base + '/*/')
    for pred_data_dir in pred_data_dir_list:
        # each model folder
        print('start to eval {}'.format(pred_data_dir))
        eval_all_pred(pred_data_dir=pred_data_dir + '/')

    if ensemble:
        # save binary predictions at the best thresh for each
        _save_binary_best_thresh(pred_data_dir_list=pred_data_dir_list, column_name=score_column_name, cpu_num=cpu_num)
        # save final ensembled predictions
        save_final_pred_path = pred_data_dir_base + '/ensemble_preds/'
        ensemble_util.ensemble_dirs(binary_mask_path_list=glob(pred_data_dir_base + '/*/*/'), cpu_num=cpu_num, save_path=save_final_pred_path)
        # eval. these preds should be binary by now. just set 0.5
        df = eval_all_pred(pred_data_dir=save_final_pred_path, thresh_list=[0.5])
        return df




def _save_binary_best_thresh(pred_data_dir_list=['/data/pneumo_log/val_1/val_predictions/2019_0815_1742/best_weights/', '/data/pneumo_log/val_1/val_predictions/2019_0815_1742/snapshot_model_2/'],
                    column_name='score', cpu_num=16):

    '''
    pred_data_dir_list is a list of path that saves predictions (from pred_util._save_preds()) and should have data frame with thresh and column name
    if you want to use TTA column_name should be 'mean_score'
    '''
    print('start ensemble...')
    for pred_data_dir in pred_data_dir_list:

        df = pd.read_csv(pred_data_dir + '/thresh_evaluation.csv')

        # after groupby the index will be thresh so just get index as the thresh value
        best_thresh = df.groupby('thresh').mean()[column_name].idxmax()
        best_score = df.groupby('thresh').mean()[column_name].max()
        print('for {}, best score is {} at thresh={}'.format(pred_data_dir, best_score, best_thresh))

        p = Pool(processes=cpu_num)
        # set save_binary is True and save binary pred based on the best thresh
        job_args = [(data_path, [best_thresh], True) for data_path in glob(pred_data_dir + '/*.npy')]

        list(tqdm(p.imap(_wrap_load_eval_w_thresh_list, job_args), total=len(job_args)))



def eval_all_pred(pred_data_dir='/data/pneumo_log/val_1/val_predictions/2019_0805_0344/best_weights/', thresh_list=None, cpu_num=16, save=True):
    '''
    Return all scores based on thresh in thresh_list for all files under pred_data_dir. pred_data_dir should be from pred_util._save_preds()
    '''
    
    if thresh_list is None:
        thresh_list = np.linspace(0, 1, 21)#0, 0.05, 0.1,,, so on
        thresh_list = np.concatenate([thresh_list,np.linspace(0, 0.05, 10)],axis=0) # 0, 0.00555556, 0.0111, ...
        thresh_list = np.unique(thresh_list)    

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

def _load_eval_w_thresh_list(data_path, thresh_list=[0.5], save_binary=False):
    '''
    set save_path if save_binary is True.
    '''
    
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
        dice, binary_pred = evaluation(mask, pred,  thresh)
        # for save. need mask here to use this functin to the saved data to evaluate
        data = {'pred':binary_pred, 'mask':mask}
        result['score'] = dice
        if 'aug_pred' in data.keys():
            aug_dice, aug_binary_pred = evaluation(mask, aug_pred,  thresh)
            mean_dice, mean_binary_pred = evaluation(mask, mean_pred,  thresh)
            result['aug_score'] = aug_dice
            result['mean_score'] = mean_dice
            data['aug_pred'] = aug_binary_pred
            data['mean_pred'] = mean_binary_pred
            
        result['image_id'] = image_id
        result['thresh'] = thresh

        if save_binary:
            save_dir = '/'.join(data_path.split('/')[:-1]) + '/binary_thresh_' + str(thresh) + '/'
            data_prep._make_dir(save_dir)
            file_name = save_dir + '/' + str(image_id)
            np.save(file_name, data)
        
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
        return 1.0, pred
    # if the ground truth has no mask but prediction has masks, return 0
    elif (gt.sum() == 0) and (pd.sum() > 0):
        return 0.0, pred
    # if the ground truth has mask but prediction has no masks, ofcause return 0
    elif (gt.sum() > 0) and (pd.sum() == 0):
        return 0.0, pred
        
    else:
        dice = 2*np.logical_and(pd, gt).sum()/(
        pd.sum() + gt.sum()
    )
        
        # return dice and binary pred
        return dice, pred
        

    
    
    
    
        