import numpy as np
import pandas as pd
import pydicom
from glob import glob
from tqdm import tqdm
from mask_functions import mask2rle
import pred_util, data_prep
import ensemble_util

def make_submission(model_path, thresh, small_thresh=2048, test_base_path='/data/pneumo/dicom-images-test/', save=False):

    '''
    save submission file under a folder that model file is saved.
    Set save=True if you want to save binary pred numpy.
    all binary (small removed) npy and submission csv will be saved in /data/pneumo_log/val_1/2019_0815_1742/submission/best_weights/

    '''
    # with snapshot, there is a case where there are multiple model files in the dir so make `submission` folder
    # best_weights
    model_file_name = model_path.split('/')[-1].split('.')[0]
    # 2019_0815_1742
    dir_name = model_path.split('/')[-2]
    # /data/pneumo_log/val_1/2019_0815_1742/submission/best_weights
    save_dir = '/'.join(model_path.split('/')[:-1]) + '/submission/' + model_file_name
    data_prep._make_dir(save_dir)
    # 'submission_' 
    file_name = 'submission_' + dir_name + '_' + model_file_name + '.csv'
    test_data_path_list = glob(test_base_path + '/*/*/*.dcm')
    model = pred_util._load_model(model_path=model_path)
    rles = []
    im_ids = []

    for path in tqdm(test_data_path_list):
        im_id = path.split('/')[-1].split('.dcm')[0]
        im_ids.append(im_id)
        im = pydicom.dcmread(path).pixel_array
        # no need preprocess
        pred = pred_util._pred_img(im, model)
        # pred is already 1024 * 1024 but no binary. its values are 0-1
        binary_pred = np.where(pred>thresh, 1, 0)

        # zero out the smaller regions
        if binary_pred.sum() < small_thresh:
            binary_pred[:] = 0
        # binary -> 0, 255 and transpose for submission format
        if save:
            # save numpy. This is usually needed for ensemble. In order to use ensemble_util._ensemble_preds on the data later,
            # save it as dictionary like {'pred':}
            np.save(save_dir+'/'+im_id,{'pred':binary_pred, 'pred_row':pred})
        binary_pred = (binary_pred.T*255).astype(np.uint8)

        rles.append(mask2rle(binary_pred, 1024, 1024))

    sub_df = pd.DataFrame({'ImageId':im_ids, 'EncodedPixels':rles})
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    sub_df.to_csv(save_dir + '/' + file_name, index=False)
    if save:
        # for later ensemble
        return save_dir


def make_ensemble_submission(model_path_list,
                            thresh_list=None,
                            small_thresh=2048,
                            test_base_path='/data/pneumo/dicom-images-test',
                            column_name='score',
                            save=False,
                            save_path=None):
    '''
    ensemble: This function call make_submission() each and save each submission csv.
    1. save binary pred based on each thresh as numpy (w/ remove small masks)
    2. ensemble (max vote)
    3. remove small masks
    4. mask2rle
    5. save submission file
    '''

    if save_path is None:
        # make save path based on first model path
        save_path = '/'.join(model_path_list[0].split('/')[:-1]) + '/ensemble_submission/'
    print('start making ensemble submission under {}'.format(save_path))
    data_prep._make_dir(save_path)
    # get best thresholds if thresh_list is None
    if thresh_list is None:
        thresh_list = []
        for model_path in model_path_list:
            thresh = _get_best_threshold(model_path, column_name=column_name)
            thresh_list.append(thresh)
    binary_mask_path_list = []
    for i, model_path in enumerate(model_path_list):
        print('start pred test data by {}'.format(model_path))
        binary_mask_path = make_submission(model_path, thresh_list[i], small_thresh=small_thresh, test_base_path=test_base_path, save=True)
        # save path should be like /data/pneumo_log/val_1/2019_0815_1742/submission/best_weights/

        binary_mask_path_list.append(binary_mask_path)

    # save ensembled data
    ensemble_util.ensemble_dirs(binary_mask_path_list=binary_mask_path_list, cpu_num=16, save_path=save_path, data_key='pred')

    # assume the pred is already binary. don't need to set thresh
    sub_df = _make_submission_from_predictions(save_path,small_thresh=small_thresh)
    file_name = 'ensemble_submission.csv'
    sub_df.to_csv(save_path + '/' + file_name, index=False)
    print('saved submission file at {}'.format(save_path + '/' + file_name))
    print('$kaggle competitions submit siim-acr-pneumothorax-segmentation -f {} -m "snapshot ensembles"'.format(save_path + '/' + file_name))



def _make_submission_from_predictions(preds_path, small_thresh=2048, thresh=0.5):


    pred_path_list = glob(preds_path + '/*.npy')

    rles = []
    im_ids = []
    # TODO make multiprocessing it takes about 25 min...
    for pred_path in tqdm(pred_path_list):
        pred = np.load(pred_path)[()]['pred']
        im_id = pred_path.split('/')[-1].split('.npy')[0]
        im_ids.append(im_id)
        
        rle = _convert_pred_to_rle(pred, thresh=thresh, small_thresh=small_thresh)
        rles.append(rle)

    sub_df = pd.DataFrame({'ImageId':im_ids, 'EncodedPixels':rles})
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    return sub_df


def _convert_pred_to_rle(pred, thresh=0.5, small_thresh=2048):
     # pred is already 1024 * 1024 but no binary. its values are 0-1
    pred = np.where(pred>thresh, 1, 0)

    # zero out the smaller regions
    if pred.sum() < small_thresh:
        pred[:] = 0
    # binary -> 0, 255 and transpose for submission format
    pred = (pred.T*255).astype(np.uint8)
    return mask2rle(pred, 1024, 1024)





def _get_best_threshold(model_path, column_name='score'):
    '''
    get the best threshold for the model path. It assume you already run eval_util.eval_all_pred()
    '''

    # compute saved prediction path. should be the same as pred_util._save_preds()
    #'/data/pneumo_log/val_1/'
    base_save_path = '/'.join(model_path.split('/')[:-2])
    #2019_0805_0344
    folder_name = model_path.split('/')[-2]
    # '/data/pneumo_log/val_1/val_predictions/2019_0805_0344/best_weights/'
    # extended for snapshots
    pred_data_dir = base_save_path + '/val_predictions/' + folder_name + '/' + model_path.split('/')[-1].split('.')[0] + '/'
    df = pd.read_csv(pred_data_dir + '/thresh_evaluation.csv')
    # after groupby the index will be thresh so just get index as the thresh value
    best_thresh = df.groupby('thresh').mean()[column_name].idxmax()

    return best_thresh

















