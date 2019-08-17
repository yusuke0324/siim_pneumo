import numpy as np
import pandas as pd
import pydicom
from glob import glob
from tqdm import tqdm
from mask_functions import mask2rle
import pred_util

def make_submission(model_path, thresh, small_thresh=2048, test_base_path='/data/pneumo/dicom-images-test/'):

    '''
    save submission file under a folder that model file is saved.
    '''
    save_dir = '/'.join(model_path.split('/')[:-1])
    file_name = 'submission_' + save_dir.split('/')[-1] + '.csv'
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
        pred = np.where(pred>thresh, 1, 0)

        # zero out the smaller regions
        if pred.sum() < small_thresh:
            pred[:] = 0
        # binary -> 0, 255 and transpose for submission format
        pred = (pred.T*255).astype(np.uint8)

        rles.append(mask2rle(pred, 1024, 1024))

    sub_df = pd.DataFrame({'ImageId':im_ids, 'EncodedPixels':rles})
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    sub_df.to_csv(save_dir + '/' + file_name, index=False)
