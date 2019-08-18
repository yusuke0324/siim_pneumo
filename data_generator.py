import os
from glob import glob
import numpy as np
import random
from scipy.ndimage import zoom
from tqdm import tqdm
from multiprocessing import pool, cpu_count
import pandas as pd
import threading
from tensorflow.python.keras.utils import to_categorical
from scipy import ndimage
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
import cv2
from copy import deepcopy # this will be deep copy: refer here https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)
# from https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder
# NOTE: in my data generator, this augmentation will be applied BEFORE resize.
# That's why random sized crop should be range 624 - 1024
AUGMENTATIONS = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
         ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
        ], p=0.3),
    RandomSizedCrop(min_max_height=(624, 1024), height=1024, width=1024,p=0.25),
    ToFloat(max_value=1)
],p=1)


#################### Now make the data generator threadsafe ####################
# from https://github.com/keras-team/keras/issues/1638#event-546371414
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


# try to implement generator with Sequence to enable model train with multiprocessing thread safe
# ref:
# https://www.kaggle.com/nikhilroxtomar/generators-for-keras-model
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    
    '''
    Usage
    ----------------------------------------
        train_gen = data_generator.DataGenerator(mode='train')
        val_gen = data_generator.DataGenerator(mode='val')

        i, m = train_gen.__getitem__(0)
    
    Argments
    ----------------------------------------
        data_path: data path that have data set in it. These files are assumed to be made by data_prep.make_fold()
        target_shape: output shape of this generator. 2 rank (h, w).
        seed: random seed
        batch_size: batch size
        train_split: 0-1, ratio of training data

    
        

    '''

    def __init__(self, 
                 data_path='/data/pneumo/fold/[!1]/',
                 target_shape=(144, 144),
                 output_ch=1,
                 img_ch=1,
                 seed=1,
                 batch_size=32,
                 train_split=0.9,
                 aug=AUGMENTATIONS,
                 # mode='train',
                 stratified=True,
                 additional_data_path_list=None, #['/data/pneumo_log/val_1/2019_0815_1742/submission/best_weights//pseudo/*.npy','/data/pneumo_log/val_1/val_predictions/2019_0815_1742/best_weights/pseudo_train_fold/*.npy'],
                 ):

        self.data_path = data_path
        self.target_shape = target_shape
        self.output_ch = output_ch
        self.img_ch = img_ch
        self.seed = seed
        self.batch_size = batch_size
        self.train_split = train_split
        self.stratified = stratified
        self.all_data_paths = glob(data_path+'*.npy')
        self.data_num = len(self.all_data_paths)
        self.val_index = 0
        self.train_paths, self.val_paths = self._split_train_val()
        self.additional_data_path_list=additional_data_path_list
        self._add_data()
        self.aug=aug

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_paths) / self.batch_size))

    # the additional data should be pseudo labeling
    # this should be called AFTER train, val split and just add them to TRAIN
    def _add_data(self):
        if self.additional_data_path_list is None:
            return
        else:
            for path in self.additional_data_path_list:
                self.train_paths = np.concatenate([glob(path+'/*.npy'), self.train_paths], axis=0)
    def _split_train_val(self, stratified=True):

        # shuffle with seed
        random.seed(self.seed)
        random.shuffle(self.all_data_paths)
        
        if self.stratified:
            # stratified based on class (0,1)
            # load csv and get class info for all paths and split them with stratify=class
            csv_path = '/'.join(self.data_path.split('/')[:-2]) + '/meta.csv'
            all_data =pd.read_csv(csv_path)
            # since all_data don't have npy path in fold (only have original path), use image_id instead of path
            train_all_data= pd.DataFrame([[path.split('/')[-1].split('.npy')[0], path] for path in self.all_data_paths], columns=['image_id', 'npy_path'])
            image_id_class_df = train_all_data.merge(all_data, on='image_id')

            paths = image_id_class_df['npy_path'].values
            classes = image_id_class_df['class'].values

            train_paths, val_paths, _, _ = train_test_split(paths, classes, stratify=classes, test_size=1-self.train_split)

        else:
            train_paths = self.all_data_paths[0:int(self.train_split * self.data_num)]
            val_paths = self.all_data_paths[int(self.train_split * self.data_num):self.data_num]

        return train_paths, val_paths

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.data_paths)

    def get_train_val_gens(self):
        # make train gen
        train_gen = deepcopy(self)
        train_gen.mode = 'train'
        train_gen.data_paths = self.train_paths
        # make val gen
        val_gen = deepcopy(self)
        val_gen.mode = 'val'
        val_gen.data_paths = self.val_paths

        return train_gen, val_gen


    def __getitem__(self, index):
        # Select batch
        # what's this
        # if (index+1)*self.batch_size > len(self.data_paths):
        #     self.batch_size = len(self.data_paths) - index*self.batch_size
        paths = self.data_paths[index*self.batch_size:(index+1)*self.batch_size]
        img_list = []
        mask_list = []

        for path in paths:

            data = np.load(path)[()]
            mask = data['mask']
            img = data['img']

            # do augomentation before all other preprocessing!
            if (self.mode=='train') and (self.aug is not None):
                # img, mask = _augmentation(img, mask, flip=self.flip, maxshift=self.maxshift, maxrotate=self.maxrotate, flag2d=True)
                auged = self.aug(image=img, mask=mask)
                img = auged['image']
                mask = auged['mask']

            #[TODO] need to crop img and mask

            img = cv2.resize(img, self.target_shape)
            # keep 0 or 1
            mask = cv2.resize(mask, self.target_shape, interpolation=cv2.INTER_NEAREST)


            img = _normalize(img)

            # expand dim for 1 channel

            if self.img_ch == 1:
                img = np.expand_dims(img, axis=-1)
            elif self.img_ch == 3:
                img = np.stack((img, img, img), axis=-1)
            img_list.append(img)
            mask = np.expand_dims(mask, axis=-1)


            mask_list.append(mask)
            # roi_list.append(roi)
            if len(img_list) == self.batch_size:
                volumes = np.array(img_list)
                masks = np.array(mask_list)

                # for next generation, reset them
                img_list = []
                mask_list = []        

                return volumes, masks


    # @threadsafe_generator
    # def generate(self, train=True):
    #     while True:
    #         if train:
    #             # shuffle train ids to make sure every batch has random ids. If you don't shuffle here, all batch will have the same ids.
    #             random.shuffle(self.train_paths)
    #             paths = self.train_paths
    #         else:
    #             # use index instead of shuffling for val ids to make sure all batches covers all val data.
    #             paths = self.val_paths[self.val_index*self.batch_size:(self.val_index+1)*self.batch_size]
    #             # update val_index for next generate
    #             self.val_index += 1 
    #             # self.val_index += self.batch_size # =+1 is correct, > len(self.val_paths) is wrong
    #             # if self.val_index > len(self.val_paths):
    #             if self.val_index > len(self.val_paths)//self.batch_size:
    #                 self.val_index = 0


    #         img_list = []
    #         mask_list = []

    #         for path in paths:

    #             print(path)

    #             data = np.load(path)[()]
    #             img = data['img']
    #             mask = data['mask']

    #             # do augomentation before all other preprocessing!
    #             if train and self.aug:
    #                 img, mask = _augmentation(img, mask, flip=self.flip, maxshift=self.maxshift, maxrotate=self.maxrotate, flag2d=True)

    #             #[TODO] need to crop img and mask

    #             img = cv2.resize(img, self.target_shape)
    #             # keep 0 or 1
    #             mask = cv2.resize(mask, self.target_shape, interpolation=cv2.INTER_NEAREST)

    #             img = _normalize(img)

    #             # expand dim for 1 channel
    #             img = np.expand_dims(img, axis=-1)
    #             mask = np.expand_dims(mask, axis=-1)

    #             img_list.append(img)
    #             mask_list.append(mask)
    #             # roi_list.append(roi)

    #             if len(img_list) == self.batch_size:
    #                 volumes = np.array(img_list)
    #                 masks = np.array(mask_list)

    #                 # for next generation, reset them
    #                 img_list = []
    #                 mask_list = []        

    #                 yield volumes, masks

def _normalize(image):
    '''
    rerange from 0 to 1
    '''

    min_v = image.min()
    max_v = image.max()

    # image value range will be shifted from min~max to 0~(max-min)
    image = image - min_v
    # image value range will be 0~1
    # there is cases when max_v == min_v because of 3d crop algorithms
    if not max_v == min_v:
        image = image / (max_v - min_v)
        # in case where min and max value is same, always image = image - min_v will be 0
        # so there is nothing to do here.


    return image


# this is old. it's no longer used.
def _augmentation(volume, mask, flip=False, maxshift=[0, 0, 0], maxrotate=0, flag2d=False):
    '''
    augmentation. this can be used for volume and mask 
    The volume shape should be (z, x, y) so it shouldn't be called after transpose
    maxshift and max raote should be 0<int
    set flag2d = True for image
    '''
    if flag2d:
        # in 2d case, cannot shift z axis
        maxshift[0] = 0
        # (x, y) -> (z, x, y)
        volume = np.expand_dims(volume, axis=0)
        mask = np.expand_dims(mask, axis=0)
    # FLIP
    if flip:
        if random.choice([True, False]):
            volume = np.flip(volume, axis=2)
            mask = np.flip(mask, axis=2)

    cval = np.min(volume)
    #Shift
    shift0 = random.randint(-maxshift[0], maxshift[0])
    shift1 = random.randint(-maxshift[1], maxshift[1])
    shift2 = random.randint(-maxshift[2], maxshift[2])

    shift = [shift0, shift1, shift2]
    volume = ndimage.shift(volume, shift, order=0, cval=cval)
    mask = ndimage.shift(mask, shift, order=0)

    # Rotate
    angle= random.randint(-maxrotate, maxrotate)
    volume = ndimage.rotate(volume,angle, axes=(1, 2), order=0, reshape=False, cval=cval)
    mask = ndimage.rotate(mask,angle, axes=(1, 2), order=0, reshape=False)

    if flag2d:
        volume = np.squeeze(volume)        
        mask = np.squeeze(mask)        

    return volume, mask

