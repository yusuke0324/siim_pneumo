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
import cv2


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
                 aug=True,
                 flip=False,
                 maxshift=[0, 20, 20],
                 maxrotate=10,
                 mode='train'

                 ):

        self.data_path = data_path
        self.target_shape = target_shape
        self.output_ch = output_ch
        self.img_ch = img_ch
        self.seed = seed
        self.batch_size = batch_size
        self.train_split = train_split

        self.mode = mode
        self.all_data_paths = glob(data_path+'*.npy')
        self.data_num = len(self.all_data_paths)
        self.val_index = 0
        self.train_paths, self.val_paths = self._split_train_val()
        if self.mode == 'train':
            self.data_paths = self.train_paths
        elif self.mode == 'val':
            self.data_paths = self.val_paths
        self.aug = aug
        self.flip = flip
        self.maxshift = maxshift
        self.maxrotate = maxrotate

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_paths) / self.batch_size))

    
    def _split_train_val(self):

        # shuffle with seed
        random.seed(self.seed)
        random.shuffle(self.all_data_paths)
        
        train_paths = self.all_data_paths[0:int(self.train_split * self.data_num)]
        val_paths = self.all_data_paths[int(self.train_split * self.data_num):self.data_num]

        return train_paths, val_paths

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.data_paths)


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
            img = data['img']
            mask = data['mask']

            # do augomentation before all other preprocessing!
            if self.mode=='train' and self.aug:
                img, mask = _augmentation(img, mask, flip=self.flip, maxshift=self.maxshift, maxrotate=self.maxrotate, flag2d=True)

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
            mask = np.expand_dims(mask, axis=-1)

            img_list.append(img)
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

