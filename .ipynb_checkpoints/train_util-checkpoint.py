import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from datetime import datetime as dt
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler, Callback
from keras.models import Model
from keras.models import load_model

from keras.callbacks import LambdaCallback
from operator import itemgetter
from skimage.transform import resize
import cv2
from glob import glob
from keras import backend as K
from keras.losses import binary_crossentropy
import multiprocessing
import keras
from time import sleep
import matplotlib.pyplot as plt

import tensorflow as tf


# fix a problem :Unable to create file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')
#https://github.com/keras-team/keras/issues/11101
class PatchedModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(PatchedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        
                        saved_correctly = False
                        while not saved_correctly:
                            try:
                                if self.save_weights_only:
                                    self.model.save_weights(filepath, overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                                saved_correctly = True
                            except Exception as error:
                                print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                                sleep(5)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                saved_correctly = False
                while not saved_correctly:
                    try:
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                        saved_correctly = True
                    except Exception as error:
                        print('Error while trying to save the model: {}.\nTrying again...'.format(error))
                        sleep(5)


# https://www.kaggle.com/meaninglesslives/unet-plus-plus-with-efficientnet-encoder
# cosine lr schedule
class SnapshotScheduleBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def _cosine_anneal_schedule(self, t):
            cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
            cos_inner /= self.T // self.M
            cos_out = np.cos(cos_inner) + 1
            return float(self.alpha_zero / 2 * cos_out)

    def get_scheduler(self):
        # put thie return into callbacks
        return LearningRateScheduler(schedule=self._cosine_anneal_schedule)



def _epochOutput(epoch, logs):

    print("Finished epoch: " + str(epoch))
    print(logs)

    # if os.listdir(dirname)

def _delete_oldest_weightfile(dirname):
    weight_files = []
    for file in os.listdir(dirname):
        base, ext = os.path.splitext(file)
        if ext == 'hdf5':
            weight_files.append([file, os.path.getctime(file)])

    weight_files.sort(key=itemgetter(1), reverse=True)
    os.remove(weight_files[-1][0])

def _get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y_%m%d_%H%M')
    return tstr

def _make_dir(dir_name):
    if not os.path.exists('dir_name'):
        os.makedirs(dir_name)

def train(model, train_gen, val_gen, steps_per_epoch=None, optimizer='adam', log_dir='./log', epochs=100, loss='binary_crossentropy', metrics=['accuracy'], lr_mode='reduce', reduce_lr_factor=0.2, reduce_lr_patience=10, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_gen)

    sub_dir = _get_date_str()
    log_dir = log_dir + '/' + sub_dir
    # make log dir
    _make_dir(log_dir)
    # saved model path
    fpath = log_dir + '/weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # callback
    tb_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    cp_cb = PatchedModelCheckpoint(filepath=log_dir+'/best_weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    batchLogCallback = LambdaCallback(on_epoch_end=_epochOutput)
    # csv_logger = CSVLogger(log_dir + '/training.log')
    csv_logger = AllLogger(log_dir + '/training.log')
    callbacks = [batchLogCallback, csv_logger, cp_cb]
    if lr_mode == 'reduce':
        callbacks.append(ReduceLROnPlateau(factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=1))
    elif lr_mode == 'cosine':
        schedule_builder = SnapshotScheduleBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=1e-3)
        callbacks.append(schedule_builder.get_scheduler())


    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print(model.summary())
    model.fit_generator(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        validation_data=val_gen,
        workers=multiprocessing.cpu_count() - 2, #https://github.com/keras-team/keras/issues/11101
        validation_steps=validation_steps,
        callbacks=callbacks,
        use_multiprocessing=True # Don't use multiprocessing https://keunwoochoi.wordpress.com/2017/08/24/tip-fit_generator-in-keras-how-to-parallelise-correctly/
        )

    model.save(log_dir + '/' + str(epochs) + 'epochs_final_save')

class AllLogger(keras.callbacks.Callback):

    def __init__(self, log_file_path):
        super(AllLogger, self).__init__()
        self.log_file_path = log_file_path

    def on_train_begin(self, logs={}):
        self.logs_list = []
        self.epochs = []
        # leraning rates
        self.lrs = []
    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch)
        # dictionary is mutable and keras is going to modify 'logs' over this training.
        # so copy logs and then append it to the list.
        self.logs_list.append(logs.copy())
        self.lrs.append(K.eval(self.model.optimizer.lr))
        self.save_logs()

    def save_logs(self):
        log_df = pd.DataFrame(self.logs_list)
        epoch_lrs_df = pd.DataFrame({'epoch':self.epochs, 'learning_rate':self.lrs})
        all_log_df = epoch_lrs_df.merge(log_df, left_index=True, right_index=True)
        all_log_df.to_csv(self.log_file_path)

# not used yet
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')

    
