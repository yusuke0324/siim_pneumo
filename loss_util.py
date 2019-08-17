import numpy as np
import keras.backend as K
from keras.losses import binary_crossentropy # what is the difference with K.binary_crossentropy
import tensorflow as tf

# ref https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/101429#latest-592529
def lovasz_loss(truth, logit, margin=[1,5]):

    def compute_lovasz_gradient(truth): #sorted
        truth_sum    = K.sum(truth)
        intersection = truth_sum - K.cumsum(truth, 0)
        union        = truth_sum + K.cumsum(1 - truth, 0)
        jaccard      = 1. - intersection / union
        jaccard      = K.concatenate([jaccard[0:1], jaccard[1:] - jaccard[:-1]], axis=0)

        gradient     = jaccard
        return gradient

    def lovasz_hinge_one(truth , logit):

        m = tf.where(K.equal(truth, 1), margin[1] * K.ones_like(truth), margin[0] * K.ones_like(truth))

        truth = K.cast(truth, dtype = logit.dtype)
        sign  = 2. * truth - 1.
        hinge = (m - logit * K.stop_gradient(sign))
        hinge, permutation = tf.nn.top_k(hinge, k=K.shape(hinge)[0])
        hinge = K.relu(hinge)

        truth = K.gather(truth, permutation)
        gradient = compute_lovasz_gradient(truth)

        loss = K.dot(K.expand_dims(hinge, 0), K.stop_gradient(K.expand_dims(gradient, -1)))
        
        return loss

    #----
    batch_size = K.shape(logit)[0]
    loss = K.map_fn(lambda x: lovasz_hinge_one(truth[x], logit[x]), K.arange(batch_size), dtype='float32')

    return loss

def criterion_pixel(truth_pixel, logit_pixel):
    batch_size = K.shape(logit_pixel)[0]
    logit = K.reshape(logit_pixel, (batch_size,-1))
    truth = K.reshape(truth_pixel, (batch_size,-1))

    loss = lovasz_loss(truth, logit)  

    loss = K.mean(loss)
    return loss


# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
def get_dice_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  denominator is never empty 
        # hence it is safe to divide by its number of pixels
        numerataor = 2. * np.sum(t * p)
        denominator = true + pred
        dice = numerataor / denominator
        
        # why...?
        # dice metrric is a stepwise approximation of the real iou over 0.5
        # dice = np.floor(max(0, (dice - 0.45)*20)) / 10
        
        metric += dice
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_dice_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_dice_vector, [label, pred > 0.5], tf.float64)


def dice_coef_flat(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss_flat(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss_flat(y_true, y_pred)

def create_dice_coef(smooth=1., axis=[1, 2, 3]):
    '''
    simple dice for 2 class segmentation
    since this is for single channel, axis=[1, 2, 3] is for 2d, axis=[1, 2, 3, 4] is for 3d
    '''
    def dice_coef(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
        denom = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
        return K.mean( (2. * intersection + smooth) / (denom + smooth), axis=0)
    return dice_coef


def create_dice_loss(smooth=1, axis=[1, 2, 3]):
    '''
    simple dice loss for 2 class segmentation
    1-dice
    '''
    def dice_loss(y_true, y_pred):
        dice_coef = create_dice_coef(smooth=smooth, axis=axis)
        return 1.0 - dice_coef(y_true, y_pred)
    return dice_loss


def create_weighted_binary_crossentropy(zero_weight=0.1, one_weight=0.9):
    # https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
    # if positive is small, set one_weights > zero_weights to penalize false negative
    # in non categorical case, need to calculate (1-y_true) coeficient too
    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return weighted_binary_crossentropy

def create_categorical_focal_loss(gamma=2., weights=[1., 1.]):

    def categorical_focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = -K.mean(weights * y_true * K.pow(1. - y_pred, gamma) * K.log(y_pred))

        return loss
    return categorical_focal_loss