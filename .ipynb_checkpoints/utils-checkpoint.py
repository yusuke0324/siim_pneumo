import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.ndimage import zoom
import scipy
import random
from shutil import copyfile
from skimage import measure, morphology
from decimal import localcontext, Decimal, ROUND_HALF_UP
from heapq import nlargest
from scipy import ndimage
from skimage import morphology

DEFAULT_COLOR = [0, 0, 255]
DEFAULT_HU_MAX = None
DEFAULT_HU_MIN = None
DEFAULT_OVERLAY_ALPHA = 0.3


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


def class_to_color(segmentation, color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation,1)] = color
    return seg_color


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed

def vis_slices(volume_data, seg_data, num=30, cols=6, figsize=[15, 15], transpose=False):
    '''
    image_data and seg_data should be (z, h, w, 1) shape
    '''
    volume_data = np.squeeze(volume_data)
    seg_data = np.squeeze(seg_data)
    gray_volume = hu_to_grayscale(volume_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
    color_seg = class_to_color(seg_data, DEFAULT_COLOR)
    overlayed = overlay(gray_volume, color_seg, seg_data, DEFAULT_OVERLAY_ALPHA)

    total_slices = volume_data.shape[0]
    rows = num // cols + 1
    interval = total_slices / num
    # if the number of total slice is less than num, set 1 to interval
    if interval < 1:
        interval = 1

    fix, ax = plt.subplots(rows, cols, figsize=figsize)

    for i in range(num):
        indx = int(i * interval)
        if indx + 1 > total_slices:
            break
        ax[int(i/cols), int(i%cols)].set_title('slice %d' % indx)
        ax[int(i/cols), int(i%cols)].imshow(overlayed[indx], cmap=
            'gray')
        ax[int(i/cols), int(i%cols)].axis('off')
    plt.show()