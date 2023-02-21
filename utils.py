import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform
from skimage import img_as_ubyte
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import time
import datetime

# def crop_border(image_pth):
#     im = skio.imread(image_pth)
#     height = im.shape[0]
#     width = im.shape[1]
#     # def get_ratio(height, width, dim = 0):

#     horizontal_val_lst = np.sum((255-im),axis= 1)

    
    
        # max_args = np.argsort(horizontal_val_lst)
        # ratio = min((max_args[-1], height - max_args[-1])) / height

        # return ratio
    # left_ ratio


    """对mask区域进行计算，在4倍mask大小的图片上进行inpainting."""



