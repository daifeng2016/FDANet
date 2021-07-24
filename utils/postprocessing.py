import multiprocessing as mp

import numpy as np
from skimage.transform import resize
from skimage.morphology import erosion, dilation, rectangle
from tqdm import tqdm
# from pydensecrf.densecrf import DenseCRF2D
# from pydensecrf.utils import unary_from_softmax
# import pydensecrf.densecrf as dcrf
# from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
import time
#from pycocotools import mask as cocomask
#import pandas as pd
import cv2
import skimage.io
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_holes



def post_proc(img, min_size=2000, area=10):
    #t0 = time.time()
    #print('Remove small holes: %d.' % min_size)
    img=dilate_image(img,dilate_selem_size=2)#connnect small hole between pixels 2 is bettern than 5, which will overfill the hole

    ind = remove_small_holes(label(img), area_threshold=min_size, connectivity=img.ndim)

    #print('Remove samll region: %d.' % area)
    img = ind.astype(np.uint8)
    lab_arr = label(img)
    lab_atr = regionprops(lab_arr)

    def fun(atr):
        if atr.area <= area:
            min_row, min_col, max_row, max_col = atr.bbox
            t = lab_arr[min_row:max_row, min_col:max_col]
            t[t == atr.label] = 0

    list(map(fun, lab_atr))
    ind = lab_arr > 0
    view = np.zeros(img.shape, np.uint8)
    view[ind] = 255
    return view

def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (C x H x W).

    """
    n_channels = image.shape[0]
    resized_image = resize(image, (n_channels,) + target_size, mode='constant')
    return resized_image

def categorize_image(image):
    """Maps probability map to categories. Each pixel is assigned with a category with highest probability.

    Args:
        image (numpy.ndarray): Probability map of shape (C x H x W).

    Returns:
        numpy.ndarray: Categorized image of shape (H x W).

    """
    return np.argmax(image, axis=0)

def erode_image(mask, erode_selem_size):
    """Erode mask.

    Args:
        mask (numpy.ndarray): Mask of shape (H x W) or multiple masks of shape (C x H x W).
        erode_selem_size (int): Size of rectangle structuring element used for erosion.

    Returns:
        numpy.ndarray: Eroded mask of shape (H x W) or multiple masks of shape (C x H x W).

    """
    if not erode_selem_size > 0:
        return mask
    selem = rectangle(erode_selem_size, erode_selem_size)
    if mask.ndim == 2:
        eroded_image = erosion(mask, selem=selem)
    else:
        eroded_image = []
        for category_mask in mask:
            eroded_image.append(erosion(category_mask, selem=selem))
            eroded_image = np.stack(eroded_image)
    return eroded_image


def dilate_image(mask, dilate_selem_size):
    """Dilate mask.

    Args:
        mask (numpy.ndarray): Mask of shape (H x W) or multiple masks of shape (C x H x W).
        dilate_selem_size (int): Size of rectangle structuring element used for dilation.

    Returns:
        numpy.ndarray: dilated Mask of shape (H x W) or multiple masks of shape (C x H x W).

    """
    if not dilate_selem_size > 0:
        return mask
    selem = rectangle(dilate_selem_size, dilate_selem_size)
    if mask.ndim == 2:
        dilated_image = dilation(mask, selem=selem)
    else:
        dilated_image = []
        for category_mask in mask:
            dilated_image.append(dilation(category_mask, selem=selem))
        dilated_image = np.stack(dilated_image)
    return dilated_image

def model_ensemble(res1,res2):
    assert res1.shape==res2.shape,"res1 and res2 must be same"
    res1=res1.reshape(res1.shape[0],res1.shape[1])
    res2 = res2.reshape(res2.shape[0], res2.shape[1])
    height,width=res1.shape
    res=np.zeros(res1.shape)
    for i in range(height):
        for j in range (width):
            if res1[i,j]>0 or res2[i,j]>0:
                res[i,j]=255.0

    return res


def dense_crf(img, output_probs, compat_gaussian=3, sxy_gaussian=1,
              compat_bilateral=10, sxy_bilateral=1, srgb=50, iterations=5):
    """Perform fully connected CRF.

    This function performs CRF method described in the following paper:

        Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
        Philipp Krähenbühl and Vladlen Koltun
        NIPS 2011
        https://arxiv.org/abs/1210.5644

    Args:
        img (numpy.ndarray): RGB image of shape (3 x H x W).
        output_probs (numpy.ndarray): Probability map of shape (C x H x W).
        compat_gaussian: Compat value for Gaussian case.
        sxy_gaussian: x/y standard-deviation, theta_gamma from the CRF paper.
        compat_bilateral: Compat value for RGB case.
        sxy_bilateral: x/y standard-deviation, theta_alpha from the CRF paper.
        srgb: RGB standard-deviation, theta_beta from the CRF paper.
        iterations: Number of CRF iterations.

    Returns:
        numpy.ndarray: Probability map of shape (C x H x W) after applying CRF.

    """
    height = output_probs.shape[1]
    width = output_probs.shape[2]

    crf = DenseCRF2D(width, height, 1)
    unary = unary_from_softmax(output_probs)
    org_img = img
    org_img = org_img.transpose(1, 2, 0)
    org_img = np.ascontiguousarray(org_img, dtype=np.uint8)

    crf.setUnaryEnergy(unary)

    crf.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    crf.addPairwiseBilateral(sxy=sxy_bilateral, srgb=srgb, rgbim=org_img, compat=compat_bilateral)

    crf_image = crf.inference(iterations)
    crf_image = np.array(crf_image).reshape(output_probs.shape)

    return crf_image

def crf(img, prob):
    '''
    input:
      img: numpy array of shape (num of channels, height, width)
      prob: numpy array of shape (1, height, width), neural network last layer sigmoid output for img

    output:
      res: (1, height, width)

    Modified from:
      http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
      https://github.com/yt605155624/tensorflow-deeplab-resnet/blob/e81482d7bb1ae674f07eae32b0953fe09ff1c9d1/inference_crf.py
    '''
    #func_start = time.time()

    img = np.swapaxes(img, 0, 2)
    # img.shape: (width, height, num of channels)

    num_iter = 5

    prob = np.swapaxes(prob, 1, 2)  # shape: (1, width, height)

    # preprocess prob to (num_classes, width, height) since we have 2 classes: car and background.
    num_classes = 2
    probs = np.tile(prob, (num_classes, 1, 1))  # shape: (2, width, height) tile函数的主要功能就是将一个数组重复一定次数形成一个新的数组
    probs[0] = np.subtract(1, prob) # class 0 is background
    probs[1] = prob                 # class 1 is car

    d = dcrf.DenseCRF(img.shape[0] * img.shape[1], num_classes)

    unary = unary_from_softmax(probs)  # shape: (num_classes, width * height)
    unary = np.ascontiguousarray(unary)
    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Note that this potential is not dependent on the image itself.

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(255, 255, 255),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)


    Q = d.inference(num_iter)  # set the number of iterations
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))
    # res.shape: (width, height)

    res = np.swapaxes(res, 0, 1)  # res.shape:    (height, width)
    res = res[np.newaxis, :, :]   # res.shape: (1, height, width)

    #func_end = time.time()
    # print('{:.2f} sec spent on CRF with {} iterations'.format(func_end - func_start, num_iter))
    # about 2 sec for a 1280 * 960 image with 5 iterations
    return res







