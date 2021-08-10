import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score


def visualize(seg_map, img):
    """
    Overlay the segmentation map onto the input color image img
    :param seg_map: segmentation map of size H x W
    :param img:     color image of size H x W x 3
    :return:        image overlaid the color segmentation map
    """
    # Generate the segmentation map in the RGB color with the color code
    # Class 0: black; Class 1: Green; Class 2: Red
    COLOR_CODE = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    segmap_rgb = np.zeros(img.shape)
    for k in np.unique(seg_map):
        segmap_rgb[seg_map == k] = COLOR_CODE[k]
    segmap_rgb = (segmap_rgb * 255).astype('uint8')

    # Super-impose the color segmentation map onto the color image
    overlaid_img = cv2.addWeighted(img, 1, segmap_rgb, 0.9, 0)

    return overlaid_img


def accuracy(seg_map, gt):
    """
    Calculate pixel accuracy
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: pixel accuracy
    """
    return np.mean(seg_map == gt)


def iou(seg_map, gt):  # inputs are Numpy arrays
    """
    Calculate mean IoU (a.k.a., Jaccard Index) of an individual segmentation map. Note that, for the whole dataset, we
    must take average the mIoUs
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: mIoU
    """
    # 'micro': Calculate metrics globally by counting the total TP, FN, and FP
    # 'macro': calculate metrics for each label, and find their unweighted mean.
    return jaccard_score(gt.flatten(), seg_map.flatten(), average='macro')


def f1_score(seg_map, gt):
    """
    Calculate F-measure (a.k.a., Dice Coefficient, F1-score)
    :param seg_map: segmentation map as a 2D array of size [H, W]
    :param gt: ground-truth as a 2D array of size [H, W]
    :return: F-measure
    """
    return f1_score(gt.flatten(), seg_map.flatten(), average="macro")
