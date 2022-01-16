# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 15:31:36 2021

@author: Abubakkar Siddique
"""
import numpy as np

import matplotlib.image as mpimg
import pywt
import pywt.data


features = []
WPTImages = []


def computeStatistics(arr):
    image = arr[0]
    minImg = np.min(image)
    maxImg = np.max(image)
    if(minImg == 0):
        minImg = 1
    ratio = maxImg / minImg
    features.append(round(ratio, 2))

    for i in range(1, 21):
        image = arr[i]
        size = image.size
        image = np.reshape(image, size)
        image.sort()
        median = np.median(image)
        std = np.std(image)
        half = size/2
        num = int(half)
        Q1 = np.median(image[:num])
        Q3 = np.median(image[num:])
        IQR = Q3 - Q1
        features.extend([round(median, 2), round(std, 2), round(IQR, 2)])


def segmentation(arr):
    coeffs2 = pywt.dwt2(arr, 'db3', mode='periodization')
    LL, (LH, HL, HH) = coeffs2
    WPTImages.extend([LL, LH, HL, HH])


def featureExtraction(original):
    features.clear()
    WPTImages.clear()
    WPTImages.append(original)
    segmentation(original)
    segmentation(WPTImages[1])
    segmentation(WPTImages[2])
    segmentation(WPTImages[3])
    segmentation(WPTImages[4])
    computeStatistics(WPTImages)


def normal_vs_abnormal():
    return np.array(features)


def Fatty_vs_cirrhotic():
    index_array = [0, 5, 6, 11, 12, 25, 26, 27, 32, 33, 50, 51, 56, 57]
    tempArray = []
    for index in index_array:
        tempArray.append(features[index])
    return np.array(tempArray)
