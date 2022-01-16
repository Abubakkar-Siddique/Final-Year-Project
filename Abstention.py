#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 12:18:23 2021

@author: Abubakkar Siddique
"""
""" Discription: Dr. defined ROIs of dataset1 with PCA features on abstention SVM using 5 fold cross validation

"""


""" Following function is used to create examples """




from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.decomposition import *
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from classifiers import *
from example import Example
import glob
import numpy as np
import pandas as pd
import time
import matplotlib.image as mpimg
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
def create_Examples(image_array):

    data = []
    ex = Example()
    ex.features_w = image_array[0]
    ex.features_u = image_array[0]
    ex.raw_features = image_array[0]
    data.append(ex)

    return data


def create_Examples1(image_array, y):

    data = []
    num_pos = float(np.sum([y > 0]))
    num_neg = float(np.sum([y < 0]))
    for j in range(len(y)):
        ex = Example()
        ex.features_w = image_array[j]
        ex.features_u = image_array[j]
        ex.raw_features = image_array[j]
        ex.label = y[j]
        if ex.label > 0:
            ex.gamma = 1.0/num_pos
        else:
            ex.gamma = 1.0/num_neg
        data.append(ex)
    return data


lu = 1e-07
lw = 1e-06
c2 = 0.06


def reject(image):

    train_ind = [0, 1,   2,   3,  4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  30,  31,  32, 33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50, 51,  52,  53,  54, 55,  56,  57,  58,  59,  60,  61,
                 62,  63,  64,  65,  66,  67,  68, 69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80, 81,  82,  83,  93,  94,  95, 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118]

    df_read = pd.read_csv("./Dataset1_Normal_Abnormal/Dataset_doctor_NA.csv")
    image_list = []
    y = []

    for i in np.array(df_read):
        image_list.append(i[:-1])
        y.append(i[-1])

    image_array = np.array(image_list)
    y = np.array(y, dtype=int)

    classifier = linclass_rej(epochs=600, Lambda_u=lu,
                              Lambda_w=lw, alpha=1.0, c=c2)
    Etr = create_Examples1(image_array[train_ind], y[train_ind])
    classifier.train3(Etr)

    rej_score = classifier.reject(image)

    rej = rej_score > 0

    return rej
