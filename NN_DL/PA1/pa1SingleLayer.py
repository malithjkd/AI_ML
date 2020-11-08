"""
MalithJKD
Logistric Regression with a Neural Networks Mindset
07.10.2020
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline

# load the dateset 
train_set_x_orig, train_set_y, test_set_x, test_set_y, classes = load_dataset()

# to see the data
index = 2
plt.imshow(train_set_x_orig[index])
#print("y = " + str(train))