# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 11:11:35 2020

@author: Christof
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU, LeakyReLU
from keras import optimizers, initializers, callbacks
import NNUtility as nnu
import matplotlib.pyplot as plt
import os
import pickle
import math
import itertools as it

# fh32 = open('../Data/Testdata/out32ktest.txt')
# fh = open('../Data/Testdata/singleout')
# out = np.loadtxt(fh)
# top = np.ceil(np.max(out[:, 0]))
# bot = np.floor(np.min(out[:, 0]))
# out[:, 0] = 2 * (out[:, 0] - bot) / (top - bot) - 1

# for i in range(out.shape[1]):
#    print('Mean: ', np.mean(out[:, i]))
#    print('Max: ', np.max(out[:, i]))
#    print('Min: ', np.min(out[:, i]))


print(tempwidth)
