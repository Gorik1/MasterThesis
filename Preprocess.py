# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:15:56 2018

@author: ChristofBackhaus
"""

# Load data
# form normal distribution
# draw samples from distribution

import numpy as np
trainfh = open('./summary_train.txt')
traindata = np.loadtxt(trainfh)

for element in (traindata[:, -2].size):
    norm_dist[element] = np.random.normal(loc=traindata[element, -2],
                                 scale=traindata[element, -1], size=10)
    