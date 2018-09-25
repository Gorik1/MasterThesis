# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:38:58 2018

@author: ChristofBackhaus
"""

# Analyse Training Data
# Mean, Median, min, max, variance

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import NNUtility as nnu

fh = open('summary_train.txt')
train = np.loadtxt(fh)

label = train[:, -2]
std = train[:, -1]

percent = std / label


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(label, std, bins=10,
                                      range=[[0, 3.5], [0.2, 2.2]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.1, yedges[:-1] + 0.1, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')

plt.show()
