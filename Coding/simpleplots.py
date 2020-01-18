# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:21:52 2020

@author: Christof
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

#Simple plots for ReLU and ELU

path = 'C:/MasterThesis/Output/'

#ReLU
filename = 'ReLU.png' 
label = 'Rectified Linear Unit'

x_ReluNeg = np.arange(-1,0,0.01)
x_ReluPos = np.arange(0,1.01,0.01)
y_ReluNeg = np.zeros((100))
y_ReluPos = x_ReluPos
x_Relu = np.concatenate((x_ReluNeg, x_ReluPos))
y_Relu = np.concatenate((y_ReluNeg, y_ReluPos))

FigReLu = plt.figure()
plt.plot(x_Relu, y_Relu)
plt.xticks(np.arange(-1,1.01,0.5))
plt.yticks(np.arange(0,1.01,0.5))
plt.savefig('ReLU.pdf')
plt.close()

#Elu
x_EluNeg = np.arange(-4,0,0.01)
x_EluPos = np.arange(0,4.01,0.01)
y_EluNeg = 1 * (np.exp(x_EluNeg) - 1)
y_EluPos = x_EluPos 
x_Elu = np.concatenate((x_EluNeg, x_EluPos))
y_Elu = np.concatenate((y_EluNeg, y_EluPos))

FigElu = plt.figure()
plt.grid()
plt.plot(x_Elu, -1 * np.ones(y_Elu.shape), 'k--', label='$\\alpha$')
plt.plot(x_Elu, y_Elu, label='Exponential linear unit')
plt.xticks(np.arange(-4,4.01,2))
plt.yticks([-1, 0, 1, 2, 3, 4],[-1, 0, 1, 2, 3, 4])
plt.legend()
plt.savefig('Elu.pdf')
