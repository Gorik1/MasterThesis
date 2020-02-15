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

history = pickle.load(open('../Coding/equal131072relu1/history', "rb"))

loss = history['loss']
val_loss = history['val_loss']

Epochs = len(history['loss'])
x = np.arange(0, Epochs, 1)

FigReLu = plt.figure()
plt.plot(x, loss, label='Loss')
plt.text(round(Epochs-5, -1), loss[-1]+loss[-1]/10, '%.3f' % loss[-1])
plt.plot(x, val_loss, label='Validation Loss')
plt.text(round(Epochs-5, -1), val_loss[-1]+val_loss[-1]/10, '%.3f' % val_loss[-1])
plt.title('Test Title')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.xticks(np.arange(-1,1.01,0.5))
# plt.yticks(np.arange(0,1.01,0.5))
# plt.savefig('Test.pdf')
# plt.close()
