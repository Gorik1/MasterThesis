# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:22:35 2020

@author: Christof
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

#Density Profiles

# Set parameters
nped = 0.7
n0 = 1
nsep = 0.5
rped = 0.6
rsep = 1
m = (nsep- nped) / (rsep-rped)
b = nsep - m* rsep
alphan = 1
lambdan = 0.35
x = np.arange(0,1.51,0.01)

# Define x-values
xcore = np.arange(0, rped, 0.001)
xped = np.arange(rped, rsep, 0.001)
xsep = np.arange(rsep, 1.5*rsep, 0.001)

# Calculate y-values
ycore = nped + (n0 - nped) * (1 - (xcore / rped)**2)**alphan
yped = m * xped + b
ysep = nsep * np.exp(-(xsep-rsep)/lambdan)
pmin = nsep * np.exp(-0.35/lambdan)

# Plot graph in parts
fig, ax = plt.subplots()
plt.plot(xcore, ycore, c='k')
plt.plot(xped, yped, c='k')
plt.plot(xsep, ysep, c='k')

# Put in Lines for indicators
plt.hlines(nped, 0, rped)
plt.hlines(nsep, 0, rsep)
plt.hlines(pmin, 0, 1.35*rsep)

plt.vlines(rped, 0, nped)
plt.vlines(rsep, 0, nsep)

# Put in Wall
plt.axvline(1.35*rsep, 0, 1, lw='3', c='k')

ax.fill_between(x, 0, 1, where=x >= 1.35*rsep, facecolor='black', alpha=0.5)

# Set x/y-ticks and labels
plt.xticks([0, rped, rsep, 1.35*rsep], ('0', '$r_{ped}$', '$r_{sep}$', 'Wall'))
plt.yticks([0, pmin, nped, nsep, n0], ('0', '$P_{min}$', '$P_{sep}$', '$P_{ped}$', '$P_{max}$'))
plt.xlabel('Radius')
plt.ylabel('Density / Temperature')

# Add descriptive Text
plt.text(0.15, 0.8, 'Core')
plt.text(0.75, 0.68, 'Pedestal')
plt.text(1.15, 0.45, 'SOL')

# Make Graph stick to border
ax.margins(0)

# Save Plot
plt.savefig('DensityPlot.pdf')
plt.show()
plt.close()