# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:36:54 2019

@author: Christof
"""

import numpy as np
import sobol_seq
import matplotlib.pyplot as plt

random = np.random.rand(1024, 3)
sobol = sobol_seq.i4_sobol_generate(3, 1024)

print(random.shape)
print(sobol.shape)

rand_dist = plt.figure()
plt.title('Example 3D Distribution of Random Points')
plt.subplots_adjust(wspace=0, hspace=0)
for i in range(3):
    plt.subplot(3, 3, i+1)
    plt.plot(random[:, 0], random[:, i], 'b,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
    plt.subplot(3, 3, i+4)
    plt.plot(random[:, 1], random[:, i], 'g,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
    plt.subplot(3, 3, i+7)
    plt.plot(random[:, 2], random[:, i], 'r,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
plt.savefig('Dist_Rand1024_Pix.pdf', format='pdf')
plt.show()

sov_dist = plt.figure()
plt.subplots_adjust(wspace=0, hspace=0)
plt.title('Example 3D Distribution of Sobol Points')
for i in range(3):
    plt.subplot(3, 3, i+1)
    plt.plot(sobol[:, 0], sobol[:, i], 'b,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
    plt.subplot(3, 3, i+4)
    plt.plot(sobol[:, 1], sobol[:, i], 'g,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
    plt.subplot(3, 3, i+7)
    plt.plot(sobol[:, 2], sobol[:, i], 'r,')
    plt.tick_params(axis='both', which='both',
                    bottom=False, top=False, right=False, left=False,
                    labelbottom=False, labeltop=False, labelright= False,
                    labelleft=False)
plt.savefig('Dist_Sobol1024_Pix.pdf', format='pdf')
plt.show()
