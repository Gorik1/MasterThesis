# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:45:14 2020

@author: Christof
"""

import numpy as np
import sobol_seq
import matplotlib.pyplot as plt
import scipy

random = np.random.rand(512, 2)
sobol = sobol_seq.i4_sobol_generate(2, 512)

print(random.shape)
print(sobol.shape)

rand_dist = plt.figure()
plt.title('Example 2D Distribution of Random Points')
plt.plot(random[:, 0], random[:, 1], 'bo', mfc='none')
plt.savefig('2D_Dist_Rand512.pdf', format='pdf')
plt.show()

sov_dist = plt.figure()
plt.title('Example 2D Distribution of Sobol Points')
plt.plot(sobol[:, 0], sobol[:, 1], 'bo', mfc='none')
plt.savefig('2D_Dist_Sobol512.pdf', format='pdf')
plt.show()