# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:31:06 2019

@author: Christof
"""

import classNN.py as CNN

Width = {0: 50, 1: 100, 2: 200, 3: 300}
Depth = {0: 3, 1: 5, 2: 10, 3: 15}
Drop = {0: , 1: , 2: , 3: }


if (Depth[i] * Width[i] > 2**17):
    print('Parameter number exceeds Data Points, please reconsider choice of Network size')
    exit


for CNN.NN().run()