# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:19:37 2019

@author: Christof
"""

import numpy as np

die_range = np.arange(1, 21)

i = 0
total = 0
while i < 10000:
    i += 1
    rolls = np.random.choice(die_range, size=2, replace=True, p=None)
    total += sum(rolls) - max(rolls)
    
average = total / 10000
print(average)