# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:13:55 2020

@author: Christof
"""

import numpy as np

np.random.seed(220220)
amount = 131072

full = np.zeros((amount, 36))
test = np.random.uniform(low=-1, high=1, size=(amount, 14))
full[:, [0, 1, 2, 4, 5, 9, 10, 24, 25, 26, 28, 29, 33, 34]] = test
full[:, [12, 13, 14, 16, 17, 21, 22]] = test[:, :7]


def save_data(data, name, partition, amount):
    # fdata = open(name, 'w')
    fdata = '{}{}'.format(name, '.txt')
    np.savetxt(fdata, data)
    size = np.int(amount/partition)

    for i in range(partition):
        fdatai = '{}{}{}'.format(name, i, '.txt')
        np.savetxt(fdatai, data[size*i:size*(i+1), :])


save_data(test, 'test', 8, amount)
save_data(full, 'full', 8, amount)
