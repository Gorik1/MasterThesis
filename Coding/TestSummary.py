# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:11:46 2018

@author: ChristofBackhaus
"""
import numpy as np

fhtest32 = open('../Data/TestData/testpar32k.par')
fhtest64 = open('../Data/TestData/testpar64k.par')
fhtestres = open('../Data/TestData/singleout')
fhsave = open('../Data/TestData/TestSummary.txt', mode='w')

data32 = np.loadtxt(fhtest32)
data32_no_0 = np.delete(data32, [3, 6, 7, 8, 11, 15, 18, 19, 20, 23, 27, 30, 31, 32, 35], axis=1)
data32_TI_EQ_TE = np.delete(data32_no_0, [7, 8, 9, 10, 11, 12, 13], axis=1)
print(data32_no_0.shape)
print(data32_TI_EQ_TE.shape)

data64 = np.loadtxt(fhtest64)
data64_no_0 = np.delete(data64, [3, 6, 7, 8, 11, 15, 18, 19, 20, 23, 27, 30, 31, 32, 35], axis=1)
data64_TI_EQ_TE = np.delete(data64_no_0, [7, 8, 9, 10, 11, 12, 13], axis=1)
print(data64_no_0.shape)
print(data64_TI_EQ_TE.shape)

total = np.concatenate((data32_TI_EQ_TE, data64_TI_EQ_TE), axis=0)
print(total.shape)

res = np.loadtxt(fhtestres)
summary = np.concatenate((total[:res.shape[0], :], res), axis=1)
time = np.ones((summary.shape[0], 1))
summary = np.concatenate((summary, time), axis=1)
print(summary.shape)

np.savetxt(fhsave, summary)
