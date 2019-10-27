# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 01:11:46 2018

@author: ChristofBackhaus
"""
import numpy as np

fhpar1 = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/scaledtestpar32k.par')
fhpar2 = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/scaledtestpar64k.par')
# fhout1 = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/out32ktest.txt')
fhout = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/singleout')
#fhtime1 = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/timeout32ktest.txt')
#fhtime2 = open('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/timeout64ktest.txt')

#out1 = np.loadtxt(fhout1)
out = np.loadtxt(fhout)
#time1 = np.loadtxt(fhtime1)
#time2 = np.loadtxt(fhtime2)
par1 = np.loadtxt(fhpar1)
par2 = np.loadtxt(fhpar2)
#par1 = out[:,:14]
#par2 = out2[:,:14]

#totalout = np.concatenate((out1, out2), axis=0)
#totaltime = np.concatenate((time1, time2), axis=0)
totalpar = np.concatenate((par1, par2), axis=0)
print(totalpar.shape)
length = out.shape[0]
beginning = totalpar[:length,:]
print(beginning.shape)
difference = totalpar.shape[0] - length
end = totalpar[difference:,:]
print(end.shape)
#print(totalout.shape)
#print(totaltime.shape)
#print(totalpar.shape)

#totaltime = np.reshape(totaltime, (65536, 1))

#total = np.concatenate((totalpar, totalout, totaltime), axis=1)
totalbeginning = np.concatenate((beginning, out), axis=1)
totalend = np.concatenate((end, out), axis=1)

np.savetxt('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/TestSummaryBeg.txt', beginning)
np.savetxt('C:/Users/Christof/Documents/GitHub/MasterThesis/Data/TestData/TestSummaryEnd.txt', end)