# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:27:26 2018

@author: Christof
"""

import numpy as np
import sobol_seq


class GenerateData(object):
    def __init__(self, HeadPath):
        # headfh = open(HeadPath)

        self.TiTe = True
        self.ConstCol = np.array([[4, 7, 8, 9, 12],
                                  [4, 7, 8, 9, 12],
                                  [4, 7, 8, 9, 12]])
        self.TotNumPar = 36
        self.Bounds = [-1, 1]
        self.NumSob = 1000
        self.NumRan = 0.5 * self.NumSob
        self.ProfileSize = np.array([12, 12, 12])
        return

    def CheckTiTe(self):
        self.NumPar = self.TotNumPar
        if self.TiTe:
            self.NumPar = self.NumPar - (self.ConstCol[0].size +
                                         self.ProfileSize[1] + self.ConstCol[2].size)
        else:
            for profile in self.ConstCol:
                self.NumPar -= profile.size
        return

    def Generate(self):
        self.SobolPoints = sobol_seq.i4_sobol_generate(self.NumPar, self.NumSob)
        self.TestPoints = np.random.rand(self.NumRan, self.NumPar)
        return

    def ScaleToInterval(self, data):
        return data * (self.Bounds[1] - self.Bounds[0]) + self.Bound[0]

    def InsertConstCol(self, Data):
        # prepare output array
        Full = np.zeros([self.NumSob, self.TotNumPar])
        # prepare indexing of the constants column
        PrevProfSize = 0
        for i in range(self.ConstCol[:, 0].size):
            PrevProfSize += self.ProfileSize[i]
            self.ConstCol[i, :] = self.ConstCol[i, :] + i * PrevProfSize - 1
        # insert Data and const columns in correct order
        k = 0
        for j in range(self.TotNumPar):
            if j in self.ConstCol:
                k += 1
            else:
                Full[:, j] = Data[:, j-k]
        return Full

    def Save(self, data, path='./', name='summary'):
        fname = 
        np.savetxt()
        