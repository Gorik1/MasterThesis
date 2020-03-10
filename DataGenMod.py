# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:27:26 2018

@author: Christof
"""

import numpy as np
import sobol_seq


class GenerateData(object):
    def __init__(self, Mode='Test', Amount=2**17, Bounds=[-1, 1], seed=220220, TiTe=True):
        self.Bounds = Bounds
        self.TiTe = True
        self.mode = Mode
        self.amount = Amount

        # Hardcoded profiles for now
        self.ProfileSize = np.array([12, 12, 12])
        self.TotNumPar = np.sum(self.ProfileSize)
        # FIXME If profile constant columns do not share the same dimension this will be an array
        # of lists and break checktite
        self.ConstCol = np.array([[3, 6, 7, 8, 11],
                                  [3, 6, 7, 8, 11],
                                  [3, 6, 7, 8, 11]])
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
        if self.mode == 'Test':
            data = np.random.rand(self.amount, self.NumPar)
        elif self.mode == 'Training':
            data = sobol_seq.i4_sobol_generate(self.NumPar, self.amount)
        else:
            print('Invalid Data Mode')
            exit

        return data

    def InsertConstCol(self, data):
        # prepare output array
        Full = np.zeros((self.amount, self.TotNumPar))
        # FIXME prepare mask, hard coded for now
        # mask = np.zeros((self.NumPar), dtype=np.integer)
        mask = np.array([3, 6, 7, 8, 11, 15, 18, 19, 20, 23, 27, 30, 31, 32, 35])
        k = 0
        # prepare indexing of the constants column
        PrevProfSize = 0
        for i in range(self.ConstCol[:, 0].size):
            self.ConstCol[i, :] = self.ConstCol[i, :] + PrevProfSize
            PrevProfSize += self.ProfileSize[i]
        # insert Data and const columns in correct order
        for j in range(21):
            if j not in self.ConstCol:
                # mask[k] = j
                k += 1
        print(mask.shape)
        print(data.shape)
        Full[:, mask] = data
        return Full

    def ScaleToInterval(self, data):
        return data * (self.Bounds[1] - self.Bounds[0]) + self.Bound[0]

    def Save(self, data, path='./', name='summary'):
        fname = '{}{}'.format(path, name)
        np.savetxt(fname, data)

    def run(self):
        self.CheckTiTe()
        data = self.Generate()
        full = self.InsertConstCol(data)
        return full


if __name__ == '__main__':
    gen = GenerateData(Mode='Test', Amount=2**17)
    data = gen.run()
    gen.Save(data, name='test217.txt')
