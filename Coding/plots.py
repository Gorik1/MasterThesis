# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:28:26 2018

@author: ChristofBackhaus
"""

# Plotting input and output in 1d and 2d histograms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

path = 'C:/MasterThesis/Output/'
filename = 'TrainingData'
fh = open('C:/Users/ChristofBackhaus/Documents/GitHub/MasterThesis/Data/TrainData/TrainSummary.txt')
summary = np.loadtxt(fh)

label = summary[:, -3]
std = summary[:, -2]*label
time = summary[:, -1]
stdxtime = std * time
inputs = summary[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]

percent = std / label

dict = {'0': 'P1', '1': 'P2', '2': 'P3', '3': 'P4', '4': 'P5', '5': 'P6',
        '6': 'P7', '7': 'P1N', '8': 'P2N', '9': 'P3N', '10': 'P4N',
        '11': 'P5N', '12': 'P6N', '13': 'P7N', '14': 'SpR', '15': 'STD',
        '16': 'Time'}

dirname = '{}{}'.format(path, filename)
os.makedirs(dirname, exist_ok=True)

for column in range(summary.shape[1] - 3):
    temp = str(column)
    Fig = plt.figure(num=dict[temp])
    plt.rcParams['agg.path.chunksize'] = 10000
    # xrange = [summary[:, column].min(), summary[:, column].max()]
    # yrange = [label.min(), label.max()]
    # plt.errorbar(summary[:, column], label, std, ',')
    plt.plot(summary[:, column], label, ',')
    plt.xlabel(dict[temp])
    plt.ylabel('Sputter Rate')
    plotname = '{}{}{}{}{}'.format(path, filename, '/', dict[temp], '.pdf')
    Fig.savefig(plotname, dpi=200, format='pdf')
    plt.close(Fig)

    FigHist = plt.figure(num=temp)
    plt.hist(summary[:, column], bins=100, rwidth=0.75)
    plt.ylabel('Count')
    plt.ylim(1280, 1340)
    plotname = '{}{}{}{}{}'.format(path, filename, '/', dict[temp], 'Hist.pdf')
    FigHist.savefig(plotname, dpi=200, format='pdf')
    plt.close(FigHist)
