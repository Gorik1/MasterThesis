# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 10:09:02 2018

@author: c.backhaus
"""

# Programm Ablauf Plan
# 1) Generate Parametersets(Number of Parameters, TiEQTe, TestOrTrain)
# *In Principal this step can be skipped for Trainingdata since generating Sobolpoints once is enough
# *But since the overhead is very small it remains implemented until further notice
# *This allows for an easy switch to randomly drawn training points
# *Precision length after decimal of 2^n points is n (to be precise the 2^nth number would already need n+1)
#   a) Check whether Train or Test
#   b) Create appropriate amount of Parametersets
# 2) Scale Input parameters to interval and save for later summary(Parametersetset)
#   a) Scale to Interval
#   b) Save for later use
# 3) Run Eirene for Trainingdata(Parameterset, number of cores)
#   a) determine whether training or test data
#   b) change input appropriately
#   c) Set number of cores for task (set shell command)
#   d) Start run
# 3.5) Clean outputfile to reduce read in time every so often
#   a) Append current output content to totaloutput file
# 4) Save Eirene Output
# 5) Concatenate Parameters and output to summary file

# import subprocess
import numpy as np
from fileinput import os
import SobolUtility as su

num_train = 10000
num_test = num_train / 2
Stepsize = num_train / 100
ProcNum = 0  # Number of Processors -1 (Since python counts from 0), for later MPI implementation
ParMode = ['Test', 'Train']  # Kind of Parameterset to generate, process
TiEQTe = True  # Logical for to set T_i = T_e (reduce dimensionality)
Bounds = [-1, 1]  # Interval boundaries for input parameters
WorkPath = "/home/c.backhaus/python/RedMod/"  # Working directory for generating parameter set
ParPath = "/home/c.backhaus/1d-farming/Eirene/input/"  # Path of input file
OutPath = "/home/c.backhaus/python/RedMod/summary"  # Path for summary file


i = 0

# Generate Parameterset
for kind in ParMode:
    Param = su.GenerateParam(kind, num_train, num_par, TiEQTe)
    temp_par = su.Scale2Interval(Bounds, Param)
    scalepar = '{}{}{}'.format(WorkPath, kind, '.par')
    np.savetxt(scalepar, temp_par)

    if (kind == 'Train'):
        num_sample = num_train
    elif (kind == 'Test'):
        num_sample = num_train / 2
    else:
        num_sample = 0
        print('Error in parameter kind selection')

    for i in range(num_sample):
        np.savetxt(ParPath, Param)  # Save current Parameters to input folder
        # Run Eirene with current Parameterset
        os.system(". /home/c.backhaus/1d-farming/Eirene/input/singlerun.sh")
        if i % Stepsize == 0:
            su.CleanUp

    # Concatenate scaled input parameters and Eirene output for summary
    inpfh = open(scalepar)
    inp_par = np.loadtxt(inpfh)
    # num_proc output noch nicht implementiert
    # for i in range(num_proc):
    #     id = '{}{}{}'.format(path, 'outfarmproc', i)
    #     idfh = open(id)
    #     result = np.loadtxt(idfh)
    #     summary_fh = open(summary_mpi, mode='w')
    #     np.savetxt(summary_fh, result)
    outfh = open('/home/c.backhaus/1d-farming/Eirene/input/outfarm.proc')
    Eirene_out = np.loadtxt(outfh)
    summary_data = np.concatenate((inp_par, Eirene_out), axis=1)
    summarykind = '{}{}{}'.format(OutPath, kind, '.txt')
    np.savetxt(summarykind, summary_data)
