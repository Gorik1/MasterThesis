# -*- coding: utf-8 -*-

import numpy as np
import sobol_seq


def Gen_Sobol(ConstCol, TotParNum, Sobnum):
    # Check whether TI = TE, if so set the to equal parameters and reduce dim for sobol point generation
    if (TiEQTe == 1):
        ConstCol[1] = ConstCol[0]
        Parnum = TotParNum - ConstCol[2].size - 2*ConstCol[0].size

        # Generate the Sobol Points
        Temp = sobol_seq.i4_sobol_generate(Parnum, Sobnum)
        SobPar = np.concatenate((Temp[:, :(TotParNum - 2*ConstCol[0].size)],
                                 Temp[:, :(TotParNum - 2*ConstCol[0].size)],
                                 Temp[:, (TotParNum - 2*ConstCol[0].size):]), axis=1)
    else:
        SobPar = sobol_seq.i4_sobol_generate(Parnum, Sobnum)

    return SobPar


def Rel_Gen_Sob(TotParNum, TiEQTe, Sobnum):
    if (TiEQTe == 1):
        Parnum = TotParNum - 7
    Sobol_Points = sobol_seq.i4_sobol_generate(Parnum, Sobnum)
    return Sobol_Points


def InsertConst(ConstPos, Sobnum, TotNumDim, Data):
    # prepare output array
    Full = np.zeros([Sobnum, TotNumDim])
    # prepare indexing of the constants column
    for i in range(3):
        ConstPos[i, :] = ConstPos[i, :] + i * 12 - 1
    # insert Data and const columns in correct order
    k = 0
    for j in range(36):
        if j in ConstPos:
            k += 1
        else:
            Full[:, j] = Data[:, j-k]
    return Full


def Scale2Interval(Bounds, SobolPoints):
    SobolPoints = SobolPoints * (Bounds[1] - Bounds[0]) - (Bounds[1] - Bounds[0]) / 2
    return SobolPoints


def Scale2Eirene(Scales, Units, SobolPoints):
    # Scale Sobolpoints to physical input parameters for EIRENE
    for i in range(3):
        TotalScaleSlope = Scales[i, 1][:] - Scales[i, 0][:]
        TotalScaleOffset = (Scales[i, 1] + Scales[i, 0]) / 2
        SobolPoints[:, i*12:(i+1)*12] = (SobolPoints[:, i*12:(i+1)*12] *
                                         TotalScaleSlope * Units[i, :]) + TotalScaleOffset


def Write_TempPar(Index, SourcePath, TargetPath):
    fh_sobpar = open(SourcePath)
    sob_par = np.loadtxt(fh_sobpar)
    temp = np.reshape(sob_par[Index, :], (1, 36))
    np.savetxt(TargetPath, temp, fmt='%13.11f', delimiter=' ')


def Write_TempPar3(Index, SourcePath, TargetPath):
    fh_sobpar = open(SourcePath)
    sob_par = np.loadtxt(fh_sobpar)
    temp = np.zeros([3, 12])
    temp[0] = np.reshape(sob_par[Index, :12], (1, 12))
    temp[1] = np.reshape(sob_par[Index, 12:24], (1, 12))
    temp[2] = np.reshape(sob_par[Index, 24:], (1, 12))
    np.savetxt(TargetPath, temp, fmt='%13.11f', delimiter=' ')


def Gen_TestPar(ConstCol, TotParNum, Sobnum):
    Parnum = TotParNum - ConstCol[2].size - 2*ConstCol[0].size
    Temp = np.random.rand(Sobnum, Parnum)
    TestPar = np.concatenate((Temp[:, :(TotParNum - 2*ConstCol[0].size)],
                              Temp[:, :(TotParNum - 2*ConstCol[0].size)],
                              Temp[:, (TotParNum - 2*ConstCol[0].size):]), axis=1)
    return TestPar


def CleanUp(NumProc, Path, ResultPath, Stepsize):
    fh = list(3)
    summary = np.array([NumProc*Stepsize, 36])
    NumProc = max([1, NumProc])
    for i in range(NumProc):
        fh[i] = np.loadtxt('{}{}.format(Path, i)')

    for j in range(Stepsize):
        for i in range(NumProc):
            summary[i*j] = fh[i][j, :]
    np.savetxt(ResultPath, summary)
    # Open and close files to delete content
    for i in range(NumProc):
        fh[i] = open('{}{}.format(Path, i)', 'w')
        fh[i].close()
