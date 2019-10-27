# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:31:38 2018

@author: ChristofBackhaus
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers, initializers, callbacks
import NNUtility as nnu
from matplotlib import pyplot
import os
import numpy as np

###############################################################################
# DATASPLIT INTELLIGENTLY: CHOOSE "COMPLETE" SETS FOR TRAININGS AND VALIDATION#
###############################################################################


class NN(object):

    def __init__(self, wi=14, de=5, parnum=2**17, dr=0.25, ac='relu', bs=2048,
                 vs=0.25, ds=1, mode='equal'):

        ### INITALIZE HYPERPARAMETERS ###
        self.width = wi # Integer
        self.depth = de # Integer
        self.droprate = dr # Float 0 <= x < 1
        self.activation = ac # String 'relu' 'elu' 'sigmoid' etc.
        self.batchsize = bs # Integer 
        self.valsplit = vs # Float 0 <= x < 1
        self.datasplit = ds # Float 0 <= x < 1
        self.buildmode = mode # Structure of NN
        self.paramnum = parnum

        # GENERATE PATHNAME
        self.name = '{}{}{}{}'.format(mode, str(self.paramnum), ac, str(ds))
        self.path = '{}{}'.format('./', self.name)
        

        # INITALIZE CHOICE OF KERAS FUNCTIONS #
        self.model = Sequential()

        self.sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
        self.adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        self.cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                          patience=30)
        initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                     distribution='normal',
                                     seed=None)
        
        ### Open Training/Test data atm hard coded
        self.x_train, self.y_train, self.stdtime_train = nnu.load_set('../Data/TrainData/TrainSummary.txt')
        self.x_test, self.y_test, self.stdtime_test = nnu.load_set('../Data/TestData/TestSummary.txt')
        ### Choose Train and Validation Data
        if self.datasplit < 1:
            length = self.x_train[:, 0].size * self.datasplit
            self.x_validate = self.x_train[:, :length]
            self.y_validate = self.y_train[:, :length]

    def setup(self):
        ### Choose how to form the NN
        ### Default, all layers contains equal amounts of Neurons 
        if self.buildmode == 'equal':
            self.model.add(Dense(14, activation='sigmoid'))
            for i in range(self.depth):
                self.model.add(Dense(self.width, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            self.model.add(Dense(1, activation='relu'))

        ### Triangle like shape, largest at the beginning and smallest at end
        elif self.buildmode == 'triangle':
            self.model.add(Dense(14, activation='sigmoid'))
            minwidth = 7
            for i in range(self.depth):
                delta = int((self.width - minwidth) / self.depth)
                tempwidth = self.width - i * delta
                self.model.add(Dense(tempwidth, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            self.model.add(Dense(1, activation='relu'))

        ### Triangle like shape, largest at the end and smallest at beginning
        elif self.buildmode == 'reverse triangle':
            self.model.add(Dense(14, activation='sigmoid'))
            minwidth = 7
            for i in range(self.depth):
                delta = int((self.width - minwidth) / self.depth)
                tempwidth = minwidth + i * delta
                self.model.add(Dense(tempwidth, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            self.model.add(Dense(1, activation='relu'))

        ### Widest at center equally dropping of to beginning and end 
        elif self.buildmode == 'centered':
            self.model.add(Dense(14, activation='sigmoid'))
            minwidth = 7
            for i in range(self.depth):
                delta = int(2 * (self.width - minwidth) / self.depth)
                tempwidth = self.width - abs(self.depth / 2 - i) * delta
                self.model.add(Dense(self.width, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            self.model.add(Dense(1, activation='relu'))
            
        elif self.buildmode == 'NNGP':
            self.model.add(Dense(14, activation='sigmoid'))
            mindepth = 3
            modifier = int(np.sqrt(self.depth / mindepth))
            tempdepth = int(self.depth / modifier**2)
            while tempdepth < mindepth:
                modifier = modifier - 1
                tempdepth = int(self.depth / modifier**2)
            tempwidth = self.width * modifier
            for i in range(tempdepth):
                self.model.add(Dense(tempwidth, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            self.model.add(Dense(1, activation='relu'))
        
        else:
            print('Invalid NN build mode')
            exit

    def run(self):
        NN.setup()
        self.model.compile(optimizer=self.adagrad, loss='mean_squared_error',
                           metrics=[nnu.metr_abs_dif, nnu.metr_rel_dif])
        self.hist = self.model.fit(x=self.x_train, y=self.y_train, batch_size=2000, epochs=10000,
                                   verbose=1, callbacks=[self.cb], validation_split=0,
                                   validation_data=(self.x_validate, self.y_validate), shuffle=True,
                                   class_weight=None, sample_weight=None, initial_epoch=0,
                                   steps_per_epoch=None, validation_steps=None)
        # print(self.hist)

    ### Evaluate Trained Network ###
    @todo 
    def evaluate(self):
        self.model.evaluate(self.x_test, self.y_test,
                            batch_size=self.batchsize)

    def plot(self):
        
        pyplot.plot(self.hist.history['val_loss'])
        pyplot.show()
        pyplot.plot(self.hist.history['loss'])
        pyplot.show()

    def save(self):
        # check exist otherwise create subdirectory to save to
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # save NN
        name = '{}{}'.format(self.path, '/network.h5')
        self.model.save(name)
        