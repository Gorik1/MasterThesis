# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:31:38 2018

@author: ChristofBackhaus
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU, LeakyReLU
from keras import optimizers, initializers, callbacks
import NNUtility as nnu
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
###############################################################################
# DATASPLIT INTELLIGENTLY: CHOOSE "COMPLETE" SETS FOR TRAININGS AND VALIDATION#
###############################################################################


class NN(object):

    def __init__(self, wi=14, de=20, parnum=2**17, dr=0.4, ac='relu', bs=2048,
                 vs=0.25, ds=1, mode='equal'):

        ### INITALIZE HYPERPARAMETERS ###
        self.width = wi  # Integer
        self.depth = de  # Integer
        self.droprate = dr  # Float 0 <= x < 1
        self.activation = ac  # String 'relu' 'elu' 'sigmoid' etc.
        self.batchsize = bs  # Integer 
        self.valsplit = vs  # Float 0 <= x < 1
        self.datasplit = ds  # Float 0 <= x < 1
        self.buildmode = mode  # Structure of NN
        self.paramnum = parnum

        # GENERATE PATHNAME
        self.name = '{}{}{}{}{}{}'.format(mode,
                                         '{}{}'.format('W', str(self.width)),
                                         '{}{}'.format('D', str(self.depth)),
                                         str(self.paramnum),
                                         self.activation,
                                         str(ds))
        self.path = '{}{}'.format('../Data/Results/', self.name)


        # INITALIZE CHOICE OF KERAS FUNCTIONS #
        self.model = Sequential()

        self.sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
        self.adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        self.adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                    epsilon=None, decay=0., amsgrad=False)
        self.cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                          patience=50, verbose=1, mode='min',
                                          baseline=None)
                                          # restore_best_weights=True)
        initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                     distribution='normal',
                                     seed=None)

        ### Open Training/Test data atm hard coded
        self.x_train, self.y_train, self.stdtime_train = nnu.load_set('../Data/TrainData/TrainSummary.txt')
        print('Training Data loaded')
        self.x_test, self.y_test, self.stdtime_test = nnu.load_set('../Data/TestData/TestSummary.txt')
        print('Test Data loaded')
        ### Choose Train and Validation Data
        if self.datasplit < 1:
            length = int(self.x_train[:, 0].size * self.datasplit)
            self.x_validate = self.x_train[length:, :]
            self.x_train = self.x_train[:length, :]
            self.y_validate = self.y_train[length:]
            self.y_train = self.y_train[:length]
        else:
            length = round(self.x_test[:, 0].size / 2)
            self.x_validate = self.x_test[:length, :]
            self.x_test = self.x_test[length:, :]
            self.y_validate = self.y_test[:length]
            self.y_test = self.y_test[length:]

    def setup(self):
        self.model.add(Dense(14, activation='linear'))
        ### Choose how to form the NN
        ### Default, all layers contains equal amounts of Neurons 
        if self.buildmode == 'equal':
            for i in range(self.depth):
                self.model.add(Dense(self.width, activation=self.activation))
                self.model.add(Dropout(self.droprate))

        ### Triangle like shape, largest at the beginning and smallest at end
        elif self.buildmode == 'triangle':
            minwidth = 7
            for i in range(self.depth):
                delta = int((self.width - minwidth) / self.depth)
                tempwidth = self.width - i * delta
                self.model.add(Dense(tempwidth, activation=self.activation))
                self.model.add(Dropout(self.droprate))

        ### Triangle like shape, largest at the end and smallest at beginning
        elif self.buildmode == 'reverse triangle':
            minwidth = 7
            for i in range(self.depth):
                delta = int((self.width - minwidth) / self.depth)
                tempwidth = minwidth + i * delta
                self.model.add(Dense(tempwidth, activation=self.activation))
                self.model.add(Dropout(self.droprate))

        ### Widest at center equally dropping of to beginning and end 
        elif self.buildmode == 'centered':
            minwidth = 7
            for i in range(self.depth):
                delta = int(2 * (self.width - minwidth) / self.depth)
                tempwidth = self.width - abs(self.depth / 2 - i) * delta
                self.model.add(Dense(self.width, activation=self.activation))
                self.model.add(Dropout(self.droprate))
            
        elif self.buildmode == 'NNGP':
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
        
        else:
            print('Invalid NN build mode')
            exit

        self.model.add(Dense(1, activation='linear'))

    def run(self):
        NN.setup(self)
        self.model.compile(optimizer=self.adam, loss='mean_squared_error')
                            # , metrics=[nnu.metr_abs_dif, nnu.metr_rel_dif])
        self.hist = self.model.fit(x=self.x_train, y=self.y_train, batch_size=16384, epochs=10000,
                                   verbose=1, callbacks=[self.cb], validation_split=0,
                                   validation_data=(self.x_validate, self.y_validate), shuffle=True,
                                   class_weight=None, sample_weight=None, initial_epoch=0,
                                   steps_per_epoch=None, validation_steps=None)
        print(self.hist)

    ### Evaluate Trained Network ###
    # @todo 
    def evaluate(self):
        self.testres = self.model.evaluate(self.x_test, self.y_test,
                            batch_size=self.batchsize)

    def plot(self):
        Epochs = len(self.hist.history['loss'])
        loss = self.hist.history['loss']
        val_loss = self.hist.history['val_loss']

        Fig = plt.figure()
        plt.plot(loss)
        plt.plot(val_loss)
        plt.hlines(self.testres, 0, Epochs, colors='k', label='Test loss')

        plt.text(round(Epochs-5, -1), 1.1*loss[-1], '%.3f' % loss[-1])
        plt.text(round(Epochs-5, -1), 1.1*val_loss[-1], '%.3f' % val_loss[-1])
        plt.text(round(Epochs-5, -1), 0.9*self.testres, '%.3f' % self.testres)
        
        plt.title(self.name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        Fig.savefig('{}{}'.format(self.path, '/Plot.pdf'), format='pdf')
        plt.show()

    def save(self):
        # check exist otherwise create subdirectory to save to
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # save NN
        nameh5 = '{}{}'.format(self.path, '/network.h5')
        self.model.save(nameh5)

        namehist = '{}{}'.format(self.path, '/history')
        with open(namehist, 'wb') as file_pi:
            pickle.dump(self.hist.history, file_pi)

        nametest = '{}{}'.format(self.path, '/TestRes.txt')
        with open(nametest, 'w') as file_pi:
            file_pi.write('%f' % self.testres)


if __name__ == "__main__":
    Net = NN(wi=32, de=32)
    Net.run()
    Net.evaluate()
    Net.save()
    Net.plot()
