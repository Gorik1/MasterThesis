# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:24:24 2020

@author: Christof
"""

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, PReLU, LeakyReLU
from keras import optimizers, initializers, callbacks, regularizers
import NNUtility as nnu
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


class NN(object):

    def __init__(self, xtrain, ytrain, xval, yval, xtest, ytest, wi=14, dr=0.4, ac='relu', acpar=0.1, bs=2048):

        # INITALIZE HYPERPARAMETERS ###
        self.width = wi  # Integer
        self.droprate = dr  # Float 0 <= x < 1
        self.activation = ac  # String 'relu' 'elu' 'sigmoid' etc.
        self.activation_par = acpar
        self.batchsize = bs  # Integer
        self.x_train = xtrain
        self.x_validate = xval
        self.x_test = xtest
        self.y_train = ytrain
        self.y_validate = yval
        self.y_test = ytest

        # GENERATE PATHNAME
        self.name = '{}{}{}{}{}'.format(self.activation, self.batchsize, self.droprate, self.width, self.activation_par)
        self.path = '{}{}'.format('../Data/Results/AutoEncoder/', self.name)

        # INITALIZE CHOICE OF KERAS FUNCTIONS #
        self.model = Sequential()

        self.sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
        self.adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        self.adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                    epsilon=10e-8, decay=0.001, amsgrad=False)
        self.cb = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                          patience=50, verbose=1, mode='min',
                                          baseline=None,
                                          restore_best_weights=True)
        initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                     distribution='normal',
                                     seed=None)
        initializers.he_normal(151119)
        initializers.Zeros()

    def setup(self):
        start = 14
        width = 5
        a = np.arange(width, start, 1, dtype=np.integer)
        tempwidth = np.concatenate((np.flip(a[1:]), a))
        tempdepth = tempwidth.shape[0]

# Add layers to model
        self.model.add(Dense(14, activation='linear'))

        if self.activation == 'LeakyReLU':
            for i in range(tempdepth):
                self.model.add(Dense(tempwidth[i],
                                     kernel_initializer='he_normal',
                                     bias_initializer='zeros',
                                     kernel_regularizer=regularizers.l2(0.01),
                                     bias_regularizer=None,
                                     activity_regularizer=None))
                self.model.add(LeakyReLU(alpha=0.2))
                self.model.add(Dropout(self.droprate))
        else:
            for i in range(tempdepth):
                self.model.add(Dense(tempwidth[i], activation=self.activation,
                                     kernel_initializer='he_normal',
                                     bias_initializer='zeros',
                                     kernel_regularizer=regularizers.l2(0.01)))
                self.model.add(Dropout(self.droprate))

        self.model.add(Dense(14, activation='linear'))

    def run(self):
        NN.setup(self)
        print('Setup Successful')
        self.model.compile(optimizer=self.adam, loss='mean_squared_error')
        # , metrics=[nnu.metr_abs_dif, nnu.metr_rel_dif])
        print('Begin Fit')
        self.hist = self.model.fit(x=self.x_train, y=self.y_train, batch_size=self.batchsize, epochs=10000,
                                   verbose=1, callbacks=[self.cb], validation_split=0,
                                   validation_data=(self.x_validate, self.y_validate), shuffle=True,
                                   class_weight=None, sample_weight=None, initial_epoch=0,
                                   steps_per_epoch=None, validation_steps=None)

    # Evaluate Trained Network ###
    def evaluate(self):
        self.testres = self.model.evaluate(self.x_test, self.y_test,
                                           batch_size=self.batchsize)

    def plot(self):
        Epochs = len(self.hist.history['loss'])
        loss = self.hist.history['loss']
        val_loss = self.hist.history['val_loss']

        Fig = plt.figure()
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.hlines(self.testres, 0, Epochs, colors='k', label='Test loss')

        plt.text(round(Epochs-5, -1), 1.1*loss[-1], '%.3f' % loss[-1])
        plt.text(round(Epochs-5, -1), 1.1*val_loss[-1], '%.3f' % val_loss[-1])
        plt.text(round(Epochs-5, -1), 0.9*self.testres, '%.3f' % self.testres)

        plt.title(self.name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.legend()

        Fig.savefig('{}{}'.format(self.path, '/Plot.pdf'), format='pdf')
        plt.show()

    def save(self):
        # check exist otherwise create subdirectory to save to

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # save NN as H5 file
        nameh5 = '{}{}'.format(self.path, '/network.h5')
        self.model.save(nameh5)

        namehist = '{}{}'.format(self.path, '/history')
        with open(namehist, 'wb') as file_pi:
            pickle.dump(self.hist.history, file_pi)

        nametest = '{}{}'.format(self.path, '/TestRes.txt')
        with open(nametest, 'w') as file_pi:
            file_pi.write('%f' % self.testres)

        with open(self.path + '/Report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


if __name__ == "__main__":

    # Open Training/Test data atm hard coded
    x_train, y_dummy, stdtime_train = nnu.load_set('../Data/TrainData/TrainSummary.txt')
    y_train = x_train
    print('Training Data loaded')
    x_test, y_dummy, stdtime_test = nnu.load_set('../Data/TestData/TestSummary.txt')
    y_test = x_test
    print('Test Data loaded')

    # Choose Train and Validation Data
    length = round(x_test[:, 0].size / 2)
    x_validate = x_test[:length, :]
    x_test = x_test[length:, :]
    y_validate = x_validate
    y_test = x_test

    for activation in (['LeakyReLU', 'relu', 'elu', 'tanh', 'sigmoid']):
        print('Activation:', activation)
        for batch in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            print('Batchsize: ', batch)
            for droprate in np.arange(0, 0.9, 0.1):
                for minimum in range(14):
                    if activation == 'LeakyReLU':
                        for ac_par in np.arange(0.1, 1, 0.1):
                            Net = NN(x_train, y_train, x_validate, y_validate, x_test, y_test,
                                     minimum, droprate, activation, ac_par, batch, )
                    else:
                        Net = NN(x_train, y_train, x_validate, y_validate, x_test, y_test,
                                 minimum, droprate, activation, 0.1, batch,)
                    Net.run()
                    Net.evaluate()
                    Net.save()
                    Net.plot()
