# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:31:38 2018

@author: ChristofBackhaus
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

    def __init__(self, wi=14, de=20, parnum=2**17, dr=0.4, ac='relu', bs=2048,
                 vs=0.25, ds=1, mode='equal'):

        # INITALIZE HYPERPARAMETERS ###
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

        # Open Training/Test data atm hard coded
        self.x_train, self.y_train, self.stdtime_train = nnu.load_set('../Data/TrainData/TrainSummary.txt')
        print('Training Data loaded')
        self.x_test, self.y_test, self.stdtime_test = nnu.load_set('../Data/TestData/TestSummary.txt')
        print('Test Data loaded')

        # Choose Train and Validation Data
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

        mod = 0
        summe = 0
        a = np.ones(self.depth, dtype=int)

        # Choose how to form the NN

        # Default, all layers contains equal amounts of Neurons
        if self.buildmode == 'equal':
            tempwidth = self.width * a

        # Triangle like shape, largest at the beginning and smallest at end
        # Input Width = minimum Depth of Triangle
        elif self.buildmode == 'triangle':
            while summe < self.paramnum:
                mod = mod + 1
                for i in range(self.depth):
                    a[i] = self.width + mod * i

                summe = np.sum(a[:-1] * a[1:]) + np.sum(a) + 14 * a[0] + a[-1]

            tempwidth = a

        # Triangle like shape, largest at the end and smallest at beginning
        # Same as Triangle , but output is flipped
        elif self.buildmode == 'reverse triangle':
            while summe < self.paramnum:
                mod = mod + 1
                for i in range(self.depth):
                    a[i] = self.width + mod * i

                summe = np.sum(a[:-1] * a[1:]) + np.sum(a) + 14 * a[0] + a[-1]

            tempwidth = np.flip(a)

        # Sandclock
        elif self.buildmode == 'clock':
            while summe < self.paramnum:
                mod = mod + 1
                for i in range(self.depth):
                    a[i] = self.width + mod * i

                summe = 2 * np.sum(a[:-1] * a[1:]) + 2 * np.sum(a) + 15 * a[-1]

            tempwidth = np.append(np.flip(a), a[1:])

        # Widest at center equally dropping of to beginning and end
        elif self.buildmode == 'centered':
            while summe < self.paramnum:
                mod = mod + 1
                for i in range(self.depth):
                    a[i] = self.width + mod * i

                summe = 2 * np.sum(a[:-1] * a[1:]) + np.sum(a) + 15 * a[-1]

            tempwidth = np.append(a, np.flip(a[:-1]))

        else:
            print('Invalid NN build mode')
            exit

        adjust_length = 3
        m_front = np.floor((tempwidth[0] - 14) / (adjust_length+1))
        m_back = np.floor((tempwidth[-1] - 1) / (adjust_length+1))
        width_front = np.zeros(adjust_length, dtype=int)
        width_back = np.zeros(adjust_length, dtype=int)

        for i in range(adjust_length):
            width_front[i] = 14 + (i+1) * m_front
            width_back[i] = 1 + (i+1) * m_back
        twidth = np.concatenate((width_front, tempwidth, width_back))
        tdepth = twidth.shape[0]

# Add layers to model
        self.model.add(Dense(14, activation='linear'))

        if self.activation == 'LeakyReLU':
            for i in range(tdepth):
                self.model.add(Dense(twidth[i],
                                     kernel_initializer='he_normal',
                                     bias_initializer='zeros',
                                     kernel_regularizer=regularizers.l2(0.01),
                                     bias_regularizer=None,
                                     activity_regularizer=None))
                self.model.add(LeakyReLU(alpha=0.2))
                self.model.add(Dropout(self.droprate))
        else:
            for i in range(tdepth):
                self.model.add(Dense(twidth[i], activation=self.activation,
                                     kernel_initializer='he_normal',
                                     bias_initializer='zeros',
                                     kernel_regularizer=regularizers.l2(0.01)))
                self.model.add(Dropout(self.droprate))

        self.model.add(Dense(1, activation='linear'))

    def run(self):
        NN.setup(self)
        print('Setup Successful')
        self.model.compile(optimizer=self.adam, loss='mean_squared_error')
                            # , metrics=[nnu.metr_abs_dif, nnu.metr_rel_dif])
        print('Begin Fit')
        # self.hist = self.model.fit(x=self.x_train, y=self.y_train, batch_size=16384, epochs=50)
        self.hist = self.model.fit(x=self.x_train, y=self.y_train, batch_size=16384, epochs=10000,
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

        with open(self.path + 'report.txt', 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: fh.write(x + '\n'))


if __name__ == "__main__":
    Net = NN(50, 32, 2**17, 0.4, 'LeakyReLU', 16384, 0, 1, 'equal')
    Net.run()
    Net.evaluate()
    Net.save()
    Net.plot()
