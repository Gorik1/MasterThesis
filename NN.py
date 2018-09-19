from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import plot_model
import NNUtility as nnu
import numpy as np
import math

train_x, train_y = nnu.load_set('./summary_train.txt')
validate_x, validate_y = nnu.load_set('./summary_validate.txt')

model = Sequential()
model.add(Dense(14, input_dim=14, activation='sigmoid'))
model.add(Dense(7, activation='relu'))
model.add(Dense(2, activation='relu'))

sgd = optimizers.SGD(lr=0.01, momentum=0.01, decay=0.001)

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
model.fit(x=train_x, y=train_y, batch_size=32, epochs=100, verbose=1,
          callbacks=None, validation_split=0,
          validation_data=(validate_x, validate_y), shuffle=False,
          class_weight=None, sample_weight=None, initial_epoch=0,
          steps_per_epoch=None, validation_steps=None)

plot_model(model, to_file='.\result.png')
