from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.utils import plot_model
import NNUtility as nnu
import numpy as np
import math
<<<<<<< HEAD
import matplotlib.pyplot as plt
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras

path = 'C:/Users/Christof/Documents/GitHub/MasterThesis/Data/'

path_train = '{}{}'.format(path, 'TrainData/TrainSummary.txt')
path_test_beg = '{}{}'.format(path, 'TestData/TestSummaryBeg.txt')
path_test_end = '{}{}'.format(path, 'TestData/TestSummaryEnd.txt')

train_x, train_y = nnu.load_set(path_train)
trainlength = train_x.shape[0]

rand_x, rand_y = nnu.load_set(path_test_beg)
randlength = rand_x.shape[0]
valsplit = 0.5
randsplit = int(randlength * valsplit)

validate_x = rand_x[:, :randsplit]
validate_y = rand_y[:randsplit]
eval_x = rand_x[:, randsplit:]
eval_y = rand_y[randsplit:]

sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
Cback_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                            patience=20, verbose=1, mode='auto',
                                            baseline=None)
=======
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

train_x, train_y = nnu.load_set('./summary_train.txt')
validate_x, validate_y = nnu.load_set('./summary_validate.txt')

sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
Cback_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                            patience=20, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=False)
>>>>>>> parent of 3f1ee5f... Update 20/11/18
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                   distribution='normal', seed=None)

# Currently 
model = Sequential()
model.add(Dense(14, input_dim=14, activation='sigmoid', ))
<<<<<<< HEAD
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit(x=train_x, y=train_y, batch_size=int(trainlength/5), epochs=10000, verbose=1,
                 callbacks=Cback_early, validation_split=0,
=======
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit(x=train_x, y=train_y, batch_size=32, epochs=10000, verbose=1,
                 callbacks=, validation_split=0,
>>>>>>> parent of 3f1ee5f... Update 20/11/18
                 validation_data=(validate_x, validate_y), shuffle=True,
                 class_weight=None, sample_weight=None, initial_epoch=0,
                 steps_per_epoch=None, validation_steps=None)



plot_model(model, to_file='./result.png')
print(hist.history)
<<<<<<< HEAD

score = model.evaluate(eval_x, eval_y, verbose=1)

model.save('Showcase.h5')

=======

>>>>>>> parent of 3f1ee5f... Update 20/11/18
SVG(model_to_dot(model).create(prog='dot', format='svg'))
