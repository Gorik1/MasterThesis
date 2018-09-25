from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.utils import plot_model
import NNUtility as nnu
import numpy as np
import math
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

train_x, train_y = nnu.load_set('./summary_train.txt')
validate_x, validate_y = nnu.load_set('./summary_validate.txt')

sgd = optimizers.SGD(lr=0.01, momentum=0.001, decay=0.001)
Cback_early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5,
                                            patience=20, verbose=1, mode='auto',
                                            baseline=None, restore_best_weights=False)
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in',
                                   distribution='normal', seed=None)

model = Sequential()
model.add(Dense(14, input_dim=14, activation='sigmoid', ))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
hist = model.fit(x=train_x, y=train_y, batch_size=32, epochs=10000, verbose=1,
                 callbacks=, validation_split=0,
                 validation_data=(validate_x, validate_y), shuffle=True,
                 class_weight=None, sample_weight=None, initial_epoch=0,
                 steps_per_epoch=None, validation_steps=None)



plot_model(model, to_file='./result.png')
print(hist.history)

SVG(model_to_dot(model).create(prog='dot', format='svg'))
