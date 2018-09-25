import numpy as np

def load_set(path):
    data = np.loadtxt(path)
    x_data = data[:, :-2]
    y_data = data[:, -2:-1]
    return x_data, y_data

def norm_var(data):
    for column in range(data.shape[1]):
        data[column] = data[column] / np.var(data[column])
    return data

def custom_loss(y_true, y_pred):
    ret_up = np.square(y_true[0] + y_true[1] - y_pred[0])
    ret_low = np.square(y_true[0] - y_true[1] - y_pred[1])
    return ret_up, ret_low