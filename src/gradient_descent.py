import numpy as np
import time
import math
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from config.displayConfig import DisplayConfig
from utils.math_utils import reLu, reLu_deriv, leaky_relu_deriv, softmax
from utils.color import green_str, red_str, gray_str, yellow_str
from utils.plots.dynamicLinePlot import DynamicLinePlot

dataconfig = DataConfig()
modelconfig = ModelConfig()
displayconfig = DisplayConfig()
display_delay = displayconfig.SHOW_ACCURACY_TIME


def init_params(num_inputs:int, num_outputs:int, size_layer1:int):
    """Initialize weights and offsets"""
    W1 = np.random.randn(size_layer1, num_inputs) * np.sqrt(2.0 / num_inputs)
    b1 = np.zeros((size_layer1, 1))
    W2 = np.random.randn(num_outputs, size_layer1) * np.sqrt(2.0 / size_layer1)
    b2 = np.zeros((num_outputs, 1))
    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = reLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = np.concatenate([softmax(Z2[(dataconfig.STRINGS*i):(dataconfig.STRINGS*(i+1))]) for i in range(dataconfig.STRINGS)])
    return Z1, A1, Z2, A2


def one_hot(Y:np.ndarray) -> np.ndarray:
    nrow = Y.shape[1]
    one_hot_Y = np.empty((nrow,0))
    for i in dataconfig.Y_STRINGS:
        one_hot_Y_string_i = np.zeros((nrow, dataconfig.STRINGS))
        one_hot_Y_string_i[np.arange(nrow), Y[i] -1] = 1
        one_hot_Y= np.concatenate((one_hot_Y, one_hot_Y_string_i), axis=1)
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def mask(Y:np.ndarray) -> np.ndarray:
    mask_Y = np.zeros((Y.shape[1], dataconfig.STRINGS*dataconfig.STRINGS)).T
    mask_indices = np.argwhere(Y != 0)  # Get indices where Y is not zero
    for i, j in mask_indices:
        if i in dataconfig.Y_STRINGS:
            mask_Y[dataconfig.STRINGS*i : dataconfig.STRINGS*(i+1), j] = 1  # Set consecutive indices of strings to 1
    return mask_Y


def backward_prop(Z1:np.ndarray, A1:np.ndarray, Z2:np.ndarray, A2:np.ndarray, W2:np.ndarray, X:np.ndarray, Y:np.ndarray):
    m = Y.shape[1]
    one_hot_Y = one_hot(Y) 
    dZ2 = (A2 - one_hot_Y) * mask(Y)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * reLu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    error_sum = np.sum(np.abs(dZ2))
    return dW1, db1, dW2, db2, error_sum


def updata_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    pred_strings = np.array([np.argmax(A2[(dataconfig.STRINGS*i):(dataconfig.STRINGS*(i+1))], 0) + 1
                                 for i in range(dataconfig.STRINGS)])
    return pred_strings


def get_accuracy(pred, Y):
    for n in range(20):
        print(f"pred{str(n).zfill(2)} : [", end='')
        for i in (dataconfig.Y_STRINGS):
            print(gray_str(pred[i][n] if Y[i][n]==0 else (green_str((pred[i][n])) if pred[i][n]==Y[i][n] else red_str(pred[i][n]))), end= ' ')
        print(']', end='  ')
        print()
    mask = Y != 0
    return np.sum((pred == Y) & mask) / np.sum(mask)


def gradient_descent(X,Y, iterations, alpha):
    dynLinePlot = DynamicLinePlot('Variables vs. Iteration', 'Iteration', 'Proportion')
    num_outputs = dataconfig.STRINGS * dataconfig.STRINGS
    W1, b1, W2, b2 = init_params(X.shape[0], num_outputs, modelconfig.SIZE_LAYER1)
    
    prevacc = 0
    start_time = time.time()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2, error_sum = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updata_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % display_delay == 0):
            acc = get_accuracy(get_predictions(A2), Y)
            err = round((error_sum/Y.shape[1])/12, 5)

            end_time = time.time()
            hours = (int)((end_time-start_time)/display_delay*(iterations-i))//3600
            minutes = (int)(((end_time-start_time)/display_delay*(iterations-i))//60)%60
            seconds = round(((end_time-start_time)/display_delay*(iterations-i))%60)
            
            print()
            print()
            print(f"Iterations: {i}")
            print(f"Time per iteration: {round((end_time-start_time)/display_delay,3)}s ; ETA: {hours}h {minutes}m {seconds}s")
            print(f"Accuracy: {yellow_str(round(acc,4))}")
            print(f"Relative delta acc: {green_str('+' + str(round((acc-prevacc)/prevacc*100,3)) + "%") if acc-prevacc>0 else red_str(round((acc-prevacc)/prevacc*100,3)) + "%"}")
            print(f"Outputs avg error: {red_str(round((error_sum/Y.shape[1])/12, 5))}")
            dynLinePlot.update(i, {'acc':acc,'err':err})
            prevacc = acc
            start_time = time.time()
    return W1, b1, W2, b2, dynLinePlot