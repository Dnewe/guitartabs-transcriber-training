import numpy as np
import config
from utils.math_utils import sigmoid, reLu, reLu_deriv, softmax


def init_params(num_inputs:int, num_outputs:int, size_layer1:int):
    w1 = np.random.randn(size_layer1,num_inputs)
    b1 = np.random.randn(size_layer1,1)
    w2 = np.random.randn(num_outputs,size_layer1)
    b2 = np.random.randn(num_outputs,1)
    return w1, b1, w2, b2


def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = reLu(z1)
    z2 = w2.dot(a1) + b2
    #output_note = sigmoid(z2[config.Y_MULTI_START:config.Y_MULTI_END]) if config.MULTICLASS_LABELS else np.empty((0,z2.shape[1]))
    #output_fret1 = softmax(z2[config.Y_SINGLE_START:config.Y_SINGLE_END]) if config.SINGLECLASS_LABELS else np.empty((0,z2.shape[1]))
    a2 = np.concatenate((sigmoid(z2[config.Y_MULTI_START:config.Y_MULTI_END]) if config.MULTICLASS_LABELS else np.empty((0,z2.shape[1])),
                         softmax(z2[config.Y_SINGLE_START:config.Y_SINGLE_END]) if config.SINGLECLASS_LABELS else np.empty((0,z2.shape[1]))))
    return z1, a1, z2, a2


def one_hot_y(Y):
    Y_note = Y[config.Y_MULTI_START:config.Y_MULTI_END] if config.MULTICLASS_LABELS else np.empty((0,1))
    Y_fret1 = np.zeros((Y.size, 25))
    Y_fret1[np.arange(Y.size), Y] = 1  


def backward_prop(z1:np.ndarray, a1:np.ndarray, z2:np.ndarray, a2:np.ndarray, w2:np.ndarray, X:np.ndarray, Y:np.ndarray):
    m = Y.size
    dZ2 = a2 - Y
    dW2 = 1 / m * dZ2.dot(a1.T)
    db2 = 1 / m * np.sum(dZ2) 
    dZ1 = w2.T.dot(dZ2) * reLu_deriv(z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def updata_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha):
    w1 = w1 - alpha * dW1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dW2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.concatenate(((a2 >= 0.5).astype(int) if config.MULTICLASS_LABELS else np.empty((0,a2.shape[1])),
                           np.argmax(a2, 0) if config.SINGLECLASS_LABELS else np.empty((0,a2.shape[1]))))
    #return np.argmax(a2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X,Y, iterations, alpha):
    num_outputs = (config.Y_MULTI_START - config.Y_MULTI_END) if config.MULTICLASS_LABELS else 0 + ((config.Y_SINGLE_START-config.Y_SINGLE_END)*24) if config.SINGLECLASS_LABELS else 0
    w1, b1, w2, b2 = init_params(X.shape[0], num_outputs, config.SIZE_LAYER1)
    
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(z1, a1, z2, a2, w2, X, Y)
        w1, b1, w2, b2 = updata_params(w1, b1, w2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print(f"Iterations: {i}")
            print(f"Accuracy: {get_accuracy(get_predictions(a2), Y)}")
    return w1, b1, w2, b2