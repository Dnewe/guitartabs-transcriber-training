import numpy as np
import config
from utils.math_utils import reLu, reLu_deriv, leaky_relu_deriv, softmax
from utils.color import green_str, red_str, gray_str
from utils.plots.dynamicLinePlot import DynamicLinePlot


def init_params(num_inputs:int, num_outputs:int, size_layer1:int):
    """Initialize weights and offsets parameters"""
    #W1 = np.random.rand(size_layer1,num_inputs) - 0.5
    #b1 = np.random.rand(size_layer1,1) - 0.5
    #W2 = np.random.rand(num_outputs,size_layer1) - 0.5
    #b2 = np.random.rand(num_outputs,1) - 0.5
    W1 = np.random.randn(size_layer1, num_inputs) * np.sqrt(2.0 / num_inputs)
    b1 = np.zeros((size_layer1, 1))
    W2 = np.random.randn(num_outputs, size_layer1) * np.sqrt(2.0 / size_layer1)
    b2 = np.zeros((num_outputs, 1))
    return W1, b1, W2, b2


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = reLu(Z1)
    Z2 = W2.dot(A1) + b2
    #print()
    #print("Z2")
    #print(Z2[0:25,0])
    A2_1 = softmax(Z2[0:config.MAX_POSITION])
    A2_2 = softmax(Z2[config.MAX_POSITION:config.MAX_POSITION*2])
    A2_3 = softmax(Z2[config.MAX_POSITION*2:config.MAX_POSITION*3])
    A2_4 = softmax(Z2[config.MAX_POSITION*3:config.MAX_POSITION*4])
    A2_5 = softmax(Z2[config.MAX_POSITION*4:config.MAX_POSITION*5])
    A2_6 = softmax(Z2[config.MAX_POSITION*5:config.MAX_POSITION*6])
    #print()
    #print(f"A2 (shape {A2_1[0:25,0].shape})")
    #print(A2_1[0:25,0])
    #print()
    #print("Maximum A2_1")
    #print(np.max(A2_1[0:25,0]))
    A2 = np.concatenate((A2_1, A2_2, A2_3, A2_4, A2_5, A2_6))
    return Z1, A1, Z2, A2


def one_hot(Y:np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, config.MAX_POSITION))
    one_hot_Y[np.arange(Y.size), Y-1] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def mask(Y):
    mask_Y = np.zeros((Y.shape[1],(config.MAX_POSITION)*Y.shape[0])).T
    mask_indices = np.argwhere(Y != 0)  # Get indices where Y is not zero
    for i, j in mask_indices:
        mask_Y[config.MAX_POSITION * i:config.MAX_POSITION * i + config.MAX_POSITION, j] = 1  # Set consecutive indices to 1
    return mask_Y


def backward_prop(Z1:np.ndarray, A1:np.ndarray, Z2:np.ndarray, A2:np.ndarray, W2:np.ndarray, X:np.ndarray, Y:np.ndarray):
    m = Y.shape[1]
    one_hot_Y = np.concatenate((one_hot(Y[0]),one_hot(Y[1]),one_hot(Y[2]),one_hot(Y[3]),one_hot(Y[4]),one_hot(Y[5])), axis=0)
    #print(f" shape : {one_hot_Y.shape}, {one_hot(Y[0]).shape}")
    #print(one_hot_Y[24:48])
    dZ2 = (A2 - one_hot_Y) * mask(Y)
    #print("A2")
    #print(np.max(A2[0:25]))
    #print("dZ2")
    #print(dZ2[24:48])
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * reLu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2


def updata_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2, Y):
    return np.array((np.argmax(A2[0:config.MAX_POSITION*1], 0)+1,
                           np.argmax(A2[config.MAX_POSITION*1:config.MAX_POSITION*2], 0)+1,
                           np.argmax(A2[config.MAX_POSITION*2:config.MAX_POSITION*3], 0)+1,
                           np.argmax(A2[config.MAX_POSITION*3:config.MAX_POSITION*4], 0)+1,
                           np.argmax(A2[config.MAX_POSITION*4:config.MAX_POSITION*5], 0)+1,
                           np.argmax(A2[config.MAX_POSITION*5:config.MAX_POSITION*6], 0)+1))


def get_accuracy(pred, Y):
    for n in range(20):
        #print(f"pred{n} : {pred[0:6,n]}")
        print(f"pred{str(n).zfill(2)} : [", end='')
        for i in range(6):
            print(gray_str(pred[i][n] if Y[i][n]==0 else (green_str((pred[i][n])) if pred[i][n]==Y[i][n] else red_str(pred[i][n]))), end= ' ')
        print(']', end='  ')
        #print(f"gt{n} : {Y[0:6,n]}")
        print()
    mask = Y != 0
    return np.sum((pred == Y) & mask) / np.sum(mask)


def gradient_descent(X,Y, iterations, alpha):
    dynLinePlot = DynamicLinePlot('Accuracy vs. Iteration', 'Iteration', 'Accuracy')
    num_outputs = (config.Y_END - config.Y_START) * (config.MAX_POSITION)
    W1, b1, W2, b2 = init_params(X.shape[0], num_outputs, config.SIZE_LAYER1)
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updata_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % config.SHOW_ACCURACY_TIME == 0):
            acc = get_accuracy(get_predictions(A2,Y), Y)
            print(f"Iterations: {i}")
            print(f"Accuracy: {acc}")
            dynLinePlot.update(i, acc)
    return W1, b1, W2, b2, dynLinePlot