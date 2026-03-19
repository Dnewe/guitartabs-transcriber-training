import numpy as np
from utils.plots.dynamicLinePlot import DynamicLinePlot
from neuralnet.neuralNetwork import NeuralNetwork
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from config.displayConfig import DisplayConfig
from neuralnet.utils import *
from utils.color import green_str, red_str, gray_str, yellow_str
import time
import copy

dataconfig = DataConfig()
modelconfig = ModelConfig()
displayconfig = DisplayConfig()

display_delay = 1


def get_eta(start_time, end_time, max_epochs, epoch):
    hours = (int)((end_time-start_time)/display_delay*(max_epochs-epoch))//3600
    minutes = (int)(((end_time-start_time)/display_delay*(max_epochs-epoch))//60)%60
    seconds = round(((end_time-start_time)/display_delay*(max_epochs-epoch))%60)
    return hours, minutes, seconds


def train(X_train:np.ndarray, y_train:np.ndarray,
          X_val:np.ndarray, y_val:np.ndarray, 
          epochs:int):
    
    dynLinePlot = DynamicLinePlot('Variables vs. Epoch', 'Epoch', 'Proportion')

    model = NeuralNetwork(modelconfig.LAYERS, modelconfig.ALPHA, modelconfig.MOMENTUM)

    patience = modelconfig.EARLYSTOP_PATIENCE
    no_improvement = 0
    best_epoch = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)

    start_time = time.time()
    for i in range(epochs):
        for j in range(0, X_train.shape[0], modelconfig.BATCH_SIZE):
            X = X_train[j:j+modelconfig.BATCH_SIZE]
            y = y_train[j:j+modelconfig.BATCH_SIZE]
            model.forward(X)
            model.backward(y)
        
        # train
        model.forward(X_train)
        acc_train_n = get_accuracy_note(y_train, model.output)
        acc_train_b = get_accuracy_beat(y_train, model.output)
        loss_train = loss_BCE(y_train, model.output)

        # validation
        model.forward(X_val)
        acc_val_n = get_accuracy_note(y_val, model.output)
        acc_val_b = get_accuracy_beat(y_val, model.output)
        loss_val = loss_BCE(y_val, model.output)

        # best model / patience
        if loss_val < best_loss:
            no_improvement = 0
            best_loss = loss_val
            best_epoch = i
            best_model = copy.deepcopy(model)
        else:  
            no_improvement +=1
            print(f"no improvement {no_improvement}")
            if no_improvement > patience: 
                print("Patience reached, stopping early.")
                break

        # time
        end_time = time.time()
        hours, minutes, seconds = get_eta(start_time, end_time, epochs, i)

        print(f"\n\nEpoch: {i}")
        print(f"Time per epoch: {round((end_time-start_time)/display_delay,3)}s ; ETA: {hours}h {minutes}m {seconds}s")
        print(f"TRAIN: Accuracy [note|beat]: [{yellow_str(round(acc_train_n,4))} | {yellow_str(round(acc_train_b,4))}]  | Loss: {yellow_str(round(loss_train,4))}")
        print(f"VAL  : Accuracy [note|beat]: [{yellow_str(round(acc_val_n,4))} | {yellow_str(round(acc_val_b,4))}]  | Loss: {yellow_str(round(loss_val,4))}")
        dynLinePlot.update(i, {'acc_train':acc_train_n,'acc_val':acc_val_n, 'loss_train':loss_train, 'loss_val':loss_val}, (best_epoch, best_loss))
        start_time = time.time()
    
    return best_model, dynLinePlot


def test(model, X:np.ndarray, y:np.ndarray):
    model.forward(X)
    acc_n = get_accuracy_note(y, model.output)
    acc_b = get_accuracy_beat(y, model.output)
    loss = loss_BCE(y, model.output)
    print(f"TEST : Accuracy [note|beat]: [{yellow_str(round(acc_n,4))} | {yellow_str(round(acc_b,4))}]  | Loss: {yellow_str(round(loss,4))}")
    return acc_b, loss