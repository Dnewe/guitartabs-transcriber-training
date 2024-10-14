import math
import pandas as pd
import numpy as np
import config


def read_data(datacsv_path:str):
    # config parameters
    y_start = min(config.Y_MULTI_START if config.MULTICLASS_LABELS else math.inf, config.Y_SINGLE_START if config.SINGLECLASS_LABELS else math.inf)
    y_end = max(config.Y_MULTI_END if config.MULTICLASS_LABELS else 0, config.Y_SINGLE_END if config.SINGLECLASS_LABELS else 0)
    x_start = config.X_STARTINDEX

    # read data
    df = pd.read_csv(datacsv_path)
    data = np.array(df)
    m, n = data.shape

    # shuffle data
    np.random.shuffle(data)

    # separate data
        # dev
    data_dev = data[0:((int) ((1-config.TRAIN_PROP)*m))].T
    Y_dev = data_dev[y_start:y_end]
    X_dev = data_dev[x_start:n]
        # train
    data_train = data[((int) ((config.TRAIN_PROP)*m)):m].T
    Y_train = data_train[y_start:y_end]
    X_train = data_train[x_start:n]

    return Y_dev, X_dev, Y_train, X_train



"""
def read_data_(datacsv_path:str):
    df = pd.read_csv(datacsv_path)
    rawdata = np.array(df)
    m, n = rawdata.shape

    # shuffle data
    np.random.shuffle(rawdata)

    # creating array of size m and 6 times size n except for label string and fret
    data: np.ndarray = np.zeros((m, n*6))  

    for i in range(m):
        # add strings and frets labels
        stringslabel = eval(rawdata[i][0])
        fretslabel = eval(rawdata[i][1])
        for j,string in enumerate(stringslabel):
            data[i][(int) (string)-1] = 1
            data[i][(int) (string)-1 + 6] = (int) (fretslabel[j])

        # add pitches
        for j in range(2,n):
            pitches = eval(rawdata[i][j])
            for k in range(6):
                data[i][12+(j-2)*6+k] = pitches[k] if len(pitches)>k else 0
    
    data_dev = data[0:((int) ((1-config.TRAIN_PROP)*m))].T
    Y_dev = data_dev[0:6]
    X_dev = data_dev[12:n*6]

    data_train = data[((int) ((config.TRAIN_PROP)*m)):m].T
    Y_train = data_train[0:6]
    X_train = data_train[12:n*6]

    return Y_dev, X_dev, Y_train, X_train"""