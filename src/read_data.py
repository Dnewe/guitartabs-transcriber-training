from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
import pandas as pd
import numpy as np

dataconfig = DataConfig()
modelconfig = ModelConfig()


def read_data(datacsv_path:str):
    # config parameters
    y_start = min(dataconfig.Y_STRINGS)
    y_end = max(dataconfig.Y_STRINGS)
    x_start = min(dataconfig.X_DATA)

    # read data
    df = pd.read_csv(datacsv_path)
    data = np.array(df)
    m, n = data.shape

    # shuffle data
    np.random.shuffle(data)
    print(data.shape)

    # separate data
        # dev ((int) (m-(config.TRAIN_PROP)*m)):m
    data_dev = data[((int) ((modelconfig.TRAIN_PROP)*m)): m].T# [0:((int) ((1-config.TRAIN_PROP)*m))].T
    Y_dev = data_dev[y_start:y_end+1]
    X_dev = data_dev[x_start:n]
        # train 0:((int) ((1-config.TRAIN_PROP)*m))
    data_train = data[0 : ((int) ((modelconfig.TRAIN_PROP)*m))].T # ((int) (m-(config.TRAIN_PROP)*m)):m].T
    Y_train = data_train[y_start:y_end+1]
    X_train = data_train[x_start:n]

    # adujst X
    min_value = max(np.min(X_train[X_train>0]), np.min(X_dev[X_dev>0]))
    X_dev += -min_value+1
    X_train += -min_value+1

    print(Y_train.shape)

    return Y_dev, X_dev, Y_train, X_train
