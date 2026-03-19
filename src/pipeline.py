import os
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from read_data import read_data
from train import *
from write_data import write_data


def run(args):
    data_path = os.path.join(args.input, "data.csv")
    metadata_path = os.path.join(args.input, "metadata.json")
    outdir_path = args.output

    print("Reading metadata..")
    dataconfig = DataConfig()
    dataconfig.load_json(metadata_path)
    modelconfig = ModelConfig()
    modelconfig.load_default()

    print("Reading data..")
    X_train, y_train, X_val, y_val, X_test, y_test, train_mean, train_std = read_data(data_path)
    modelconfig.MEAN = train_mean
    modelconfig.STD = train_std

    print("Training..")
    model, dynLinePlot = train(X_train, y_train, X_val, y_val, modelconfig.EPOCHS)

    print("Finished training")
    acc, loss = test(model, X_test, y_test)

    print("Write Data")
    write_data(data_path, outdir_path, model, acc, loss, dynLinePlot)