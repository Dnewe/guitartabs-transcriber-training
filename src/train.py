import os
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from read_data import read_data
from gradient_descent import gradient_descent, forward_prop, get_accuracy, get_predictions
from write_data import write_data


def run(args):
    data_path = os.path.join(args.input, "data.csv")
    metadata_path = os.path.join(args.input, "metadata.json")
    outdir_path = args.output

    print("Reading metadata..")
    dataconfig = DataConfig()
    dataconfig.initialize_from_json(metadata_path)
    modelconfig = ModelConfig()
    modelconfig.initialize()

    print("Reading data..")
    Y_dev, X_dev, Y_train, X_train = read_data(data_path)

    print("Training..")
    w1, b1, w2, b2, dynLinePlot = gradient_descent(X_train, Y_train, modelconfig.ITERATIONS, modelconfig.ALPHA)

    print("Finished training")
    _,_,_,a2 = forward_prop(w1,b1,w2,b2, X_dev)
    acc = get_accuracy(get_predictions(a2), Y_dev)
    print(f"Accuracy on test set: {round(acc,4)*100}% (size of test set: {Y_dev.shape[1]})")

    write_data(data_path, outdir_path, w1, b1, w2, b2, acc, dynLinePlot)