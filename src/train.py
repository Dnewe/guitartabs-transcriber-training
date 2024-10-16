import config
import os
from read_data import read_data
from gradient_descent import gradient_descent, forward_prop, get_accuracy, get_predictions
from utils.fs_io import create_dir, write_ndarray_to_csv, write_lines_to_textfile


def create_modeldir(input:str, outdir:str):
    modeldir = os.path.join(outdir, "model_" + os.path.basename(input).replace('.csv','').replace('data_',''))
    create_dir(modeldir)
    return modeldir


def write_modelcsvfiles(modeldir:str, **kwargs):
    for key, value in kwargs.items():
        print(f"shape {key}: {value.shape}")
        write_ndarray_to_csv(os.path.join(modeldir, f"{key}.csv"), value)


def write_infotxt(modeldir:str, acc:float):
    configtxt = [f'Train data: {config.TRAIN_PROP*100}%',
                 f'',
                 f'Iterations: {config.ITERATIONS}',
                 f'Alpha: {config.ALPHA}',
                 f'',
                 f'Layer1 size: {config.SIZE_LAYER1}',
                 f'',
                 f'Accuracy on test data: {round(acc,4)*100}%']
    write_lines_to_textfile(os.path.join(modeldir, 'info.txt'), configtxt)


def run(args):
    datacsv_path = args.input
    outdir_path = args.output

    print("Reading data..")
    Y_dev, X_dev, Y_train, X_train = read_data(datacsv_path)

    print("Training..")
    w1, b1, w2, b2, dynLinePlot = gradient_descent(X_train, Y_train, config.ITERATIONS, config.ALPHA)

    print("Finished training")
    _,_,_,a2 = forward_prop(w1,b1,w2,b2, X_dev)
    acc = get_accuracy(get_predictions(a2, Y_dev), Y_dev)
    print(f"Accuracy on test set: {round(acc,4)*100}% (size of test set: {Y_dev.shape[1]})")

    modeldir = create_modeldir(datacsv_path, outdir_path)

    print('Saving Image..')
    dynLinePlot.save(os.path.join(modeldir,'acc_vs_it.png'))

    print("Writing model data..")
    write_modelcsvfiles(modeldir, W1=w1, b1=b1, W2=w2, b2=b2)
    write_infotxt(modeldir, acc)