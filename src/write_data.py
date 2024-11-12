import os
from config.dataConfig import DataConfig
from config.modelConfig import ModelConfig
from utils.fs_io import write_json, create_dir, write_ndarray_to_csv, write_lines_to_textfile

dataconfig = DataConfig()
modelconfig = ModelConfig()


def write_metadata(modeldir:str):
    metadata = {'model': modelconfig.to_dict(),
                'data': dataconfig.to_dict()}
    write_json(os.path.join(modeldir, 'metadata.json'), metadata)


def create_modeldir(input:str, outdir:str):
    modeldir = os.path.join(outdir, "model_" + os.path.basename(os.path.dirname(input)))
    create_dir(modeldir)
    return modeldir


def write_modelcsvfiles(modeldir:str, **kwargs):
    for key, value in kwargs.items():
        print(f"shape {key}: {value.shape}")
        print(value[0][0])
        write_ndarray_to_csv(os.path.join(modeldir, f"{key}.csv"), value)


def write_infotxt(modeldir:str, acc:float):
    configtxt = [f'Accuracy on test data: {round(acc,4)*100}%']
    write_lines_to_textfile(os.path.join(modeldir, 'info.txt'), configtxt)


def write_data(data_path, outdir_path, w1, b1, w2, b2, acc, dynLinePlot): 
    modeldir = create_modeldir(data_path, outdir_path)

    print('Saving Image..')
    dynLinePlot.save(os.path.join(modeldir,'acc_vs_it.png'))

    print("Writing model data..")
    write_modelcsvfiles(modeldir, W1=w1, b1=b1, W2=w2, b2=b2)
    write_metadata(modeldir)
    write_infotxt(modeldir, acc)