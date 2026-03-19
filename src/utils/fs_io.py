import os
import numpy as np
import json
from typing import List,Dict


def create_dir(dirpath:str) -> None:
    try:
        os.mkdir(dirpath)
    except Exception as e:
        print(f"Directory at path: {dirpath} already exists.")


def write_ndarray_to_csv(csvpath: str, data: np.ndarray) -> None:
    data = data.astype(np.float64)
    np.savetxt(csvpath, data, delimiter=',', fmt='%.10f')

"""def write_ndarray_to_csv(csvpath: str, data: np.ndarray) -> None:
    df = pd.DataFrame(data)
    df.to_csv(csvpath, index=False, float_format='%.10f')"""


def write_lines_to_textfile(filepath:str, lines:List[str]) -> None:
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def write_json(filepath:str, data:Dict) -> None:
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def read_json(filepath:str):
    with open(filepath,'r') as file:
        data = json.load(file)
    return data