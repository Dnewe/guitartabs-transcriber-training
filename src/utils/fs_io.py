import os
import pandas as pd
import numpy as np
from typing import List


def create_dir(dirpath:str) -> None:
    try:
        os.mkdir(dirpath)
    except Exception as e:
        print(f"Directory at path: {dirpath} already exists.")


def write_ndarray_to_csv(csvpath: str, data: np.ndarray) -> None:
    np.savetxt(csvpath, data, delimiter=',', fmt='%f')


def write_lines_to_textfile(filepath:str, lines:List[str]) -> None:
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(line + '\n')