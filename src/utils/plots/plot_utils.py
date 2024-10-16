import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Tuple




def init_lineplot(title: str, xlabel: str, ylabel: str, ylim:Tuple[float,float] = (0,1)):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.ylim(ylim)

    ax = sns.lineplot(x=[], y=[], marker=None)

    return ax