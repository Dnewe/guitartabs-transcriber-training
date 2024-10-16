import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple,List


class DynamicLinePlot:
    x:List[float]
    y:List[float]

    title:str
    xlabel:str
    ylabel:str
    ylim:Tuple[float,float]
    fig: plt.Figure
    ax: plt.Axes

    def __init__(self, title: str, xlabel: str, ylabel: str, ylim:Tuple[float,float] = (0,1)) -> None:
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.ylim = ylim
        self.x = []
        self.y = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
    
    def update(self, x_value:float, y_value:float):
        self.x.append(x_value)
        self.y.append(y_value)

        self.ax.clear()
        sns.lineplot(x=self.x, y=self.y, marker=None, ax=self.ax)
        self.ax.set_title(self.title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel(self.xlabel, fontsize=14)
        self.ax.set_ylabel(self.ylabel, fontsize=14)
        self.ax.set_ylim(0, 1)  # Keep the y-axis between 0 and 1
        self.ax.set_xlim(min(self.x), max(self.x))

        plt.draw()
        plt.pause(0.1)

    def save(self, path:str):
        self.fig.savefig(path)
