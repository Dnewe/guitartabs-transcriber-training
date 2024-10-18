import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple,Dict
import pandas as pd


class DynamicLinePlot:
    #x:List[float]
    #y:List[float]
    data:pd.DataFrame
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
        self.data = {'x':[],'y':{}}
        self.data = pd.DataFrame({'x':[],'y':[],'id':[]})
        #self.x = []
        #self.y = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
    
    def update(self, x_value:float, y_values:Dict[str,float]):
        #self.x.append(x_value)
        #self.y.append(y_value)
        for id, y in y_values.items():
            self.data.loc[len(self.data)] = [x_value, y, id]

        '''self.data['x'].append(x_value)
        for id, value in y_values.items():
            self.data['y'][id].append(value)'''
        self.ax.clear()
        sns.lineplot(data=self.data, x='x', y='y', hue='id', marker=None, ax=self.ax) # ,x=self.x, y=self.y
        self.ax.set_title(self.title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel(self.xlabel, fontsize=14)
        self.ax.set_ylabel(self.ylabel, fontsize=14)
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(min(self.data['x']), max(self.data['x']))
        plt.draw()
        plt.pause(0.1)

    def save(self, path:str):
        self.fig.savefig(path)
