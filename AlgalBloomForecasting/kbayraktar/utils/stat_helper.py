import importlib
import rasterio
import numpy as np
import torch

rionegrodata = importlib.import_module("rionegrodata")

"""
Used for the construction of Confusion Matrics. Could be used further for data analysis.
"""
class DataAnalysis:
    def __init__(self):
        self.conf = []
        self.num = 0

    def conf_append(self,input):
        self.conf.append(input)
        self.num+=1

    def conf_calculate(self):
        return torch.div(torch.stack(self.conf).sum(dim=0),self.num)


