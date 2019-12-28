import networkx as net
from math import log2
from pandas import DataFrame
class DecisionTree:
    graph = net.Graph()

    def __init__(self, train_data):
        self.data = train_data

    def buildTree(self, data_interval):
        print(data_interval)
        print(self.total_entropy(data_interval))

    def total_entropy(self, data_interval):
        entropy = 0
        for x in data_interval['class'].unique():
            ratio = data_interval.loc[data_interval['class'] == x].shape[0] / data_interval.shape[0]
            entropy += -ratio * log2(ratio)
        return entropy

    def entropy(self, data_interval, column):
        entropy = 0
        for x in data_interval[column].unique():
            ratio = data_interval.loc[data_interval['class'] == x].shape[0] / data_interval.shape[0]
            entropy += -ratio * log2(ratio)
        return entropy
