import networkx as net
from math import log2
import matplotlib.pyplot as plt
from pandas import DataFrame


class DecisionTree:
    graph = net.DiGraph()
    counter = 0

    def __init__(self, train_data):
        self.data = train_data

    def build_tree(self, data_interval, parent = None, attribute_state = None):
        print(f"{parent}, {attribute_state}")
        print(data_interval)
        # print(self.total_entropy(data_interval))
        classes = data_interval['class'].unique().tolist()
        if len(classes) == 1 :

            print(f"lisc polaczenie {parent} -- {attribute_state} --> {classes[0]}")
            self.graph.add_node(classes[0]+'.'+ str(self.counter))
            if parent: self.graph.add_edge(parent, classes[0]+'.'+str(self.counter), user_data= attribute_state)
            self.counter+= 1
            return

        attribute = (self.best_attribute(data_interval, self.total_entropy(data_interval)), self.counter)
        self.counter += 1

        self.graph.add_node(attribute)
        if parent:
            self.graph.add_edge(parent, attribute, user_data = attribute_state)
            print(f"polaczenie {parent} -- {attribute_state} --> {attribute}")

        for state in data_interval[attribute[0]].unique():
           self.build_tree(data_interval.loc[data_interval[attribute[0]] == state], attribute, state)




    def total_entropy(self, data_interval):
        entropy = 0
        for x in data_interval['class'].unique():
            ratio = data_interval.loc[data_interval['class'] == x].shape[0] / data_interval.shape[0]
            entropy += -ratio * log2(ratio)
        return entropy

    def best_attribute(self, data_interval, entropy):
        best_parameter, best_gain = "", 0
        for c in data_interval.columns:
            if c == 'class': break
            information = 0
            for state in data_interval[c].unique():
                df = data_interval.loc[data_interval[c] == state]
                # print(df)
                # print(self.total_entropy(df))
                information += df.shape[0] / data_interval.shape[0] * self.total_entropy(df)
            print(f"parameter : {c} information : {information} gain: {entropy - information}")
            if entropy - information > best_gain:
                best_gain = entropy - information
                best_parameter = c
        print(f"najlepszy parametr {best_parameter} \n")
        return best_parameter

    def draw_tree(self):
        # self.graph.add_edge("Ammo", "Defend")
        print(self.graph.nodes)

        pos = net.spring_layout(self.graph)
        net.draw_networkx(self.graph, pos = pos)
        labels = net.get_edge_attributes(self.graph, 'user_data')
        net.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels)
        plt.show()

        # df = data_interval.loc[data_interval['State'] == 'Healthy']
        # print(df)
        # print(self.total_entropy(df))
        #
        # df = data_interval.loc[data_interval['State'] == 'Hurt']
        # print(df)
        # print(self.total_entropy(df))
