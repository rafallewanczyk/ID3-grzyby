import networkx as net
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
from math import log
import matplotlib.pyplot as plt
from pandas import DataFrame


class DecisionTree():


    def __init__(self, train_data):
        self.data = train_data
        self.graph = net.DiGraph()
        self.counter = 0
        self.labels = {}

    def build_tree(self, data_interval, parent=None, attribute_state=None):
        print(f"{parent}, {attribute_state}")
        print(data_interval)
        # print(self.total_entropy(data_interval))
        classes = data_interval['class'].unique().tolist()
        # if len(classes) == 1:
        #
        #     print(f"lisc polaczenie {parent} -- {attribute_state} --> {classes[0]}")
        #     self.graph.add_node((classes[0], self.counter))
        #     if parent: self.graph.add_edge(parent, (classes[0], self.counter), user_data=attribute_state)
        #     self.counter += 1
        #     return

        attribute = (self.best_attribute(data_interval, self.total_entropy(data_interval)), self.counter)
        self.counter += 1

        if attribute[0] == "":
            s = ""
            for state in classes:
                counter = 0
                for x in data_interval['class']:
                    if x == state:
                        counter += 1
                s += (f"{state}:{counter / len(data_interval['class'])}")
            print(f"lisc polaczenien rozne {parent} -- {attribute_state} --> {s}")
            self.graph.add_node((s, self.counter))
            self.labels[(s, self.counter)] = s
            if parent: self.graph.add_edge(parent, (s, self.counter), user_data=attribute_state)
            self.counter += 1
            return

        self.graph.add_node(attribute)
        self.labels[attribute] = attribute[0]
        if parent:
            self.graph.add_edge(parent, attribute, user_data=attribute_state)
            print(f"polaczenie {parent} -- {attribute_state} --> {attribute}")
        else:
            self.root = attribute

        for state in data_interval[attribute[0]].unique():
            self.build_tree(data_interval.loc[data_interval[attribute[0]] == state], attribute, state)

    def total_entropy(self, data_interval):
        entropy = 0
        number_of_choices = len(data_interval['class'].unique())
        if number_of_choices == 1:
            return 0
        for x in data_interval['class'].unique():
            ratio = data_interval.loc[data_interval['class'] == x].shape[0] / data_interval.shape[0]
            entropy += -ratio * log(ratio, number_of_choices)
        return entropy

    def information_gains(self, data_interval, entropy):
        gains = {}
        for c in data_interval.columns:
            if c == 'class': continue
            information = 0
            for state in data_interval[c].unique():
                df = data_interval.loc[data_interval[c] == state]
                information += df.shape[0] / data_interval.shape[0] * self.total_entropy(df)
                gains[c] = entropy - information
            print(f"parameter : {c} information : {information} gain: {entropy - information}")

        return gains

    def best_attribute(self, data_interval, entropy):
        print(f'entropia: {entropy}')
        best_parameter, best_gain = '', 0
        information_gains = self.information_gains(data_interval, entropy)
        for parameter in information_gains.keys():
            if information_gains[parameter] > best_gain:
                best_gain = information_gains[parameter]
                best_parameter = parameter
        print(f"najlepszy parametr {best_parameter} \n")
        return best_parameter

    def predict(self, data_interval, attribute=None):
        if attribute == None: attribute = self.root
        if len(list(self.graph.neighbors(attribute))) == 0:
            return attribute[0]

        for node in list(self.graph.neighbors(attribute)):
            if self.graph.get_edge_data(attribute, node)['user_data'] == data_interval[attribute[0]][0]:
                return self.predict(data_interval, node)

    def draw(self):
        #LINUX
        write_dot(self.graph, 'test.dot')
        pos = graphviz_layout(self.graph, prog='dot')

        #WINDOWS
        # pos = net.spring_layout(self.graph)

        net.draw_networkx(self.graph, pos=pos, with_labels=True, labels=self.labels)
        labels = net.get_edge_attributes(self.graph, 'user_data')
        net.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=labels)
        plt.show()

    def validate(self, validation_data):
        record_list = validation_data.values.tolist()
        columns = validation_data.columns.tolist()
        print("walidacja")
        error_ratio = 0
        for row in record_list:
            answer = self.predict(DataFrame([row[1::]], columns = columns[1::]))
            try:
                error_ratio += answer.split(":")[0] != row[0]
                if answer.split(":")[0] != row[0] :
                    print(row)
            except AttributeError:
                error_ratio += 1
        print(f"error ratio: {error_ratio/len(record_list)}")
        return error_ratio / len(record_list)

    