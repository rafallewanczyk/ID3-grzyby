from id3tree import tree
from algorithms.decisionTree import DecisionTree
from pandas import DataFrame
import numpy as np
import networkx as net
import matplotlib.pyplot as plt


X = np.array([[45, "male", "private", "m"],
              [50, "female", "private", "m"],
              [61, "other", "public", "b"],
              [40, "male", "private", "none"],
              [34, "female", "private", "none"],
              [33, "male", "public", "none"],
              [43, "other", "private", "m"],
              [35, "male", "private", "m"],
              [34, "female", "private", "m"],
              [35, "male", "public", "m"],
              [34, "other", "public", "m"],
              [34, "other", "public", "b"],
              [34, "female", "public", "b"],
              [34, "male", "public", "b"],
              [34, "female", "private", "b"],
              [34, "male", "private", "b"],
              [34, "other", "private", "b"]])

y = np.array(["(30k,38k)",
              "(30k,38k)",
              "(30k,38k)",
              "(13k,15k)",
              "(13k,15k)",
              "(13k,15k)",
              "(23k,30k)",
              "(23k,30k)",
              "(23k,30k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(15k,23k)",
              "(23k,30k)",
              "(23k,30k)",
              "(23k,30k)"])

Z = np.array([["Healthy", "In Cover", "With Ammo"],
               ["Hurt", "In Cover", "With Ammo"],
              ["Healthy", "In Cover", "Empty"],
              ["Hurt", "In Cover", "Empty"],
              ["Hurt", "Exposed", "With Ammo"]])
t = np.array(["Attack", "Attack", "Defend", "Defend", "Defend"])
# data = DataFrame(Z,columns=["State", "Cover", "Ammo"])
# data['class'] = t
data = DataFrame(X, columns = ["age", "gender", "sector", "degree"])
data["class"] = y

tree = DecisionTree(data)
tree.build_tree(data)
tree.draw_tree()






























# file = open("data/agaricus-lepiota-small.data", "r")
# data_class, data = [], []
# for x in file:
#     data_class.append(x[0])
#     data.append(x[2:-1])
# file.close()
#
# y = np.array([np.array(xi.split(',')) for xi in data])
# frame_data = DataFrame(y, columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
#                                    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
#                                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
#                                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
#                                    'spore-print-color', 'population', 'habitat'])
# frame_data_class = DataFrame([np.array(xi) for xi in data_class])
# frame_data['class'] = frame_data_class
# tree = DecisionTree(frame_data)
# tree.build_tree(frame_data)
# tree.draw_tree()
# tree = DecisionTree(frame_data)
# tree.buildTree()

# tree = tree.MyDecisionTreeClassifier()
# tree.fit(frame_data,frame_data_class)
# tree.print_tree()


