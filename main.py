from id3tree import tree
from algorithms.decisionTree import DecisionTree
from pandas import DataFrame
import numpy as np
import networkx as net
import matplotlib.pyplot as plt
from algorithms.rouletteDecisionTree import RouletteDecisionTree
import argparse
from data.reader import read_data, convert_to_frame
from algorithms.cross_validation import cross_validation


def main():
    parser = argparse.ArgumentParser(description="ID3 algorithm")
    parser.add_argument('-d', type=str, help="Data file to import")
    parser.add_argument('-k', type=int, help='k-fold cross validation')

    args = parser.parse_args()

    if args.d is None or args.k is None:
        print('Missing input file name or k')
        return

    data, names = read_data(args.d)
    normal_errors, normal_avg = cross_validation(data, names, arg.k, True)
    roulette_errors, roulette_avg = cross_validation(data, names, args.k, True)
    print(f'Normal tree === errors ratio: {normal_errors}, avarage: {normal_avg}')
    print(f'Roulette tree === errors ratio: {roulette_errors}, avarage: {roulette_avg}')
    



if __name__ == '__main__':
    main()

# X = np.array([[45, "male", "private", "m"],
#               [50, "female", "private", "m"],
#               [61, "other", "public", "b"],
#               [40, "male", "private", "none"],
#               [34, "female", "private", "none"],
#               [33, "male", "public", "none"],
#               [43, "other", "private", "m"],
#               [35, "male", "private", "m"],
#               [34, "female", "private", "m"],
#               [35, "male", "public", "m"],
#               [34, "other", "public", "m"],
#               [34, "other", "public", "b"],
#               [34, "female", "public", "b"],
#               [34, "male", "public", "b"],
#               [34, "female", "private", "b"],
#               [34, "male", "private", "b"],
#               [34, "other", "private", "b"]])
#
# y = np.array(["(30k,38k)",
#               "(30k,38k)",
#               "(30k,38k)",
#               "(13k,15k)",
#               "(13k,15k)",
#               "(13k,15k)",
#               "(23k,30k)",
#               "(23k,30k)",
#               "(23k,30k)",
#               "(15k,23k)",
#               "(15k,23k)",
#               "(15k,23k)",
#               "(15k,23k)",
#               "(15k,23k)",
#               "(23k,30k)",
#               "(23k,30k)",
#               "(23k,30k)"])
#
# Z = np.array([["rainy", "hot", "high", "false"],
#               ["rainy", "hot", "high", "false"],
#                ["rainy", "hot", "high", "true"],
#               ["overcast", "hot", "high", "false"],
#               ["sunny", "mild", "high", "false"],
#               ["sunny", "cool", "normal", "false"],
#               ["sunny", "cool", "normal", "true"],
#               ["overcast", "cool", "normal", "true"],
#               ["rainy", "mild", "high", "false"],
#               ["rainy", "cool", "normal", "false"],
#               ["sunny", "mild", "normal", "false"],
#               ["rainy", "mild", "normal", "true"],
#               ["overcast", "mild", "high", "true"],
#               ["overcast", "hot", "normal", "false"],
#               ["sunny", "mild", "high", "true"]])
# t = np.array(["no", "yes", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"])
# data = DataFrame(Z,columns=["Outlook", "Temp", "Humidity", "Windy"])
# data['class'] = t
#
#
# Z_validate = np.array([["rainy", "hot", "high", "false"],
#               ["rainy", "hot", "high", "false"],
#               ["rainy", "hot", "high", "true"],
#               ["overcast", "hot", "high", "false"],
#               ["sunny", "mild", "high", "false"],
#               ["sunny", "cool", "normal", "false"],
#               ["sunny", "cool", "normal", "true"],
#               ["overcast", "cool", "normal", "true"],
#               ["rainy", "mild", "high", "false"],
#               ["rainy", "cool", "normal", "false"],
#               ["sunny", "mild", "normal", "false"],
#               ["rainy", "mild", "normal", "true"],
#               ["overcast", "mild", "high", "true"],
#               ["overcast", "hot", "normal", "false"],
#               ["sunny", "mild", "high", "true"]])
# t_validate = np.array(["no", "yes", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"])
# data_val = DataFrame(Z_validate,columns=["Outlook", "Temp", "Humidity", "Windy"])
# data_val['class'] = t_validate
# # data = DataFrame(X, columns = ["age", "gender", "sector", "degree"])
# # data["class"] = y
#
# tree = DecisionTree(data)
# tree.build_tree(data)
# tree.validate(data_val)
# tree.draw()

# query = DataFrame([["sunny", "hot", "high", "true"]], columns=["Outlook", "Temp", "Humidity", "Windy"])
# print(tree.predict(query))






























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
# print(frame_data)
# tree = DecisionTree(frame_data)
# tree.build_tree(frame_data)
# tree.draw()

