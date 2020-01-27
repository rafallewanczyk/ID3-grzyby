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
from random import shuffle

def main():
    parser = argparse.ArgumentParser(description="ID3 algorithm")
    parser.add_argument('-d', type=str, help="Data file to import")
    parser.add_argument('-k', type=int, help='k-fold cross validation')

    args = parser.parse_args()

    if args.d is None or args.k is None:
        print('Missing input file name or k')
        return

    data, names = read_data(args.d)
    shuffle(data)
    normal_errors, normal_avg = cross_validation(data, names, args.k, True)
    roulette_errors, roulette_avg = cross_validation(data, names, args.k, False)
    print(f'Normal tree === errors ratio: {normal_errors}, avarage: {normal_avg}')
    print(f'Roulette tree === errors ratio: {roulette_errors}, avarage: {roulette_avg}')
    



if __name__ == '__main__':
    main()
