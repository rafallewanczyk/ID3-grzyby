from .decisionTree import DecisionTree
from .rouletteDecisionTree import RouletteDecisionTree
from pandas import DataFrame
from statistics import mean

def cross_validation(data, names, k, n):
    objects_number = len(data)
    lines_step = objects_number / k
    validation_start = 0
    validation_stop = lines_step
    error_ratios = []
    tree = None

    for i in range(0, k):
        train = []
        validation = []
        line_number = 1

        for line in data:
            if line_number > validation_start and line_number < validation_stop:
                validation.append(line)
            else:
                train.append(line)
            line_number += 1

        train_frame = DataFrame(train, columns=names)
        validation_frame = DataFrame(validation, columns=names)
        validation_start += lines_step
        validation_stop += lines_step

        tree = DecisionTree(train_frame) if n else RouletteDecisionTree(train_frame)
        tree.build_tree(train_frame)
        error = tree.validate(validation_frame)
        error_ratios.append(error)

    #tree.draw()
    return error_ratios, mean(error_ratios)