import numpy as np
from pandas import DataFrame

def read_data(data_file, train_num = 8416):
    train_data = []
    validation_data = []
    names = []
    with open(data_file, 'r') as reader:
        names = reader.readline().strip().split(',')

        lines_number = 0
        for line in reader.readlines():
            if lines_number < train_num : train_data.append(np.array(line.strip().split(',')))
            else : validation_data.append(np.array(line.strip().split(',')))
            lines_number += 1
    

    return DataFrame(train_data, columns=names), DataFrame(validation_data, columns=names)
     