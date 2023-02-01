import matplotlib.pyplot as plt 
import os
from os.path import join, isfile
from os import listdir
import csv 
import pandas as pd
import numpy as np

def draw_performance(result_folder_path, image_folder_path):
    only_csv_files = [f for f in listdir(result_folder_path) if isfile(join(result_folder_path, f)) and f[-4:] == ".csv"]
    only_csv_files = only_csv_files[0:1]

    for i, file in enumerate(only_csv_files):
        data = pd.read_csv(os.path.join(result_folder_path, file))
        if i == 0:
            x = np.array(list(map(int, data['Iteration'])))
            queried_per = np.array(list(map(float, data['Queried X/ All X'])))
            queried_misclassified_per = np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
            detected_pattern_per = np.array(list(map(float, data['Detected patterns / All patterns'])))
        else:   
            x += np.array(list(map(int, data['Iteration'])))
            queried_per += np.array(list(map(float, data['Queried X/ All X'])))
            queried_misclassified_per += np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
            detected_pattern_per += np.array(list(map(float, data['Detected patterns / All patterns'])))
    x = x / float(len(only_csv_files))
    queried_per = queried_per / len(only_csv_files) * 100
    queried_misclassified_per = queried_misclassified_per / len(only_csv_files) * 100
    detected_pattern_per = detected_pattern_per / len(only_csv_files) * 100
    plt.plot(x, queried_per, label='GACS')

    plt.plot(x, queried_misclassified_per, label='HFLGA')

    plt.plot(x, detected_pattern_per, label='INMA')


    plt.xlim(-1, len(x) + 1)
    plt.ylim(-2, 105)
    plt.grid('x', color='0.85',linestyle="--")
    plt.grid('y', color='0.85',linestyle="--")
    plt.ylabel("%")
    plt.xlabel("Iteration")
    plt.legend(loc='best')
    plt.savefig("test.png")

dire = "/home/ubuntu/baonn/test/ufs/queries_num_25/original_instances"
draw_performance(dire, dire)