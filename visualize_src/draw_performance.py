import matplotlib.pyplot as plt 
import os
from os.path import join, isfile
from os import listdir
import csv 
import pandas as pd
import sys
import numpy as np

def draw_performance(result_folder_path, image_save_path):
    only_csv_files = [f for f in listdir(result_folder_path) if isfile(join(result_folder_path, f)) and f[-4:] == ".csv"]

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
    plt.plot(x, queried_per, label='Queried percentage')

    plt.plot(x, queried_misclassified_per, label='Queried misclassifed percentage')

    plt.plot(x, detected_pattern_per, label='Detected pattern percentage')


    plt.xlim(-1, len(x) + 1)
    plt.ylim(-2, 105)
    plt.grid('x', color='0.85',linestyle="--")
    plt.grid('y', color='0.85',linestyle="--")
    plt.ylabel("%")
    plt.xlabel("Iteration")
    plt.legend(loc='best')
    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
    plt.savefig(image_save_path, dpi = 500, bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    figure_folder_path = sys.argv[2]

    methods = os.listdir(result_folder_path) 
    for method in methods:
        for dataset in os.listdir(os.path.join(result_folder_path, method)):          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            fig_path = os.path.join(os.path.join(figure_folder_path, method), dataset + ".pdf")
            draw_performance(res_path, fig_path)