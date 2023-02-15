import matplotlib.pyplot as plt 
import os
from os.path import join, isfile
from os import listdir
import csv 
import pandas as pd
import sys
import numpy as np

def performance(result_folder_path):
    only_csv_files = [f for f in listdir(result_folder_path) if isfile(join(result_folder_path, f)) and f[-4:] == ".csv"]
    first_inds = []
    all_inds = []
    for i, file in enumerate(only_csv_files):
        first_ind = 1.0
        all_ind = 1.0
        data = pd.read_csv(os.path.join(result_folder_path, file))
        x = np.array(list(map(int, data['Iteration'])))
        queried_per = np.array(list(map(float, data['Queried X/ All X'])))
        queried_misclassified_per = np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
        detected_pattern_per = np.array(list(map(float, data['Detected patterns / All patterns'])))
        for i in range(len(x)):
            if detected_pattern_per[i] > 0 and first_ind == 1.0:
                first_ind = queried_per[i]
            if detected_pattern_per[i] == 1.0 and all_ind == 1.0:
                all_ind = queried_per[i]
        first_inds.append(first_ind)
        all_inds.append(all_ind)
    first_inds = np.array(first_ind)
    all_inds = np.array(all_inds)
    
    return np.mean(first_inds), np.std(first_inds), np.mean(all_inds), np.std(all_inds)
    

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    table_path = sys.argv[2]
    methods = ["ufs", "gp_cdpp_0", "gp_cdpp_0.25", "gp_cdpp_0.5", "gp_cdpp_0.75", "gp_cdpp_1.0"]
    datasets = ["id_" + str(i) for i in range(1, 16)]
    f_table = {key1: {key2: None for key2 in methods} for key1 in datasets}
    a_table = {key1: {key2: None for key2 in methods} for key1 in datasets}
    for method in methods:
        for dataset in os.listdir(os.path.join(result_folder_path, method)):          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            f_mean, f_std, a_mean, a_std = performance(res_path)
            f_table[dataset][method] = str(round(f_mean, 2)) + '\u00B1' + str(round(f_std, 2))
            a_table[dataset][method] = str(round(a_mean, 2)) + '\u00B1' + str(round(a_std, 2)) 

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # write the headers row for f_table
        writer.writerow(['Dataset'] + [f'{method} (f)' for method in methods])
        # write the data rows for f_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                row_data.append(f_table[dataset][method])
            writer.writerow(row_data)
        # write the headers row for a_table
        writer.writerow(['Dataset'] + [f'{method} (a)' for method in methods])
        # write the data rows for a_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                row_data.append(a_table[dataset][method])
            writer.writerow(row_data)