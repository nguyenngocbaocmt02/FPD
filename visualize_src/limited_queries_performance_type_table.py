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
    second_inds = []
    third_inds = []
    for i, file in enumerate(only_csv_files):
        first_ind = 0.0
        second_ind = 0.0
        third_ind = 0.0
        a1 = True
        a2 = True
        a3 = True
        data = pd.read_csv(os.path.join(result_folder_path, file))
        x = np.array(list(map(int, data['Iteration'])))
        queried_per = np.array(list(map(float, data['Queried X/ All X'])))
        queried_misclassified_per = np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
        detected_pattern_per = np.array(list(map(float, data['Detected patterns / All patterns'])))
        for i in range(len(x)):
            if queried_per[i] >= 0.1 and a1:
                first_ind = detected_pattern_per[i]
                a1 = False
            if queried_per[i] >= 0.2 and a2:
                second_ind = detected_pattern_per[i]
                a2 = False
            if queried_per[i] >= 0.3 and a3:
                third_ind = detected_pattern_per[i]
                a3 = False
        first_inds.append(first_ind)
        second_inds.append(second_ind)
        third_inds.append(third_ind)
    
    return first_inds, second_inds, third_inds
    

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    table_path = sys.argv[2]
    methods = ["ufs", "gp_cdpp_0", "gp_cdpp_0.25", "gp_cdpp_0.5", "gp_cdpp_0.75", "gp_cdpp_1.0"]
    datasets = ["Low SNR datasets", "Medium SNR datasets", "High SNR datasets"]
    first_table = {key1: {key2: [] for key2 in methods} for key1 in datasets}
    second_table = {key1: {key2: [] for key2 in methods} for key1 in datasets}
    third_table = {key1: {key2: [] for key2 in methods} for key1 in datasets}
    all_first_table = {key1: [] for key1 in methods}
    all_second_table = {key1: [] for key1 in methods}
    all_third_table = {key1: [] for key1 in methods}
    for method in methods:
        for i in range(1, 16):
            dataset = "id_" + str(i)          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            first_inds, second_inds, third_inds = performance(res_path)
            all_first_table[method].append(first_inds)
            all_second_table[method].append(second_inds)
            all_third_table[method].append(third_inds)
            
            if i <= 5:
                first_table["Low SNR datasets"][method] += first_inds
                second_table["Low SNR datasets"][method] += second_inds
                third_table["Low SNR datasets"][method] += third_inds
            elif i <= 10:
                first_table["Medium SNR datasets"][method] += first_inds
                second_table["Medium SNR datasets"][method] += second_inds
                third_table["Medium SNR datasets"][method] += third_inds
            elif i <= 15:
                first_table["High SNR datasets"][method] += first_inds
                second_table["High SNR datasets"][method] += second_inds
                third_table["High SNR datasets"][method] += third_inds

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # write the headers row for f_table
        writer.writerow(['Dataset'] + [f'{method} (10%)' for method in methods])
        # write the data rows for f_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                tmp = np.array(first_table[dataset][method])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(all_first_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)
        writer.writerow([])
        writer.writerow([])
        # write the headers row for a_table
        writer.writerow(['Dataset'] + [f'{method} (20%)' for method in methods])
        # write the data rows for a_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                tmp = np.array(second_table[dataset][method])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(all_second_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)
        writer.writerow([])
        writer.writerow([])
        # write the headers row for a_table
        writer.writerow(['Dataset'] + [f'{method} (30%)' for method in methods])
        # write the data rows for a_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                tmp = np.array(third_table[dataset][method])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(all_third_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)