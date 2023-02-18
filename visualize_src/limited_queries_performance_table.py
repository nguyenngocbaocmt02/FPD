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
        data = pd.read_csv(os.path.join(result_folder_path, file))
        x = np.array(list(map(int, data['Iteration'])))
        queried_per = np.array(list(map(float, data['Queried X/ All X'])))
        queried_misclassified_per = np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
        detected_pattern_per = np.array(list(map(float, data['Detected patterns / All patterns'])))
        for i in range(len(x)):
            if queried_per[i] >= 0.1 and first_ind == 0.0:
                first_ind = detected_pattern_per[i]
            if queried_per[i] >= 0.2 and second_ind == 0.0:
                second_ind = detected_pattern_per[i]
            if queried_per[i] >= 0.3 and third_ind == 0.0:
                third_ind = detected_pattern_per[i]
        first_inds.append(first_ind)
        second_inds.append(second_ind)
        third_inds.append(third_ind)
    
    return first_inds, second_inds, third_inds
    

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    table_path = sys.argv[2]
    methods = ["ufs", "gp_cdpp_0", "gp_cdpp_0.25", "gp_cdpp_0.5", "gp_cdpp_0.75", "gp_cdpp_1.0"]
    datasets = ["id_" + str(i) for i in range(1, 16)]
    first_table = {key1: {key2: None for key2 in methods} for key1 in datasets}
    second_table = {key1: {key2: None for key2 in methods} for key1 in datasets}
    third_table = {key1: {key2: None for key2 in methods} for key1 in datasets}
    all_first_table = {key1: [] for key1 in methods}
    all_second_table = {key1: [] for key1 in methods}
    all_third_table = {key1: [] for key1 in methods}
    for method in methods:
        for dataset in os.listdir(os.path.join(result_folder_path, method)):          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            first_inds, second_inds, third_inds = performance(res_path)
            all_first_table[method].append(first_inds)
            all_second_table[method].append(second_inds)
            all_third_table[method].append(third_inds)
            
            first_table[dataset][method] = str(round(np.mean(np.array(first_inds)), 2)) + '\u00B1' + str(round(np.std(np.array(first_inds)), 2))
            second_table[dataset][method] = str(round(np.mean(np.array(second_inds)), 2)) + '\u00B1' + str(round(np.std(np.array(second_inds)), 2))
            third_table[dataset][method] = str(round(np.mean(np.array(third_inds)), 2)) + '\u00B1' + str(round(np.std(np.array(third_inds)), 2))

    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # write the headers row for f_table
        writer.writerow(['Dataset'] + [f'{method} (10%)' for method in methods])
        # write the data rows for f_table
        for dataset in datasets:
            row_data = [dataset]
            for method in methods:
                row_data.append(first_table[dataset][method])
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
                row_data.append(second_table[dataset][method])
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
                row_data.append(third_table[dataset][method])
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(all_third_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)