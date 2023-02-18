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
    
    return first_inds, all_inds
    

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    table_path = sys.argv[2]
    methods = ["ufs", "gp_cdpp_0", "gp_cdpp_0.25", "gp_cdpp_0.5", "gp_cdpp_0.75", "gp_cdpp_1.0"]
    datasets = ["id_" + str(i) for i in range(1, 16)]

    typ = ["Low SNR datasets", "Medium SNR datasets", "High SNR datasets"]
    f_table_type = {key1: {key2: [] for key2 in methods} for key1 in typ}
    a_table_type = {key1: {key2: [] for key2 in methods} for key1 in typ}
    allf_table = {key1: [] for key1 in methods}
    alla_table = {key1: [] for key1 in methods}
    for method in methods:
        for i in range(1, 16):
            dataset = "id_" + str(i)          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            f_list, a_list = performance(res_path)
            if i <= 5:
                f_table_type["Low SNR datasets"][method] += f_list
                a_table_type["Low SNR datasets"][method] += a_list
            elif i <= 10:
                f_table_type["Medium SNR datasets"][method] += f_list
                a_table_type["Medium SNR datasets"][method] += a_list
            elif i <= 15:
                f_table_type["High SNR datasets"][method] += f_list
                a_table_type["High SNR datasets"][method] += a_list
            allf_table[method].append(f_list)
            alla_table[method].append(a_list)



    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # write the headers row for f_table
        writer.writerow(['Type'] + [f'{method} (f)' for method in methods])
        # write the data rows for f_table
        
        for tp in typ:
            row_data = [tp]
            for method in methods:
                res = np.array(f_table_type[tp][method])
                row_data.append(str(round(np.mean(res), 2)) + '\u00B1' + str(round(np.std(res), 2)))
            writer.writerow(row_data)
        # write the headers row for a_table
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(allf_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)
        writer.writerow([])
        writer.writerow([])

        writer.writerow(['Type'] + [f'{method} (a)' for method in methods])
        # write the data rows for a_table
        for tp in typ:
            row_data = [tp]
            for method in methods:
                res = np.array(a_table_type[tp][method])
                row_data.append(str(round(np.mean(res), 2)) + '\u00B1' + str(round(np.std(res), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(alla_table[method])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)