import matplotlib.pyplot as plt 
import os
from os.path import join, isfile
from os import listdir
import csv 
import pandas as pd
import sys
import numpy as np

def performance(result_folder_path):
    print(result_folder_path)
    only_csv_files = [f for f in listdir(result_folder_path) if isfile(join(result_folder_path, f)) and f[-4:] == ".csv"]
    mapp = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    inds = [[] for _ in range(len(mapp))]
    first_inds = []
    all_inds = []
    for i, file in enumerate(only_csv_files):
        ind = [0.0 for _ in range(len(mapp))]
        a = [True for _ in range(len(mapp))]
        first_ind = 1.0
        all_ind = 1.0
        data = pd.read_csv(os.path.join(result_folder_path, file))
        x = np.array(list(map(int, data['Iteration'])))
        queried_per = np.array(list(map(float, data['Queried X/ All X'])))
        queried_misclassified_per = np.array(list(map(float, data['Queried misclassified samples / All misclassified samples'])))
        detected_pattern_per = np.array(list(map(float, data['Detected patterns / All patterns'])))
        for j in range(len(x)):
            for k, threshold in enumerate(mapp):
                if queried_per[j] >= threshold and a[k]:
                    ind[k] = detected_pattern_per[j]
                    a[k] = False
            if detected_pattern_per[i] > 0 and first_ind == 1.0:
                first_ind = queried_per[i]
            if detected_pattern_per[i] == 1.0 and all_ind == 1.0:
                all_ind = queried_per[i]
        first_inds.append(first_ind)
        all_inds.append(all_ind)
        for k, threshold in enumerate(mapp):
            inds[k].append(ind[k])      
    
    return inds, first_inds, all_inds
    

if __name__ == "__main__":
    result_folder_path = sys.argv[1]
    table_path = sys.argv[2]
    methods = [subfolder.name for subfolder in os.scandir(result_folder_path) if subfolder.is_dir()]
    dataset_types = ["Low SNR datasets", "Medium SNR datasets", "High SNR datasets"]

    performance_table = {key1: {key2: [[[] for _ in range(10)], [], []] for key2 in methods} for key1 in dataset_types}
    overall_performance_table = {key2: [[[] for _ in range(10)], [], []] for key2 in methods}
    for method in methods:
        for i in range(1, 16):
            dataset = "id_" + str(i)          
            res_path = os.path.join(os.path.join(os.path.join(result_folder_path, method), dataset))
            inds, first_inds, last_inds = performance(res_path)
            if i <= 5:
                dataset_type = dataset_types[0]
            elif i <= 10:
                dataset_type = dataset_types[1]
            else:
                dataset_type = dataset_types[2]

            for stt, list_value in enumerate(inds):
                performance_table[dataset_type][method][0][stt].append(list_value)
                overall_performance_table[method][0][stt].append(list_value)
            performance_table[dataset_type][method][1].append(first_inds)
            performance_table[dataset_type][method][2].append(last_inds)
            overall_performance_table[method][1].append(first_inds)
            overall_performance_table[method][2].append(last_inds)
            
    
    try:
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
    except:
        pass
    with open(table_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        for stt in range(10):
            # write the headers row for f_table
            writer.writerow(['Dataset (' + str((stt + 1) * 10) + " %" +')'] + [f'{method}' for method in methods])
            # write the data rows for f_table
            for dataset_type in dataset_types:
                row_data = [dataset_type]
                for method in methods:
                    tmp = np.array(performance_table[dataset_type][method][0][stt])
                    row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
                writer.writerow(row_data)
            row_data = ["Overall"]
            for method in methods:
                tmp = np.array(overall_performance_table[method][0][stt])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
            writer.writerow([])
            writer.writerow([])

        # write the headers row for f_table
        writer.writerow(['Dataset (First mode)'] + [f'{method}' for method in methods])
        # write the data rows for f_table
        for dataset_type in dataset_types:
            row_data = [dataset_type]
            for method in methods:
                tmp = np.array(performance_table[dataset_type][method][1])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(overall_performance_table[method][1][stt])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)
        writer.writerow([])
        writer.writerow([])

        writer.writerow(['Dataset (All mode)'] + [f'{method}' for method in methods])
        # write the data rows for f_table
        for dataset_type in dataset_types:
            row_data = [dataset_type]
            for method in methods:
                tmp = np.array(performance_table[dataset_type][method][2])
                row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
            writer.writerow(row_data)
        row_data = ["Overall"]
        for method in methods:
            tmp = np.array(overall_performance_table[method][2][stt])
            row_data.append(str(round(np.mean(tmp), 2)) + '\u00B1' + str(round(np.std(tmp), 2)))
        writer.writerow(row_data)
        writer.writerow([])
        writer.writerow([])
        
