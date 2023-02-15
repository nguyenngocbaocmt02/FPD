import umap
import matplotlib.pyplot as plt
import meerkat as mk
import os 
import copy
import sys
import csv
import yaml
import warnings
warnings.filterwarnings("ignore")
from yaml import SafeLoader
import numpy as np
import matplotlib.patches as mpatches

def read_dataset(save_path):
    dp = mk.DataPanel.read(os.path.join(save_path, "data"))
    with open(os.path.join(save_path, "config.yaml"), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return dp, cfg   

if __name__ == "__main__":
    datasets_folder_path = sys.argv[1]
    table_file_path = sys.argv[2]

    datasets = ["id_" + str(i) for i in range(1, 16)]
    info = [[] for _ in range(len(datasets))]
    for i, dataset in enumerate(datasets):      
        dataset_path = os.path.join(os.path.join(datasets_folder_path, dataset))
        dp, cfg = read_dataset(dataset_path)
        n_samples = len(dp)
        n_misclassified = 0
        n_inpattern = 0

        pattern = dp["pattern"]
        true_label = dp["true_label"]
        pseudo_label = dp["pseudo_label"]
        for j, e in enumerate(true_label):
            if pattern[j] != -1:
                n_inpattern += 1
            if true_label[j] != pseudo_label[j]:
                n_misclassified += 1
        
        info[i].append(cfg["dcbench"])
        info[i].append(str(round(float(cfg["snr"]), 2)))
        info[i].append(str(cfg["pattern_size"]))
        info[i].append(str(cfg["knn_num"]))
        info[i].append(str(n_samples))
        info[i].append(str(n_misclassified))
        
    # write the info list to a CSV file
    os.makedirs(os.path.dirname(table_file_path), exist_ok=True)
    with open(table_file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Dataset", "DcBench", "SNR", "M", "K_nn", "The number of samples", "The number of misclassifed samples"])
        for dataset, dataset_info in zip(datasets, info):
            csvwriter.writerow([dataset] + dataset_info)