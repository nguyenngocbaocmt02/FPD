import umap
import matplotlib.pyplot as plt
import meerkat as mk
import os 
import copy
import sys
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

def draw_dataset(dataset_folder_path, image_save_path):
    # UMAP
    dp, cfg = read_dataset(dataset_folder_path)
    emb = dp["emb"]
    pattern = dp["pattern"]
    true_label = dp["true_label"]
    pseudo_label = dp["pseudo_label"]

    in_pattern_samples = []
    in_pattern_pattern = []

    failed_samples = []
    pattern_failed_samples = []
    for i, e in enumerate(emb):
        if pattern[i] != -1:
            in_pattern_samples.append(emb[i])
            in_pattern_pattern.append(pattern[i])
        if true_label[i] != pseudo_label[i]:
            failed_samples.append(emb[i])
            pattern_failed_samples.append(pattern[i])

    in_pattern_samples = np.array(in_pattern_samples)
    in_pattern_pattern = np.array(in_pattern_pattern)
    print(len(in_pattern_samples), len(failed_samples))
    reducer = umap.UMAP(n_neighbors=cfg["knn_num"],
                    min_dist=0.1,
                    n_components=2,
                    metric='euclidean', n_epochs=150, random_state=42)
    reducer.fit_transform(in_pattern_samples, in_pattern_pattern)

    emb_failed_reduced = reducer.transform(failed_samples)

    plt.scatter([emb_failed_reduced[i][0] for i in range(len(pattern_failed_samples)) if pattern_failed_samples[i] == -1],
                    [emb_failed_reduced[i][1] for i in range(len(pattern_failed_samples)) if pattern_failed_samples[i] == -1],
                    label="Noise samples", edgecolors='k', linewidths=0.5)

    for pattern in set(pattern_failed_samples):
        if pattern == -1:
            continue
        plt.scatter([emb_failed_reduced[i][0] for i in range(len(pattern_failed_samples)) if pattern_failed_samples[i] == pattern],
                    [emb_failed_reduced[i][1] for i in range(len(pattern_failed_samples)) if pattern_failed_samples[i] == pattern],
                    label="Pattern " + str(pattern), marker="o", edgecolors='k', linewidths=0.5)  
    plt.legend(loc='best')
    plt.xticks([])
    plt.yticks([])
    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
    plt.savefig(image_save_path, dpi = 500, bbox_inches='tight')
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    datasets_folder_path = sys.argv[1]
    result_folder_path = sys.argv[2]

    datasets = os.listdir(datasets_folder_path) 
    
    for dataset in datasets:          
        dataset_path = os.path.join(os.path.join(datasets_folder_path, dataset))
        save_path = os.path.join(os.path.join(result_folder_path, dataset + ".pdf"))

        draw_dataset(dataset_path, save_path)