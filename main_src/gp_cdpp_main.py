import sys
import os
import yaml
import copy
import meerkat as mk
from yaml.loader import SafeLoader
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.misclassified_patterns_djs import DisjSet
from utils.dataset import Dataset
from methods.gp_cdpp.gp_cdpp import GPCDPP
sys.path.append(os.path.dirname(__file__))
import csv

def read_dataset(save_path):
    dp = mk.DataPanel.read(os.path.join(save_path, "data"))
    with open(os.path.join(save_path, "config.yaml"), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return dp, cfg   

def gp_cdpp_solve_dataset(dp, cfg, vartheta, queries_num, save_path):
    dataset = Dataset(dp, cfg["pattern_size"], cfg["knn_num"])
    model = GPCDPP(delta=np.sqrt(2) * 10 ** (-6), vartheta=vartheta, down_size=2, L=-100.0, U=100.0)
    model.fit(emb=dataset.emb, C=np.max(dataset.pseudo_label) + 1, pseudo_label=dataset.pseudo_label)
    N = dataset.emb.shape[0]
    queried_index = []
    unqueried_index = [_ for _ in range(N)]
    g_observe = np.array([0 for _ in range(N)])

    misclassified_index = []
    misclassified_mask = [False for i in range(N)]
    misclassified_patterns_djs = DisjSet(N)
    iteration = 0

    log_queried_per = []
    log_detected_missclassified_patterns_per = []
    log_misclassified_samples_per = []
    log_each_pattern = [[] for i in range(len(dataset.misclassified_patterns))]
    log_each_pattern2 = [[] for i in range(len(dataset.misclassified_patterns))]

    while True:
        iteration += 1
        queries = model.suggest_queries(g_observe=g_observe, queried_index=queried_index, unqueried_index=unqueried_index, queries_num=queries_num)
        flag = False
        for query in queries:
            if dataset.true_label[query] == dataset.pseudo_label[query]:
                g_observe[query] = model.L
            else:
                misclassified_patterns_djs.Activate(query)
                misclassified_mask[query] = True
                for j in misclassified_index:
                    if dataset.adj_matrix[j][query]:
                        misclassified_patterns_djs.Union(j, query)
                misclassified_index.append(query)    
                tmp = misclassified_patterns_djs.list[misclassified_patterns_djs.find(query)]
                if len(tmp) >= dataset.pattern_size:
                    for x in tmp:
                        g_observe[x] = model.L
                else:
                    for x in tmp:
                        g_observe[x] = model.U
            queried_index.append(query)
            unqueried_index.remove(query)   
            
        detected_pattern_mask = np.zeros(len(dataset.misclassified_patterns))
        detected_patterns = misclassified_patterns_djs.misclassified_patterns(dataset.pattern_size)
        for pattern in detected_patterns:
            pattern_id = dataset.pattern[pattern[0]]
            detected_pattern_mask[pattern_id] = 1
        num_detected_patterns = np.sum(detected_pattern_mask)


        log_queried_per.append(float(len(queried_index)) / N)
        log_misclassified_samples_per.append(float(len(misclassified_index)) / len(dataset.misclassified_index))
        log_detected_missclassified_patterns_per.append(float(num_detected_patterns) / len(dataset.misclassified_patterns))
        for ii, pattern in enumerate(dataset.misclassified_patterns):
            tmp = 0
            for sample in pattern:
                if misclassified_mask[sample] == True:
                    tmp += 1
            log_each_pattern[ii].append(float(tmp) / len(dataset.misclassified_patterns[ii]))
            log_each_pattern2[ii].append(min(1.0, float(tmp) / dataset.pattern_size))

        print("Iteration", iteration, ":", float(num_detected_patterns) / len(dataset.misclassified_patterns), len(misclassified_index) / len(dataset.misclassified_index))
        if len(unqueried_index) == 0:
            break

    rows = []
    for i in range(len(log_queried_per)):
        tmp = [i + 1]
        tmp.append(log_queried_per[i])
        tmp.append(log_misclassified_samples_per[i])
        tmp.append(log_detected_missclassified_patterns_per[i])
        for j in range(len(dataset.misclassified_patterns)):
            tmp.append(log_each_pattern2[j][i])
            tmp.append(log_each_pattern[j][i])
        rows.append(tmp)

    fields = ["Iteration", 'Queried X/ All X', 'Queried misclassified samples / All misclassified samples', 'Detected patterns / All patterns']
    for i in range(len(dataset.misclassified_patterns)):
        fields.append("Pattern " + str(i) + " (%)") 
        fields.append("Pattern " + str(i) + " (% Detected)")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # writing to csv file 
    with open(save_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields) 
        csvwriter.writerows(rows)

if __name__ == "__main__":
    queries_num = int(sys.argv[1])
    datasets_folder_path = sys.argv[2]
    result_folder_path = sys.argv[3]

    datasets = os.listdir(datasets_folder_path) 
    varthetas = [0.25, 0.5, 0, 0.75, 1.0]
    for vartheta in varthetas:
        for dataset in datasets:    
            if dataset != "id_8":
                continue      
            dataset_path = os.path.join(os.path.join(datasets_folder_path, dataset))
            save_path = os.path.join(os.path.join(os.path.join(result_folder_path, "gp_cdpp_" + str(vartheta)), dataset), "result.csv")
            if os.path.exists(save_path):
                continue
            dp, config = read_dataset(dataset_path)
            print(dataset, vartheta, config["snr"])  
            try:
                gp_cdpp_solve_dataset(dp, config, vartheta, queries_num, save_path)
            except:
                pass