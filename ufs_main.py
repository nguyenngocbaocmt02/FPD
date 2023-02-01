import sys
import os
import yaml
import meerkat as mk
from yaml.loader import SafeLoader
import numpy as np
from utils.misclassified_patterns_djs import DisjSet
from utils.instance import Instance
from methods.ufs.ufs import UniS
import csv

sys.path.append(os.path.dirname(__file__))
def read_instance(save_path):
    dp = mk.DataPanel.read(os.path.join(save_path, "data"))
    with open(os.path.join(save_path, "config.yaml"), "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    return dp, cfg   

def ufs_solve_instance(dp, cfg, queries_num, seed, save_path):
    np.random.seed(seed)
    instance = Instance(dp, cfg["pattern_size"], cfg["knn_num"])
    model = UniS()
    model.fit()

    N = instance.emb.shape[0]
    queried_index = []
    unqueried_index = [_ for _ in range(N)]
    
    misclassified_index = []
    misclassified_mask = [False for i in range(N)]
    misclassified_patterns_djs = DisjSet(N)
    iteration = 0

    log_queried_per = []
    log_detected_missclassified_patterns_per = []
    log_misclassified_samples_per = []
    log_each_pattern = [[] for i in range(len(instance.misclassified_patterns))]
    log_each_pattern2 = [[] for i in range(len(instance.misclassified_patterns))]

    while True:
        iteration += 1
        queries = model.suggest_queries(unqueried_index=unqueried_index, queries_num=queries_num)

        for query in queries:
            if instance.true_label[query] == instance.pseudo_label[query]:
                pass
            else:
                misclassified_patterns_djs.Activate(query)
                misclassified_mask[query] = True
                for j in misclassified_index:
                    if instance.adj_matrix[j][query]:
                        misclassified_patterns_djs.Union(j, query)
                misclassified_index.append(query)    
            queried_index.append(query)
            unqueried_index.remove(query)   

        detected_pattern_mask = np.zeros(len(instance.misclassified_patterns))
        detected_patterns = misclassified_patterns_djs.misclassified_patterns(instance.pattern_size)
        for pattern in detected_patterns:
            pattern_id = instance.pattern[pattern[0]]
            detected_pattern_mask[pattern_id] = 1
        num_detected_patterns = np.sum(detected_pattern_mask)


        log_queried_per.append(float(len(queried_index)) / N)
        log_misclassified_samples_per.append(float(len(misclassified_index)) / len(instance.misclassified_index))
        log_detected_missclassified_patterns_per.append(float(num_detected_patterns) / len(instance.misclassified_patterns))
        for ii, pattern in enumerate(instance.misclassified_patterns):
            tmp = 0
            for sample in pattern:
                if misclassified_mask[sample] == True:
                    tmp += 1
            log_each_pattern[ii].append(float(tmp) / len(instance.misclassified_patterns[ii]))
            log_each_pattern2[ii].append(min(1.0, float(tmp) / instance.pattern_size))

        print("Iteration", iteration, ":", float(num_detected_patterns) / len(instance.misclassified_patterns), len(misclassified_index) / len(instance.misclassified_index))
        if len(unqueried_index) == 0:
            break

    rows = []
    for i in range(len(log_queried_per)):
        tmp = [i + 1]
        tmp.append(log_queried_per[i])
        tmp.append(log_misclassified_samples_per[i])
        tmp.append(log_detected_missclassified_patterns_per[i])
        for j in range(len(instance.misclassified_patterns)):
            tmp.append(log_each_pattern2[j][i])
            tmp.append(log_each_pattern[j][i])
        rows.append(tmp)

    fields = ["Iteration", 'Queried X/ All X', 'Queried misclassified samples / All misclassified samples', 'Detected patterns / All patterns']
    for i in range(len(instance.misclassified_patterns)):
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
    dataset_folder_path = sys.argv[2]
    result_folder_path = sys.argv[3]

    full_path_to_config_dataset = os.path.join(dataset_folder_path, "config.yaml")
    with open(full_path_to_config_dataset, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=SafeLoader)
    
    seeds = [_ for _ in range(30)]
    for seed in seeds:
        for type_instance in cfg:
            type_config = cfg[type_instance]
            dire = type_config["dir"]
            instances_id = type_config["instances"]
            for instance_id in instances_id:
                print(dire, instance_id)
                instance_path = os.path.join(os.path.join(dataset_folder_path, dire), instance_id)
                save_path = os.path.join(os.path.join(result_folder_path, dire), instance_id + "_seed_" + str(seed) + ".csv")
                if os.path.exists(save_path):
                    continue
                dp, config = read_instance(instance_path)
                ufs_solve_instance(dp, config, queries_num, seed, save_path)

