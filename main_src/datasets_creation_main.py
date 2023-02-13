import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")
from yaml.loader import SafeLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dc.dcbench_instance import DcbenchInstance


def write_dataset(dp, param_dic, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dp_path = os.path.join(save_path, "data")
    param_path = os.path.join(save_path, "config.yaml")
    dp.write(dp_path)
    with open(param_path, 'w') as file:
        documents = yaml.dump(param_dic, file)

if __name__ == "__main__":
    datasets_config_path = sys.argv[1]
    datasets_dir_path = sys.argv[2]
    with open(datasets_config_path, "r") as ymlfile:
        datasets_config = yaml.load(ymlfile, Loader=SafeLoader)
    os.makedirs(os.path.dirname(os.path.join(datasets_dir_path, "config.yaml")), exist_ok=True)

    for dataset_id in datasets_config:
        dataset_config = datasets_config[dataset_id]
        dcbench = dataset_config["dcbench"]
        pattern_size = int(dataset_config["pattern_size"])
        knn_num = int(dataset_config["knn_num"])
        save_path = os.path.join(datasets_dir_path, dataset_id)
        if os.path.exists(save_path):
            continue
        dcbench_instance = DcbenchInstance(instance_id=dcbench, with_image=False)
        dp, snr = dcbench_instance.create_instance(pattern_size=pattern_size, knn_num=knn_num)
        dataset_config["snr"] = snr
        write_dataset(dp, dataset_config, save_path)
        print(dataset_id, snr)
