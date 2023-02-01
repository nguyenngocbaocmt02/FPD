import os
import sys
import yaml
from yaml.loader import SafeLoader
from dc.dcbench_instance import DcbenchInstance
sys.path.append(os.path.dirname(__file__))

def write_dataset(dp, param_dic, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dp_path = os.path.join(save_path, "data")
    param_path = os.path.join(save_path, "config.yaml")
    dp.write(dp_path)
    with open(param_path, 'w') as file:
        documents = yaml.dump(param_dic, file)

if __name__ == "__main__":
    dataset_config_path = sys.argv[1]
    dataset_dir_path = sys.argv[2]
    with open(dataset_config_path, "r") as ymlfile:
        dataset_config = yaml.load(ymlfile, Loader=SafeLoader)
    with open(os.path.join(dataset_dir_path, "config.yaml"), 'w') as file:
        documents = yaml.dump(dataset_config, file)
    for tp in dataset_config:
        original_instances_config = dataset_config[tp]
        dire = original_instances_config["dir"]
        instances = original_instances_config["instances"]
        for instance_id in instances:
            print(instance_id)
            save_path = os.path.join(dataset_dir_path, dire)
            save_path = os.path.join(save_path, instance_id)
            if os.path.exists(save_path):
                continue
            param_dic = instances[instance_id]
            instance = DcbenchInstance(instance_id=str(instance_id), with_image=False)
            dp = instance.create_instance(pattern_size=int(param_dic["pattern_size"]), knn_num=int(param_dic["knn_num"]), snr=float(param_dic["snr"]))
            del instance
            write_dataset(dp, param_dic, save_path)
