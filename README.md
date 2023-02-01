## [Efficient Failure Pattern Identification of Predictive Algorithms]

### Generate dataset.
* run command: `$ python dataset_creation_main.py <path to dataset config (.yaml)> <path to dataset save folder>`
* Config of dataset in our report in dataset_creation_config.yaml, example: `$ python dataset_creation_main.py dataset_creation_config.yaml dataset`

### Run Uniform sampling model
* run command: `$ python ufs_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python ufs_main.py 25 dataset exp/ufs`

### Run GP_CDPP model.
* run command: `$ python gp_cdpp_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python gp_cdpp_main.py 25 dataset exp/ufs`

### Generate Fig ...
...