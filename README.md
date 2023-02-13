## [Efficient Failure Pattern Identification of Predictive Algorithms]

### Generate datasets.
* run command: `$ python main_src/datasets_creation_main.py <path to datasets config (.yaml)> <path to datasets save folder>`
* Config of dataset in our report in dataset_creation_config.yaml, example: `$ python main_src/datasets_creation_main.py main_src/datasets_creation_config.yaml datasets`

### Run Uniform sampling model
* run command: `$ python main_src/ufs_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python main_src/ufs_main.py 25 datasets results`

### Run GP_CDPP model.
* run command: `$ python main_src/gp_cdpp_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python main_src/gp_cdpp_main.py 25 datasets results`

### Visualize the 2-d image of datasets
* run command: `$ python visualize_src/draw_datasets.py <path to datasets folder> <path to save folder>`
*Example: `$ python visualize_src/draw_datasets.py datasets/ figures/dataset_figures`

### Visualize the performance of models
* run command: `$ python visualize_src/draw_performance.py <path to results folder> <path to save folder>`
*Example: `$ python visualize_src/draw_performance.py results figures/performance_figures`
