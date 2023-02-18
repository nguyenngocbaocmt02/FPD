## [Efficient Failure Pattern Identification of Predictive Algorithms]

### Generate datasets.
* run command: `$ python main_src/datasets_creation_main.py <path to datasets config (.yaml)> <path to datasets save folder>`
* Config of dataset in our report in dataset_creation_config.yaml, example: `$ python main_src/datasets_creation_main.py main_src/datasets_creation_config.yaml datasets`

### Run all methods
`$ chmod +x main.sh`
`$ ./main.sh`

### Run Uniform sampling model
* run command: `$ python main_src/ufs_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python main_src/ufs_main.py 25 datasets results`

### Run GP_CDPP model.
* run command: `$ python main_src/gp_cdpp_main.py <the number of queries each round> <path to dataset folder> <path to result folder to save>`
* In our work, the number of queries each round is 25, example: `$ python main_src/gp_cdpp_main.py 25 datasets results`

### Run all visualizing script
`$ chmod +x visualize.sh`
`$ ./visualize.sh`

### Visualize the 2-d image of datasets
* run command: `$ python visualize_src/draw_datasets.py <path to datasets folder> <path to save folder>`
*Example: `$ python visualize_src/draw_datasets.py datasets/ figures/dataset_figures`

### Visualize the performance of models
* run command: `$ python visualize_src/draw_performance.py <path to results folder> <path to save folder>`
*Example: `$ python visualize_src/draw_performance.py results figures/performance_figures`

### Tables on report
*Example: `$ python visualize_src/dataset_description_table.py datasets/ figures/datasets_description.csv`
*Example `$ python visualize_src/performance_table.py results/ figures/performance_table.csv`
*Example `$ python visualize_src/performance_type_table.py results/ figures/performance_type_table.csv`
*Example `$ python visualize_src/limited_queries_performance_table.py results/ figures/limited_queries_performance_table.csv`