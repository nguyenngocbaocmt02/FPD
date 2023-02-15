#!/bin/bash

# Visualize datasets
python visualize_src/draw_datasets.py datasets/ figures/dataset_figures

# Visualize model performance
python visualize_src/draw_performance.py results figures/performance_figures

# Generate dataset description table
python visualize_src/dataset_description_table.py datasets/ figures/datasets_description.csv

# Generate performance table
python visualize_src/performance_table.py results/ figures/performance_table.csv

# Generate performance type table
python visualize_src/performance_type_table.py results/ figures/performance_type_table.csv
