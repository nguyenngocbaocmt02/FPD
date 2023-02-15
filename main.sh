#!/bin/bash

# Run UFS
python main_src/ufs_main.py 25 datasets results

# Run proposal
python main_src/gp_cdpp_main.py 25 datasets results