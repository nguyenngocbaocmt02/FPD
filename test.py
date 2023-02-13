import os
import sys
import yaml
from yaml.loader import SafeLoader
from dc.dcbench_instance import DcbenchInstance
import numpy as np
sys.path.append(os.path.dirname(__file__))


import numpy as np

# Define the original similarity matrix
original_sim_matrix = np.array([[1, 0.5, 0.7], [0.5, 1, 0.6], [0.7, 0.6, 1]])

# Define the rearrangement order
rearrangement_order = np.array([2, 0, 1])

# Use the rearrangement order to permute the rows and columns of the original similarity matrix
new_sim_matrix = original_sim_matrix[np.ix_(rearrangement_order, rearrangement_order)]

print("New Similarity Matrix:")
print(new_sim_matrix)
