import numpy as np
from scipy.spatial.distance import pdist, squareform

# each row is a sample, each column is a dimension
def gaussian_similarity_matrix(x, h):
    km = squareform(pdist(x, 'sqeuclidean')) / (-2 * h**2)
    return np.exp(km, km)

def gaussian_similarity_matrix_with_psedolabels_distribution(x, hX, mean_distribution_x, cov_distribution_x, h):
    res = gaussian_similarity_matrix(x, hX)
    tmp = gaussian_similarity_matrix(mean_distribution_x, h)
    res = np.multiply(res, tmp); del tmp
    tmp = gaussian_similarity_matrix(cov_distribution_x, h)
    res = np.multiply(res, tmp); del tmp
    return res