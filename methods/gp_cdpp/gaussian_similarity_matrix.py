import dcbench
import meerkat as mk
import numpy as np
import time
import math
import scipy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


# each row is a sample, each column is a dimension
def gaussian_similarity_matrix(x, h):
    km = squareform(pdist(x, 'sqeuclidean')) / (-2 * h**2)
    return np.exp(km, km)

def gaussian_similarity_matrix_raw(x, h):
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    pt_sq_norms = (x ** 2).sum(axis=1)
    dists_sq = np.dot(x, x.T)
    dists_sq *= -2
    dists_sq += pt_sq_norms.reshape(-1, 1)
    dists_sq += pt_sq_norms
    # turn into an RBF gram matrix
    km = dists_sq; del dists_sq
    km /= -2 * h**2
    return np.exp(km, km)

def gaussian_similarity_matrix_with_psedolabels_distribution(x, hX, mean_distribution_x, cov_distribution_x, h):
    res = gaussian_similarity_matrix_raw(x, hX)
    tmp = gaussian_similarity_matrix_raw(mean_distribution_x, h)
    res = np.multiply(res, tmp); del tmp
    tmp = gaussian_similarity_matrix_raw(cov_distribution_x, h)
    res = np.multiply(res, tmp); del tmp
    return res