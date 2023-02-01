import numpy as np
import sys
import copy
import os
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
sys.path.append(os.path.dirname(__file__))
from dpp_solver import map_inference_dpp_local_search_2
from gaussian_similarity_matrix import gaussian_similarity_matrix_with_psedolabels_distribution

def square_distance_matrix_raw(x):
    if len(x.shape) > 2:
        x = x.reshape(x.shape[0], -1)
    pt_sq_norms = (x ** 2).sum(axis=1)
    dists_sq = np.dot(x, x.T)
    dists_sq *= -2
    dists_sq += pt_sq_norms.reshape(-1, 1)
    dists_sq += pt_sq_norms
    return dists_sq

class GPCDPP:
    def __init__(self, delta, down_size, vartheta, L, U):
        self.delta = delta
        self.down_size = down_size
        self.vartheta = vartheta
        self.L = L
        self.U = U

    def fit(self, emb, C, pseudo_label):
        #The number of samples
        N = emb.shape[0]
        #The number of dimensions of a sample
        d = emb.shape[1]
        # C: the number of labels
        pca = PCA(n_components = min(d, self.down_size))
        pca.fit(emb)
        emb_pca = pca.transform(emb)
        mean_labels_pca =  np.zeros([C, self.down_size]) # C x down_size
        cov_labels_pca = np.zeros([C, self.down_size, self.down_size]) # C x d x d
        cnt_labels_pca = np.zeros([C])
        for sample, pseudolabel in zip(emb_pca, pseudo_label):
            mean_labels_pca[pseudolabel] += sample
            cnt_labels_pca[pseudolabel] += 1
            
        for label in range(C):
            mean_labels_pca[label] /= cnt_labels_pca[label]
            
        for sample, pseudolabel in zip(emb_pca, pseudo_label):
            tmp = sample - mean_labels_pca[pseudolabel]
            cov_labels_pca[pseudolabel] += np.outer(tmp.T, tmp)
            
        for label in range(C):
            cov_labels_pca[label] /= cnt_labels_pca[label]
        
        cov_labels_pca = cov_labels_pca.reshape(cov_labels_pca.shape[0], -1)   
        self.emb = emb
        self.mean_distribution_emb = mean_labels_pca[pseudo_label]
        self.cov_distribution_emb = cov_labels_pca[pseudo_label]
        self.mean_gaussian = np.array([0.0 for i in range(N)]) 

        square_distance_matrix = square_distance_matrix_raw(emb)
        self.hX = ((np.sum(square_distance_matrix) / (N * (N-1))) / (np.log((N - 1) / (self.delta ** 2)) / 2)) ** 0.5

        square_distance_matrix = square_distance_matrix_raw(self.mean_distribution_emb) + square_distance_matrix_raw(self.cov_distribution_emb)
        self.h = ((np.sum(square_distance_matrix) / (N * (N-1)))/ (np.log((N - 1) / (self.delta ** 2)) / 2)) ** 0.5

    def suggest_queries(self, g_observe, queried_index, unqueried_index, queries_num):
        # for each iteration
        N_queried = len(queried_index)
        N_unqueried = len(unqueried_index)
        N = N_queried + N_unqueried
        if N_unqueried == 0:
            return []
        mapping_to_original_data = queried_index + unqueried_index
        emb_rearrange = np.array([self.emb[i] for i in mapping_to_original_data]) 
        mean_distribution_emb_rearrange = np.array([self.mean_distribution_emb[i] for i in mapping_to_original_data]) 
        cov_distribution_emb_rearrange = np.array([self.cov_distribution_emb[i] for i in mapping_to_original_data]) 
        g_rearrange = np.array([g_observe[i] for i in mapping_to_original_data])
        m_rearrange = np.array([self.mean_gaussian[i] for i in mapping_to_original_data])
        cov_rearrange = gaussian_similarity_matrix_with_psedolabels_distribution(emb_rearrange, self.hX, mean_distribution_emb_rearrange, cov_distribution_emb_rearrange, self.h)
        try:
            K = cov_rearrange[0 : N_queried, 0 : N_queried]
            K_s = cov_rearrange[0 : N_queried, N_queried : N]
            K_st = cov_rearrange[N_queried : N, 0 : N_queried]
            K_s_s = cov_rearrange[N_queried : N, N_queried : N]
            K_inverse = np.linalg.inv(K)
            m_unqueried = m_rearrange[N_queried : N] + K_st @ K_inverse @ (g_rearrange[0 : N_queried] - m_rearrange[0 : N_queried])
            cov_unqueried = K_s_s - K_st @ K_inverse @ K_s
        except Exception as error:
            print("Self warning: ", error)
            m_unqueried = m_rearrange[N_queried:N]
            cov_unqueried = cov_rearrange[N_queried : N, N_queried : N]
        alpha = 1 / (1 + np.exp(- m_unqueried))
        beta = 0.5 * (cov_unqueried @ (alpha * (1 - alpha) * (1 - 2 * alpha)))
        VoI_unqueried = alpha + beta
        for i in range(N_unqueried):
            if VoI_unqueried[i] <= 0:
                VoI_unqueried[i] = alpha[i]
        print(np.min(m_unqueried), np.max(m_unqueried))
        print(np.min(VoI_unqueried), np.max(VoI_unqueried))
        P_s_t = np.diag(VoI_unqueried)

        identity_unqueried = np.zeros([N, N])
        for i in range(N_queried, N):
            identity_unqueried[i][i] = 1

        S_s_t = np.linalg.inv(np.linalg.inv(cov_rearrange + identity_unqueried)[N_queried : N, N_queried : N]) - np.identity(N_unqueried)
        #S_s_t = cov_rearrange[N_queried : N, N_queried : N]
        T_s_t = self.vartheta * S_s_t + (1 - self.vartheta) * P_s_t

        queries_relative = map_inference_dpp_local_search_2(T_s_t, min(queries_num, T_s_t.shape[0]))
        queries = [unqueried_index[i] for i in queries_relative[0]]
        
        for i in range(len(m_unqueried)):
            self.mean_gaussian[unqueried_index[i]] = m_unqueried[i]   
        return queries

    

