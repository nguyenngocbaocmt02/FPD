import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

class Instance:
    def __init__(self, dp, pattern_size, knn_num):
        self.pattern_size = pattern_size
        self.knn_num = knn_num
        self.emb = np.array(dp["emb"])  # N x d
        self.emb = self.emb.reshape(self.emb.shape[0], -1)
        self.N = self.emb.shape[0]
        scalar = StandardScaler()
        # fitting
        scalar.fit(self.emb)
        self.emb = scalar.transform(self.emb)
        self.emb_distance_matrix = euclidean_distances(self.emb, self.emb)

        # Pseudolable for each sample
        self.pseudo_label = dp["pseudo_label"]
        # Annotator
        self.true_label = dp["true_label"]
        self.pattern = dp["pattern"]
        self.misclassified_index = []
        for i, e in enumerate(self.emb):
            if self.pseudo_label[i] != self.true_label[i]:
                self.misclassified_index.append(i)
        self.misclassified_index = np.array(self.misclassified_index)
        
        self.adj_list = np.argpartition(self.emb_distance_matrix, self.knn_num + 1)[:, 0:self.knn_num + 1]
        self.adj_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.adj_matrix[np.ix_([i], self.adj_list[i])] += 1
            self.adj_matrix[np.ix_(self.adj_list[i], [i])] += 1

        self.adj_matrix = self.adj_matrix > 1
        num_patterns = len(np.unique(dp["pattern"])) - 1
        self.misclassified_patterns = [[] for _ in range(num_patterns)]
        for i, e in enumerate(self.emb):
            if int(dp["pattern"][i]) != -1:
                self.misclassified_patterns[int(dp["pattern"][i])].append(i)
 

