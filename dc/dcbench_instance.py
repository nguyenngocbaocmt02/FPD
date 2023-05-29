import os
import yaml
import dcbench
import meerkat as mk
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


class DcbenchInstance:
    def __init__(self, instance_id, with_image=False):
        self.instance_id = instance_id
        self.with_image = with_image
        dcbench_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(dcbench_config_path, "r") as yaml_file:
            cfg = yaml.safe_load(yaml_file)
        dcbench_cfg = cfg["dcbench_config"]
        dcbench.config.local_dir = os.path.join(os.path.dirname(__file__), dcbench_cfg["local_dir"])
        dcbench.config.celeba_dir = os.path.join(os.path.dirname(__file__), dcbench_cfg["celeba_dir"])
        dcbench.config.imagenet_dir = os.path.join(os.path.dirname(__file__), dcbench_cfg["imagenet_dir"])
        dcbench.config.public_backet_name = dcbench_cfg["public_backet_name"]
        try:
            self.problem = dcbench.tasks["slice_discovery"].problems[instance_id]
        except:
            print("Try another instance_id")
        self.dp = mk.merge(self.problem["test_slices"], self.problem["test_predictions"], on="id")
        self.dp = mk.merge(self.problem["activations"], self.dp, on="id")
        if with_image:
            self.dp = mk.merge(self.problem["base_dataset"], self.dp, on="id")
        #self.dp = self.dp.lz" 
        
        # standardize the embs
        self.emb = np.array(self.dp["emb"])  # N x d
        self.emb = self.emb.reshape(self.emb.shape[0], -1)
        self.N = self.emb.shape[0]
        scalar = StandardScaler()
        # fitting
        scalar.fit(self.emb)
        self.emb = scalar.transform(self.emb)
        self.emb_distance_matrix = euclidean_distances(self.emb, self.emb)
        
        self.dp["emb"] = self.emb
        self.true_label = np.array(self.dp["target"])
        self.probs = np.array(self.dp["probs"])
        self.pseudo_label = np.argmax(np.array(self.dp["probs"]), axis=1)
        self.failure_id = np.array([i for i in range(self.N) if self.true_label[i] != self.pseudo_label[i]])
        if with_image:
            self.dp = copy.deepcopy(self.dp[["emb", "image", "probs"]])
        else:
            self.dp = copy.deepcopy(self.dp[["emb", "probs"]])

    def create_instance(self, pattern_size, knn_num):
        with_image = self.with_image
        adj_list = np.argpartition(self.emb_distance_matrix, knn_num + 1)[:, 0:knn_num + 1]
        adj_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            adj_matrix[np.ix_([i], adj_list[i])] += 1
            adj_matrix[np.ix_(adj_list[i], [i])] += 1
        adj_matrix = adj_matrix > 1

        failure_adj_matrix = adj_matrix[np.ix_(self.failure_id, self.failure_id)]
        graph = Graph(failure_adj_matrix)
        failure_patterns = []
        cc = graph.connectedComponents()
        for pattern in cc:
            if len(pattern) < pattern_size:
                continue
            failure_patterns.append(self.failure_id[pattern])
        pattern_mask = np.array([-1 for _ in range(self.N)])
        for i, pattern in enumerate(failure_patterns):
            for sample in pattern:
                pattern_mask[sample] = i

        cnt_signal = 0
        cnt_noise = 0
        for i, pattern in enumerate(pattern_mask):
            if pattern == -1 and self.pseudo_label[i] != self.true_label[i]:
                cnt_noise += 1
            if pattern != -1:
                cnt_signal += 1
        if with_image:
            dp = mk.DataPanel(
            {
                "emb": copy.deepcopy(self.emb),
                "image": copy.deepcopy(self.dp["image"]),
                "true_label": copy.deepcopy(self.true_label),
                "pseudo_label": copy.deepcopy(self.pseudo_label),
                "probs" : copy.deepcopy(self.probs),
                "pattern": pattern_mask
            }
            )
        else:
            dp = mk.DataPanel(
            {
                "emb": copy.deepcopy(self.emb),
                "true_label": copy.deepcopy(self.true_label),
                "pseudo_label": copy.deepcopy(self.pseudo_label),
                "probs" : copy.deepcopy(self.probs),
                "pattern": pattern_mask
            }
            )                
        print(np.unique(dp["pattern"], return_counts=True))
        return dp, cnt_signal / cnt_noise

class Graph: 
    # init function to declare class variables
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.V = self.adj_matrix.shape[0]
        self.adj_list = [(np.where(self.adj_matrix[i] == True)) for i in range(self.V)]
 
    def DFSUtil(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj_list[v][0]:
            if visited[i] == False:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = [False for i in range(self.V)]
        cc = []

        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc